"""
================================================================================
LTFormer 模型训练主脚本
================================================================================
在项目流水线中的位置：【步骤 2 - 自监督模型训练】
整体数据流（标准 PyTorch 训练模板）：
  1. 读取配置：通过 @hydra.main 加载 config/config_train_ltformer.yaml；
  2. 数据加载：TripletDataset 每次返回 1 组 (a, p, n) 图像块，DataLoader 打包成 batch；
  3. 模型实例化：ltformer_d128() 输出 128 维 L2 归一化特征描述子向量；
  4. 损失函数：loss_factory("AdaptativeTripletLoss") 即论文提出的 LT Loss；
  5. 优化更新：使用 SGD(lr=0.001, momentum=0.9) 反向传播更新 LTFormer 权重；
  6. 权重保存：每轮覆盖保存 LTFormer.pth，并按 checkpoint_freq 备份带轮数编号的权重。
================================================================================
"""
import os
import logging
try:
    from _version import __version__
except ImportError:
    __version__ = "1.0.0"
import hydra
import json
from omegaconf import OmegaConf
import warnings
from torch.autograd import Variable

import statistics
import torch

import shutil
import time


if "CUDA_VISIBLE_DEVICES" not in os.environ and torch.cuda.device_count() > 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')


import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.models.ltformer_model import ltformer_d128
from src.datasets.train_triplet_loader import TripletDataset
from src.losses.triplet_loss_layers import loss_factory
from src.utils.path import get_cwd
from src.utils.restart import restart_from_checkpoint

HYDRA_FULL_ERROR=1

writer = SummaryWriter()
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
device = torch.device("cuda")
warnings.filterwarnings("ignore")

# print(torch.cuda.device_count())


@hydra.main(version_base=None, config_path="config", config_name="config_train_ltformer")
def main(cfg):

    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    output_dir = get_cwd()
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Export dir: {output_dir}")
    logger.info("Loading parameters from config file")
    data_dir_list = cfg.paths.train_data
    nb_epoch = cfg.params.nb_epoch
    batch_size = cfg.params.batch_size
    image_size = cfg.params.image_size
    initial_lr = cfg.params.lr
    margin_value = cfg.params.margin_value
    loss_weight = cfg.params.loss_weight

    # gathers all epoch losses 
    loss_list = []

    # 1. 构建训练数据集与多线程 DataLoader（每个样本含 anchor/positive/negative 3张图）
    dataset = TripletDataset(data_dir_list, image_size)
    num_workers = min(8, os.cpu_count() or 2)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

    # 2. 初始化共享权重的轻量化 Transformer 骨干网络（输出 128 维特征描述子）
    model = ltformer_d128()

    # 3. 通过工厂函数构造损失层（默认 AdaptativeTripletLoss，即论文中的 LT Loss）
    criterion = loss_factory(
        cfg.params.loss_layer,
        batch_size=batch_size,
        margin_value=margin_value,
        loss_weight=loss_weight,
    )

    # 4. 配置随机梯度下降优化器（SGD + 动量）
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=cfg.params.momentum,
        weight_decay=cfg.params.weight_decay
    )

    # 5. 若输出目录中已有同名 .pth 权重，则自动加载以实现断点续训
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(output_dir, f"{cfg.params.model}.pth"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]

    logger.info("Start Epochs ...")
    for epoch in range(start_epoch,nb_epoch):

        model.train()
        model.cuda()
        loss_epoch = []
        dist_positive_epoch = []
        dist_negative_epoch = []

        for (idx, data) in enumerate(train_loader):
            _, inputs = data
            input_var = torch.autograd.Variable(inputs.cuda())
            
            # 丢弃最后一个不满 batch_size 的残余批次（因损失层按固定 batch_size 划分索引）
            if not (list(input_var.size())[0] == batch_size):
                continue

            # 形状变形：[B, 3, H, W] -> [B*3, 1, H, W]，让 3 组 Patch 一起前向传播
            inputs_var_batch = input_var.view(batch_size * 3, 1, image_size, image_size)

            # 前向传播提取描述子，输出形状：[B*3, 128]
            output = model(inputs_var_batch).to(device)
            # print(output.shape)

            # 计算三元组正/负样本距离与 LT Loss
            dist_positive, dist_negative, loss = criterion(output)
            if len(dist_positive) == 0 and len(dist_negative) == 0:
                continue

            # 记录当前批次的损失和正负样本平均距离用于日志监控
            loss_epoch.append(loss.item())
            dist_positive_epoch.append(dist_positive[0].item())
            dist_negative_epoch.append(dist_negative[0].item())

            # 标准反向传播三件套：清空历史梯度 -> 反向求导 -> 优化器更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(prof)
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss
        }
        torch.save(
            save_dict,
            os.path.join(output_dir, f"{cfg.params.model}.pth"),
        )
        if epoch % cfg.params.checkpoint_freq == 0 or epoch == nb_epoch - 1:
            shutil.copyfile(
                os.path.join(output_dir, f"{cfg.params.model}.pth"),
                os.path.join(output_dir, f"{cfg.params.model}_" + str(epoch) + ".pth"),
            )
        loss_list.append(statistics.mean(loss_epoch))
        mean_dist_positive = statistics.mean(dist_positive_epoch)
        mean_dist_negative = statistics.mean(dist_negative_epoch)

        logger.info(f"Epoch= {epoch:04d}  Loss= {statistics.mean(loss_epoch):0.4f}\
        Mean-Dist-Pos: {mean_dist_positive:0.4f}\
        Mean-Dist-Neg: {mean_dist_negative:0.4f}")
        writer.add_scalar("loss", statistics.mean(loss_epoch), epoch)


    checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss}
    checkpoint_export_path = os.path.join(output_dir, f"{cfg.params.model}.pth")
    torch.save(checkpoint, checkpoint_export_path)
    logger.info(f"Checkpoint savec to: {checkpoint_export_path}")


if __name__ == "__main__":
    main()
