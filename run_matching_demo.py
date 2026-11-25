"""
================================================================================
LTFormer 推理匹配与定量评估演示脚本
================================================================================
在项目流水线中的位置：【步骤 3 - 模型测试、指标评估与匹配可视化】
工作流程：
  1. 通过 model_factory 加载训练好的模型权重（如 LTFormer / HardNet / SOSNet / HyNet）；
  2. PatchDataset 读取测试图像对及真值匹配点坐标，裁剪两张图的关键点局部 Patch；
  3. feature_extraction() 调用模型把所有局部 Patch 编码为 128 维特征描述子；
  4. feature_match() 计算两组描述子的两两欧氏距离，寻找最近邻匹配点对；
  5. evaluate_matches() 统计 Precision（精准率）和 Matching Score（匹配得分），
     并在拼接图像上绘制绿线（正确匹配）与红线（错误匹配）保存可视化结果。
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
from tqdm import tqdm

import imageio

import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils.path import get_cwd
import numpy as np
import cv2
from math import sqrt


# 设置GPU卡号（仅在多卡环境且未外部指定时使用卡3，兼容Colab单卡）
if "CUDA_VISIBLE_DEVICES" not in os.environ and torch.cuda.device_count() > 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
writer = SummaryWriter()
logger = logging.getLogger("Demo_matching")
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import statistics

from src.utils.matcher import feature_match, feature_extraction, evaluate_matches
from src.datasets.test_patch_loader import PatchDataset
from src.models.arch_factory import model_factory


def export_gif(frame_path_list, out_gif_filename, fps=24):
    """将保存的匹配结果图片序列合成为 GIF 动图。"""
    frame_list = []
    for frame_path in tqdm(frame_path_list):
        frame_list.append(cv2.imread(frame_path))
    imageio.mimsave(out_gif_filename, frame_list, fps=fps)


@hydra.main(version_base=None, config_path="config", config_name="config_matching_ltformer")
def main(cfg):
    """Hydra 匹配评估主函数：读取测试集 -> 提取描述子 -> 最近邻匹配 -> 统计指标与绘图。"""
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    output_dir = get_cwd()
    logger.info(f"Working dir: {os.getcwd()}")
    logger.info(f"Export dir: {output_dir}")

    logger.info("Loading parameters from config file")
    validation_data_root = cfg.paths.demo_sequence_data
    model_name = cfg.params.model_name
    model_weights_path = cfg.params.weights_path
    image_size = cfg.params.image_size
    patch_size = cfg.params.patch_size
    distance_matching_threshold = cfg.params.distance_matching_threshold
    matching_threshold = cfg.params.matching_threshold
    use_blur = cfg.params.get("use_blur", False)

    # 1. 通过模型工厂加载指定架构与权重文件
    model = model_factory(model_name, model_weights_path)

    # 2. 构造测试集 Dataset（自动读取图像对、真值关键点并裁剪左右局部 Patch）
    test_dataset = PatchDataset(validation_data_root, patch_size, image_size=image_size, use_blur=use_blur)

    # 保存所有输出图片路径，便于后续合成 GIF
    frame_path_list = []

    # 用于汇总整个测试集的平均精确率与匹配得分
    precision = []
    matching_score = []

    # 切换评估模式（关闭 Dropout / 固定 BN 统计量）并搬移到 GPU
    model.eval()
    model.cuda()

    # 逐帧遍历测试图像对
    for id, data in enumerate(tqdm(test_dataset)):

        patch_src = data["patch_src"]
        patch_dst = data["patch_dst"]
        keypoint_src = data["keypoint_src"]
        keypoint_dst = data["keypoint_dst"]
        frame_src = data["ori_src"]
        Next_frame = data["ori_dst"]
        folder_name = data["image_name"]

        if not patch_src:
            continue

        # 3. 提取左右两张图所有局部 Patch 的 128 维特征描述子
        list_desc_src = feature_extraction(patch_src, model, image_size)
        list_desc_dst = feature_extraction(patch_dst, model, image_size)

        # 4. 计算描述子欧氏距离进行最近邻匹配
        matches, distance_list = feature_match(
            list_desc_src, list_desc_dst, matching_threshold
        )
        # 5. 根据距离阈值和空间坐标误差统计真匹配 / 误匹配 / 拒绝数
        nb_false_matching, nb_true_matches, nb_rejected_matches = evaluate_matches(
            keypoint_src,
            keypoint_dst,
            matches,
            distance_matching_threshold,
            distance_list,
            matching_threshold,
        )
        nb_received_matching = nb_false_matching + nb_true_matches
        if nb_received_matching <= 0:
            nb_received_matching = 0.1
        precision.append(nb_true_matches / (nb_received_matching))
        matching_score.append(
            nb_true_matches
            / (nb_received_matching + nb_rejected_matches)
        )

        h, w ,_= frame_src.shape

        # 6. 左右拼接原图并绘制匹配结果（绿线：正确匹配，红线：误差超标的误匹配）
        image_match = np.concatenate((frame_src, Next_frame), axis=1)
        image_match_rgb = cv2.cvtColor(image_match, 3)

        for i in range(0, len(keypoint_src)):
            # 取源点坐标 (xa, ya)、右图真值点 (xp, yp) 及模型预测匹配点 (x, y)
            xa = int(keypoint_src[i][0])
            ya = int(keypoint_src[i][1])
            xp = int(keypoint_dst[i][0])
            yp = int(keypoint_dst[i][1])
            x = int(keypoint_dst[matches[i]][0])
            y = int(keypoint_dst[matches[i]][1])
            dist = sqrt((yp - y) ** 2 + (xp - x) ** 2)

            # 在拼接图左右两侧画蓝色/红色圆点标出关键点位置（注意右图 x 需加上左图宽度 w）
            cv2.circle(
                image_match_rgb, (xa, ya), radius=2, color=(255, 0, 0), thickness=2
            )
            cv2.circle(
                image_match_rgb, (xp + w, yp), radius=2, color=(255, 0, 0), thickness=2
            )
            # 空间误差超出阈值画红线（误匹配），否则画绿线（正确匹配）
            if dist > distance_matching_threshold:
                cv2.line(image_match_rgb, (xa, ya), (x + w, y), (0, 0, 255), thickness=1)
            else:
                cv2.line(image_match_rgb, (xa, ya), (x + w, y), (0, 255, 0), thickness=1)


        if not os.path.exists(os.path.join(output_dir, folder_name)):
            os.makedirs(os.path.join(output_dir, folder_name))
        file_name = os.path.join(output_dir, folder_name, f"matches{id}_{id + 1}.png")
        frame_path_list.append(file_name)
        cv2.imwrite(file_name, cv2.resize(image_match_rgb, (0, 0), fx=0.6, fy=0.6))

    logger.info(f"Precision= {statistics.mean(precision):0.4f}")
    logger.info(f"Matching_score= {statistics.mean(matching_score):0.4f}")


if __name__ == "__main__":
    main()
