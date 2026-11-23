"""
================================================================================
模型工厂模块（Factory Pattern）
================================================================================
作用：根据配置中的模型名称字符串（model_name）和权重文件路径（model_weights_path），
      统一实例化对应网络并加载权重，方便在评估脚本中一键切换对比不同模型。
注意：不同作者开源的 .pth 权重字典格式略有差异：
  - HyNet / SOSNet 直接保存了 state_dict；
  - HardNet128 / SwinTransformer / LTFormer 将权重包在 checkpoint["state_dict"] 键下。
================================================================================
"""
import os.path as osp
import torch

from .hardnet_model import HardNet128
from .hynet_model import HyNet
from .sosnet_model import SOSNet
from .swim_model import SwinTransformer
from .ltformer_model import ltformer_d128

ACCEPTED_MODEL_NAMES = ["HyNet", "SOSNet", "HardNet128","SwinTransformer", "LTFormer"]


def model_factory(model_name, model_weights_path):
    """根据 model_name 创建网络结构并加载预训练权重。"""
    assert model_name in ACCEPTED_MODEL_NAMES
    assert osp.exists(model_weights_path)

    # 读取序列化权重文件
    state_dict = torch.load(model_weights_path)
    # 不同模型保存权重的字典结构不同：HyNet/SOSNet 为裸参数字典，其余包在 state_dict 键下
    if model_name == "HyNet":
        model = HyNet()
        model.load_state_dict(state_dict)
    elif model_name == "SOSNet":
        model = SOSNet()
        model.load_state_dict(state_dict)
    elif model_name == "HardNet128":
        model = HardNet128()
        model.load_state_dict(state_dict["state_dict"])
    elif model_name == "SwinTransformer":
        model = SwinTransformer()
        model.load_state_dict(state_dict["state_dict"])
    elif model_name == 'LTFormer':
        model = ltformer_d128()
        model.load_state_dict(state_dict["state_dict"])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model
