"""
================================================================================
Checkpoint 断点续训加载工具
================================================================================
作用：训练开始前检查输出目录是否已有上次保存的 .pth 权重文件；若有则恢复模型权重、
      优化器状态（如 SGD 动量缓存）以及当前 epoch 轮次，实现中断后继续训练。
================================================================================
"""
import argparse
import os
import pickle
from logging import getLogger
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from munkres import Munkres
import warnings
import json

logger = getLogger(__name__)


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """Re-start from checkpoint.（从已保存的 .pth 断点恢复模型、优化器状态与轮次计数）"""
    # 支持传入单个路径或候选路径列表
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    # 若尚无历史权重文件，直接从第 0 轮开始新训练
    if not os.path.isfile(ckp_path):
        return

    logger.info(f'Found checkpoint at {ckp_path}')

    # 读取断点字典（内含 epoch / state_dict / optimizer / loss 等）
    checkpoint = torch.load(ckp_path)

    # 遍历传入的对象（如 state_dict=model, optimizer=optimizer）逐一加载参数
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(
                key, ckp_path))
        else:
            logger.warning("=> failed to load {} from checkpoint '{}'".format(
                key, ckp_path))

    # 恢复标量运行变量（例如上次完成的 epoch 编号）
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]