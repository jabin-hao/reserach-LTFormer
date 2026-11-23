"""
================================================================================
Hydra 工作目录辅助工具
================================================================================
作用：Hydra 默认会把工作目录切换到 outputs/日期/时间/，get_cwd() 用于记录该输出路径
      （供保存训练日志和 .pth 权重），同时切回项目原始根目录，防止后续相对路径读取出错。
================================================================================
"""
import hydra
import os


def get_cwd():
    """
    custom function to get the current hydra output directory while keeping the original working directory
    自定义函数，用于获取当前 Hydra 输出目录，同时保留原始工作目录
    """
    original_cwd = hydra.utils.get_original_cwd()  # 获取启动 python 命令时的项目根目录
    cwd_dir = os.getcwd()                          # 获取 Hydra 当前切换到的实验输出目录 (outputs/...)
    os.chdir(original_cwd)                         # 切回项目根目录，确保配置里的相对路径仍有效
    return cwd_dir                                 # 返回输出目录供保存模型和日志
