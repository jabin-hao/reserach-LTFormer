"""
================================================================================
对比基线模型：HardNet128（基于全卷积神经网络的局部特征描述子）
================================================================================
结构特点：纯 2D 卷积 + BatchNorm(无仿射参数) + ReLU + 末尾 8x8 卷积直出 128 维 + L2 归一化。
================================================================================
"""
import torch.nn as nn
import torch


class HardNet128(nn.Module):
    """
    proposed model based on HardNet model
    tested in the Self-Supervised-Endoscopic-Image-Key-Points-Matching article
    """

    def __init__(self):
        super(HardNet128, self).__init__()
        # 7 层全卷积特征提取主干：输入 [B, 1, 128, 128] -> 输出 [B, 128, 1, 1]
        # 注：若输入尺寸为 128x128，经 4 次 stride=2 降采样后变为 8x8，最后被 8x8 卷积压缩成 1x1
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),   # 1/2
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),   # 1/4
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),   # 1/8
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 1/16 -> 8x8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),                      # 8x8 -> 1x1x128
            nn.BatchNorm2d(128, affine=False),
        )

        self.features.apply(self.weights_init)
        return

    def input_norm(self, x):
        """逐 Patch 零均值单位方差归一化，减小图像明暗变化影响。"""
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        """前向计算：Patch 标准化 -> 卷积提取 128 维向量 -> 展平 -> L2 归一化。"""
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)  # [B, 128, 1, 1] -> [B, 128]
        return L2Norm()(x)

    @staticmethod
    def weights_init(m):
        """使用正交初始化（orthogonal）稳定深层卷积梯度。"""
        try:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal(m.weight.data, gain=0.6)
                if m.bias:
                    nn.init.constant(m.bias.data, 0.01)
        except Exception as e:
            print(str(e))
            pass
        return


class L2Norm(nn.Module):
    """L2 归一化层：将描述子向量投影到单位超球面上。"""
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)  # 计算每个样本向量的 L2 模长
        x = x / norm.unsqueeze(-1).expand_as(x)                # 逐通道除以模长完成归一化
        return x
