"""
================================================================================
对比基线模型：HyNet（基于 FRN + TLU 层的局部特征描述子）
================================================================================
结构特点：用 Filter Response Normalization (FRN) 和 Thresholded Linear Unit (TLU)
         替代传统的 BatchNorm + ReLU，消除对批大小的依赖并提升特征判别力。
================================================================================
"""
import torch
"""
================================================================================
对比基线模型：HyNet（基于 FRN + TLU 层的局部特征描述子）
================================================================================
结构特点：用 Filter Response Normalization (FRN) 和 Thresholded Linear Unit (TLU)
         替代传统的 BatchNorm + ReLU，消除对批大小的依赖并提升特征判别力。
================================================================================
"""
import torch.nn as nn

EPS_L2_NORM = 1e-10


def desc_l2norm(desc):
    """将形状为 [N, C] 或 [N, C, H, W] 的描述子展平并做 L2 单位化。"""

    desc = desc.view(desc.size(0),-1)
    desc = desc / desc.pow(2).sum(dim=1, keepdim=True).add(EPS_L2_NORM).pow(0.5)
    return desc


class FRN(nn.Module):
    """滤波响应归一化层（Filter Response Normalization）：消除 BatchNorm 对批大小的依赖。"""

    def __init__(
        self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False
    ):
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        # 可学习的逐通道缩放参数 gamma (weight) 与平移参数 beta (bias)
        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True
        )
        self.bias = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True
        )
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        """初始化缩放为 1、偏置为 0。"""
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return "num_features={num_features}, eps={init_eps}".format(**self.__dict__)

    def forward(self, x):
        # 1. 对每个样本的每个通道独立计算空间二阶矩均值 nu^2（不做减均值）
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        nu2 = nu2.to(device=torch.device('cuda'))
        eps = self.eps.abs()
        eps = eps.to(device=torch.device('cuda'))
        # 2. 除以均方根完成归一化：x / sqrt(nu^2 + eps)
        x = x * torch.rsqrt(nu2 + eps)
        x = x.to(device=torch.device('cuda'))

        # 3. 逐通道仿射变换：y = gamma * x + beta
        if self.is_scale:
            self_weight = self.weight
            self_weight = self_weight.to(device=torch.device('cuda'))
            x = self_weight * x
        if self.is_bias:
            self_bias = self.bias
            self_bias = self_bias.to(device=torch.device('cuda'))
            x = x + self_bias
        return x


class TLU(nn.Module):
    """阈值线性单元（Thresholded Linear Unit）：配合 FRN 使用的可学习阈值激活 max(x, tau)。"""

    def __init__(self, num_features):
        super(TLU, self).__init__()
        self.num_features = num_features
        # 每个通道学习一个独立阈值 tau
        self.tau = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        """将可学习阈值 tau 初始化为 -1。"""
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return "num_features={num_features}".format(**self.__dict__)

    def forward(self, x):
        # 截断低于阈值 tau 的激活值
        tau = self.tau
        tau = tau.to(device=torch.device('cuda'))
        return torch.max(x, tau)


class HyNet(nn.Module):
    """HyNet model definition（对比基线：FRN + TLU + 7层全卷积局部描述子）"""

    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        # 首层先对输入单通道灰度 Patch 做 FRN+TLU，等价于可学习的输入标准化
        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        # 中间 4 层通过 stride=2 逐级降采样（共 16 倍）
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3,stride=2, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        # 末尾 8x8 卷积直出 128 维特征
        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False),
        )
    def forward(self, x, mode="eval"):
        """依次通过 7 个卷积层提取 128 维特征并做 L2 归一化；训练模式额外返回归一化前原始向量。"""
        for layer in [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
        ]:
            x = layer(x)
        desc_raw = self.layer7(x).squeeze()
        desc_raw = desc_raw.view(desc_raw.size(0),-1)

        desc = desc_l2norm(desc_raw)

        if mode == "train":
            return desc, desc_raw
        elif mode == "eval":
            return desc
