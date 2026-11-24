"""
================================================================================
LTFormer 核心网络架构实现（基于 PVTv2 改造的轻量级金字塔 Vision Transformer）
================================================================================
阅读顺序（已按“从外到内”的调用层级排列，建议从上往下顺读）：
  1. 快捷入口函数：ltformer_d128 / ltformer_d256 / ltformer_d512
  2. 整网总装主干：PyramidVisionTransformerV2（input_norm -> 4个Stage -> 池化 -> L2Norm）
  3. Stage 核心部件：OverlapPatchEmbed（降采样嵌入） -> Block（编码块）
  4. Block 内部子层：Attention（空间缩减自注意力） -> Mlp（前馈网络）
  5. 底层原子组件：DWConv（3x3深度卷积） -> L2Norm（L2单位化）
================================================================================
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


# ==============================================================================
# 1. 模型入口函数（按论文 Table 1 / Table 7 配置实例化主干网络）
# ==============================================================================

@register_model
def ltformer_d128(pretrained=False, **kwargs):
    """论文默认配置（对应 Table 1）：输出 128 维特征描述子。"""
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[16, 32, 64, 128], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def ltformer_d256(pretrained=False, **kwargs):
    """256 维描述子消融实验变体（对应 Table 7）。"""
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 128, 256], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def ltformer_d512(pretrained=False, **kwargs):
    """512 维描述子扩展变体。"""
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


# ==============================================================================
# 2. 整网总装主干：串联 input_norm -> 4 个金字塔 Stage -> 全局池化 -> L2Norm
# ==============================================================================

class PyramidVisionTransformerV2(nn.Module):
    """4 级金字塔 Vision Transformer V2 主干（按 Table 1 堆叠各阶段模块）。"""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=128, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        # 为所有层生成线性递增的随机深度丢弃率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # 循环构建 Stage 1 ~ Stage 4（每个 Stage = 1个重叠卷积嵌入 + L_i 个 Transformer 块 + 归一化层）
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            # 动态注册属性名 patch_embed1~4 / block1~4 / norm1~4
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # 注：描述子提取任务不需要分类头，直接对最后阶段 Token 平均池化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化全连接层、LayerNorm 与卷积层权重。"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        """冻结第一级块嵌入层参数（微调时可选）。"""
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        """指定不参与权重衰减（weight decay）的参数名集合。"""
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        """获取分类头（如已启用）。"""
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """重置分类头输出维度。"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """依次通过 4 个金字塔 Stage 提取多尺度特征，最后全局平均池化输出 [B, embed_dims[-1]]。"""
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)               # 卷积下采样并展平为序列 [B, H*W, C]
            for blk in block:
                x = blk(x, H, W)                   # 经过 L_i 个轻量级 Transformer Block
            x = norm(x)
            if i != self.num_stages - 1:
                # 前 3 个 Stage 结束后还原回二维特征图 [B, C, H, W]，送入下一级卷积嵌入
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 论文核心改动：去掉分类头，直接对序列 Token 维取平均得到紧凑描述子向量
        return x.mean(dim=1)

    def input_norm(self, x):
        """逐样本（per-patch）标准化：减弱可见光与近红外图像的整体灰度偏移与对比度差异。"""
        flat = x.view(x.size(0), -1)               # 展平为 [B, C*H*W] 以便求整块统计量
        mp = torch.mean(flat, dim=1)               # 每个 Patch 的像素均值 [B]
        sp = torch.std(flat, dim=1) + 1e-7         # 每个 Patch 的像素标准差 [B]（加 eps 防除零）
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x):
        """前向传播：Patch 标准化 -> 金字塔 Transformer 提取特征 -> L2 归一化。"""
        x = self.forward_features(self.input_norm(x))
        return L2Norm()(x)


# ==============================================================================
# 3. Stage 核心部件：OverlapPatchEmbed（切块降采样）与 Block（编码器块）
# ==============================================================================

class OverlapPatchEmbed(nn.Module):
    """
    重叠图像块嵌入（对应 Table 1 的 Overlapping Patch Embedding）：
    与原始 ViT 无重叠切块不同，这里要求 kernel_size > stride（如 7>4, 3>2），
    使相邻块之间保留像素重叠，从而更好地保持特征边界的连续性。
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)                        # 卷积降采样并升维：[B, in_chans, H_in, W_in] -> [B, C, H, W]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # 展平为 Transformer Token 序列：[B, H*W, C]
        x = self.norm(x)

        return x, H, W


class Block(nn.Module):
    """单个 Transformer 编码器块（对应 Table 1 的 Transformer Encoder，每级堆叠 L_i=2 层）。"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # 随机深度（DropPath）概率：随网络深度从 0 线性递增到 drop_path_rate
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Pre-Norm 残差结构：x + DropPath(Attention(LN(x))) -> x + DropPath(MLP(LN(x)))
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


# ==============================================================================
# 4. Block 内部子层：Attention（空间缩减注意力）与 Mlp（卷积增强前馈网络）
# ==============================================================================

class Attention(nn.Module):
    """
    空间缩减多头自注意力（Spatial-Reduction Attention, SRA）：
    这是 LTFormer 实现“轻量级”的核心——计算 Attention 时 Query 保持全分辨率 N=H*W，
    而 Key 和 Value 先经过 sr_ratio 倍步长卷积压缩为 (H/R)*(W/R) 个 Token，
    从而将注意力矩阵复杂度从 O(N^2) 降至 O(N^2 / R^2)。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5               # 缩放因子 1/sqrt(d_k)，防止点积过大梯度消失

        self.q = nn.Linear(dim, dim, bias=qkv_bias)            # Query 映射矩阵
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)       # Key/Value 联合映射矩阵
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)                        # 多头拼接后的输出投影
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                # 用卷积核=步长=sr_ratio 的 2D 卷积对空间尺寸做无重叠压缩
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Q 保持完整序列长度 N = H*W，拆成多头: [B, num_heads, N, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                # 先从 [B, N, C] 还原成 2D 特征图 [B, C, H, W]，卷积缩减后再展平做 LayerNorm
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                # Stage 4 分辨率已经很小 (4x4)，sr_ratio=1，直接用原序列算 K/V
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 计算缩放点积注意力权重: [B, num_heads, N, N_sr]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权求和并拼回多头 -> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    """
    改进版前馈网络（Convolutional FFN）：
    在两层线性变换 (fc1 -> fc2) 中间嵌入 3x3 深度可分离卷积 DWConv，
    用于捕捉二维局部相邻像素关系，使 PVTv2 无需显式位置编码也能感知空间位置。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)     # 通道升维（扩展 E_i 倍）
        self.dwconv = DWConv(hidden_features)                  # 3x3 深度卷积补充局部空间信息
        self.act = act_layer()                                 # 默认 GELU 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)    # 通道降回原维度
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """按层类型初始化权重：Linear 用截断正态，LayerNorm 置 1/0，Conv2d 用 Kaiming 方差。"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)                               # 需要还原回 [B, C, H, W] 做 2D 卷积再展平
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ==============================================================================
# 5. 底层原子组件：DWConv（被 Mlp 调用）与 L2Norm（被主干末尾调用）
# ==============================================================================

class DWConv(nn.Module):
    """深度可分离卷积（groups=dim）：每个通道独立做 3x3 空间卷积，参数极少且能注入位置信息。"""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, H*W, C] -> [B, C, H, W]
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)        # [B, C, H, W] -> [B, H*W, C]

        return x


class L2Norm(nn.Module):
    """L2 归一化层：将每个描述子向量除以自身的 L2 范数，约束在单位超球面上便于欧氏距离度量。"""
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)  # 计算每个样本向量的 L2 模长 [B]
        x = x / norm.unsqueeze(-1).expand_as(x)                # 按通道逐元素归一化为单位向量
        return x
