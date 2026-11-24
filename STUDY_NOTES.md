# LTFormer 与图像配准学习总结笔记

> 整理自 `article-LTFormer` 论文阅读与源码学习过程，涵盖**图像配准技术流派与 Loss 总结**、**项目分层架构**以及**核心概念速查**。

---

## 一、图像配准四大技术流派与常用 Loss 对照

并不是所有图像配准论文都使用 `HardNetLoss / LT Loss` 这类三元组损失，具体用什么 Loss 取决于方法走的技术路线：

| 技术流派 | 代表方法 | 模型输出 | 常用的核心 Loss |
| :--- | :--- | :--- | :--- |
| **1. 局部特征描述子派** | **LTFormer**、HardNet、SOSNet、HyNet | 局部图像块的特征向量（如 128 维） | **度量/对比学习 Loss**：Triplet Loss、HardNetLoss、LT Loss、对比损失（InfoNCE / Contrastive Loss） |
| **2. 稀疏图匹配 / 变换回归派** | SuperGlue、LightGlue、单应矩阵估计网络 | 点对匹配矩阵 / 仿射或单应变换矩阵 | **交叉熵 / 负对数似然 (NLL)**（把匹配看成指派分类）、**4角点坐标 L1/L2 回归误差** |
| **3. 密集光流 / 位移场派** | FlowNet2、**LiteFlowNet** | 全图逐像素二维位移场 $(\Delta x, \Delta y)$ | **EPE Loss**（预测位移与真实位移的欧氏距离）、**光度一致性误差 / SSIM** |
| **4. 医学 / 异源端到端形变配准** | VoxelMorph 等 | 非刚性形变场 + 重采样对齐图 | **互信息 (MI / NMI)**、**局部归一化互相关 (LNCC)**（不比像素亮度，比统计相关性）+ **形变场平滑正则项** |

**核心规律**：
* 只有**“提取特征向量做相似度匹配”**的路线，才常用 Triplet / Margin / InfoNCE 等度量学习 Loss；
* 在该领域，Loss 设计往往决定高维超球面空间的距离分布，因此经常是论文最核心的创新点之一（例如 HardNet 的批内最难负样本挖掘、LTFormer 的自适应动态边界 `LT Loss`）。

---

## 二、HardNetLoss vs LT Loss（`AdaptativeTripletLoss`）

### 1. HardNetLoss（NeurIPS 2017）
* **解决的痛点**：传统 Triplet Loss 随机抽取负样本大多“太容易区分”，导致 `loss = 0`、梯度消失。
* **核心做法**：
  1. 在一个 batch（如 `256` 对正样本）内计算 $256 \times 256$ 成对欧氏距离矩阵；
  2. 屏蔽对角线（自身正样本）与极近重复块（`< 0.008`）；
  3. 同时在**跨模态（a vs p）**与**同模态内部（a vs a、p vs p）**中挑出距离最近的**最难负样本 `min_neg`**；
  4. 用固定常数 `margin` 优化：$\max(0,\; \text{margin} + \text{pos} - \text{min\_neg})$。

### 2. LT Loss（`AdaptativeTripletLoss`，本文提出）
* **针对场景**：可见光与近红外等异源遥感图像，跨模态差异大且不均匀，固定 `margin` 不够灵活。
* **核心做法**：
  1. 计算锚点与正样本距离 `dist_p`、锚点与负样本距离 `dist_n`；
  2. 使用**动态自适应边界**：`margin = |dist_p + dist_n| / 2`；
  3. 优化：$\max(0,\; \text{dist\_p} + \text{margin} - \text{dist\_n})$。

---

## 三、LTFormer 项目结构与三段式流水线

本项目是典型的 PyTorch 视觉科研仓库结构（**配置与代码解耦 + 功能模块化**）：

```text
article-LTFormer/
├── config/                        # Hydra YAML 配置文件（路径与超参数集中管理）
│   ├── config_dataset.yaml
│   ├── config_train_ltformer.yaml
│   └── config_matching_ltformer.yaml
├── run_generate_dataset.py        # 【步骤 1】原版单进程三元组生成（已重构去重）
├── run_generate_dataset_fast.py   # 【步骤 1 加速版】多进程 + OpenCV + 分片断点续传
├── run_training_ltformer.py       # 【步骤 2】自监督训练入口
├── run_matching_demo.py           # 【步骤 3】推理匹配、指标评估与可视化
├── LTFormer.ipynb                 # Colab 完整复现脚本（CPU 造数据 + GPU 训练）
└── src/
    ├── datasets/                  # 只管读取磁盘数据（TripletDataset / PatchDataset）
    ├── models/                    # 网络架构 + arch_factory.py（工厂模式切换对比模型）
    ├── losses/                    # 度量学习损失 + loss_factory（工厂模式切换消融 Loss）
    └── utils/                     # 预处理、SIFT、匹配评估、Hydra 路径与断点续训工具
```

---

## 四、核心概念速查

1. **SIFT 与 128 维手工描述子**：
   * SIFT 是经典手工数学算法（不是训练出来的神经网络），本项目**只借用 SIFT 找关键点坐标**，丢弃其自带描述子。
   * SIFT 描述子为 128 维是因为：将关键点周围划分为 $4 \times 4 = 16$ 个子区域，每个子区域统计 $8$ 个方向的梯度直方图（$16 \times 8 = 128$）。深度描述子统一输出 128 维是为了与经典流水线对齐。
2. **图像增强（`5×5` 模糊 + LAB 空间 L 通道 `CLAHE`）**：
   * 专为低对比度遥感图像与 SIFT 检点设计，能压制噪声并拉开局部边缘纹理对比度；并非所有 CV 任务都需要（如自然图像分类/超分通常不用 CLAHE）。
3. **自监督三元组 `(a, p, n)` 的切法**：
   * 先将 `5556×3704` 原始大图均分为 `926×926` 网格（恰好 $6 \times 4 = 24$ 块）；
   * 再围绕每张 `926×926` 中的 SIFT 关键点抠出 `256×256` 局部块（训练时缩放到 `128×128`）：
     * `a.png`：可见光原图关键点 `b`；
     * `p.png`：近红外图经随机旋转/缩放/平移后，在对应新坐标处抠出的同一地点；
     * `n.png`：近红外原图在另一个不同关键点 `b+1` 处抠出的无关地点。
4. **PyTorch `nn.Module` 与自动求导黄金三行**：
   * `__init__` 负责声明与注册可训练层（只执行一次）；`forward` 负责定义前向数据流（调用 `model(x)` 时自动触发）。
   * 反向传播固定三步：`optimizer.zero_grad()`（清零旧梯度）$\rightarrow$ `loss.backward()`（自动求导）$\rightarrow$ `optimizer.step()`（更新权重）；推理时用 `with torch.no_grad():` 关闭计算图记账以节省显存。
