"""
================================================================================
三元组度量学习损失函数模块（工厂模式 + HardNetLoss + LT Loss）
================================================================================
核心设计模式：【工厂模式 (Factory Pattern)】
  通过 loss_factory(loss_id, ...) 根据配置文件字符串动态实例化不同损失层，
  方便做表 6 中的损失函数消融实验（对比 HardNetLoss 与 AdaptativeTripletLoss）。

包含两种核心损失：
  1. HardNetLoss：在每个 batch 中寻找最难负样本（距离最近的非匹配点）计算边界损失；
  2. AdaptativeTripletLoss（论文提出的 LT Loss）：利用动态边界 margin = |d_p + d_n| / 2
     自适应调整正负样本间隔，进一步拉近同一地物的跨模态描述子、推远异地物描述子。
================================================================================
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


LOSS_LAYER_IDS = ["HardNetLoss", "AdaptativeTripletLoss"]


def loss_factory(loss_id, batch_size, margin_value, loss_weight):
    """损失函数工厂：按名称字符串返回对应的 PyTorch nn.Module 损失层。"""
    if loss_id == "HardNetLoss":
        loss_layer = HardNetLoss(
            batch_size=batch_size,
            margin=margin_value,
            loss_weight=loss_weight,
        )
    elif loss_id == "AdaptativeTripletLoss":
        loss_layer = AdaptativeTripletLoss(
            batch_size=batch_size,
            margin=margin_value,
            lossweight=loss_weight,
        )

    return loss_layer


class HardNetLoss(torch.nn.Module):
    """HardNet 难负样本边界损失层包装类。"""

    def __init__(self, batch_size, loss_weight=0.1, margin=0.2):
        torch.nn.Module.__init__(self)

        # 按步长 3 生成一个 batch 内 anchor、positive、negative 各自的行索引
        self.anchor_ids = range(0, batch_size * 3, 3)
        self.positive_ids = range(1, batch_size * 3, 3)
        self.negative_ids = range(2, batch_size * 3, 3)

        self.nb_anchor_samples = len(self.anchor_ids)
        self.nb_positive_samples = len(self.positive_ids)
        self.nb_negative_samples = len(self.negative_ids)
        self.weight = loss_weight
        self.margin = margin
        self.labels_mini_batch = None
        self.sizeMiniBatch = len(self.anchor_ids)

        assert (
            self.nb_anchor_samples == self.nb_positive_samples
            and self.nb_positive_samples == self.nb_negative_samples
        )

    def get_weight(self):
        """返回损失权重系数。"""
        return self.weight

    def set_actual_labels(self, new_label):
        """更新当前批次标签缓存。"""
        self.labels_mini_batch = new_label
        return

    def get_label_batch(label_data, batch_size, batch_index):
        """按批次索引环形切片获取标签数据。"""
        nrof_examples = np.size(label_data, 0)
        j = batch_index * batch_size % nrof_examples

        if j + batch_size <= nrof_examples:
            batch = label_data[j : j + batch_size]
        else:
            x1 = label_data[j:nrof_examples]
            x2 = label_data[0 : nrof_examples - j]
            batch = np.vstack([x1, x2])
        batch_int = batch.astype(np.int64)

        return batch_int

    def forward(self, input):
        """取出 batch 内的 anchor 和 positive 描述子，送入 HardNet 难负样本损失。"""
        anchors = input[self.anchor_ids, :].cuda()
        positives = input[self.positive_ids, :].cuda()

        pos, min_neg, loss = loss_HardNet(
            anchors,
            positives,
            margin=self.margin,
            batch_reduce="min",
            loss_type="triplet_margin",
        )
        return pos, min_neg, loss

#计算距离矩阵
def distance_matrix_vector(anchor, positive):
    """
    利用矩阵乘法高效计算成对欧氏距离矩阵 D[i, j] = ||anchor[i] - positive[j]||_2。
    展开公式：||a - p||^2 = ||a||^2 + ||p||^2 - 2 * a^T p
    """

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1).cuda()
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1).cuda()

    eps = 1e-6
    flag =(d1_sq.repeat(1, positive.size(0)).cuda()
            + torch.t(d2_sq.repeat(1, anchor.size(0))).cuda()
            - 2.0
            * torch.bmm(anchor.unsqueeze(0).cuda(), torch.t(positive).unsqueeze(0).cuda()).squeeze(0).cuda())
    #避免负数出现（浮点精度误差可能导致微小负值，用 ReLU 截断为 0）
    torch.nn.functional.relu(flag, inplace=True).cuda()
    return torch.sqrt(flag + eps).cuda()


def loss_HardNet(
    anchor, positive, margin=1.0, batch_reduce="min", loss_type="triplet_margin"
):
    """
    HardNet 难负样本边界损失：
    对于每对正样本 (a_i, p_i)，在同批次所有非对应样本中搜索距离最近的“最难负样本”，
    最大化正样本距离与最难负样本距离之间的间隔。
    """

    assert (
        anchor.size() == positive.size()
    ), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Input must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # 对角线元素 dist_matrix[i, i] 即为第 i 对正样本的距离 pos
    pos1 = torch.diag(dist_matrix).cuda()

    # 给对角线（自身正样本）加上大惩罚 10，防止在找“最近负样本”时选中自己
    dist_without_min_on_diag = dist_matrix + eye * 10
    # 过滤距离 < 0.008 的极近样本，避免重复切块被误当成难负样本
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask

    # 跨模态找每个 anchor 对应的最近负样本距离
    min_neg = torch.min(dist_without_min_on_diag, 1)[0].cuda()

    # 同模态内部（anchor之间、positive之间）也寻找最难负样本，取三者最小值
    dist_matrix_a = distance_matrix_vector(anchor, anchor) + eps
    dist_matrix_p = distance_matrix_vector(positive, positive) + eps
    dist_without_min_on_diag_a = dist_matrix_a + eye * 10
    dist_without_min_on_diag_p = dist_matrix_p + eye * 10

    min_neg_a = torch.min(dist_without_min_on_diag_a, 1)[0].cuda()
    min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p, 0)[0]).cuda()
    min_neg_3 = torch.min(min_neg_p, min_neg_a).cuda()
    min_neg = torch.min(min_neg, min_neg_3).cuda()

    min_neg = min_neg
    pos = pos1

    # 三元组 Hinge 损失：max(0, margin + d_pos - d_min_neg)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0).cuda()
    elif loss_type == "softmax":
        exp_pos = torch.exp(2.0 - pos).cuda()
        exp_den = (exp_pos + torch.exp(2.0 - min_neg) + eps).cuda()
        loss = -torch.log(exp_pos / exp_den).cuda()
    elif loss_type == "contrastive":
        loss = (torch.clamp(margin - min_neg, min=0.0) + pos).cuda()

    loss = torch.mean(loss).cuda()
    return pos, min_neg, loss


def dist_triplet(self, input):
    """辅助函数：分别计算批内正样本对和负样本对的欧氏距离。"""
    def distance_pair(x1, x2, norm=2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2).cuda()
        out = torch.pow(diff, norm).sum(dim=1).cuda()
        return torch.pow(out + eps, 1.0 / norm).cuda()

    anchors = input[self.anchorIds, :]
    positives = input[self.positiveIds, :]
    negatives = input[self.negativeIds, :]
    dist_positive = distance_pair(anchors, positives, 2)
    dist_negative = distance_pair(anchors, negatives, 2)
    return (dist_positive, dist_negative)


class AdaptativeTripletLoss(torch.nn.Module):
    """
    论文提出的 LT Loss（自适应边界三元组损失）。
    输入形状为 [batch_size * 3, 128]，其中按每 3 个样本一组交错排列：
      - 索引 0, 3, 6, ... -> Anchor   (可见光锚点描述子)
      - 索引 1, 4, 7, ... -> Positive (近红外正样本描述子)
      - 索引 2, 5, 8, ... -> Negative (近红外负样本描述子)
    """
    def __init__(self, batch_size, lossweight=0.1, margin=0.2):
        torch.nn.Module.__init__(self)
        # 按步长 3 切分出 batch 中对应的 anchor / positive / negative 下标
        self.anchorIds = range(0, batch_size * 3, 3)
        self.positiveIds = range(1, batch_size * 3, 3)
        self.negativeIds = range(2, batch_size * 3, 3)
        self.lenpositive = len(self.positiveIds)
        self.lennegative = len(self.negativeIds)
        self.weight = lossweight
        self.margin = margin
        self.labelsMiniBatch = None
        self.sizeMiniBatch = len(self.anchorIds)
        assert len(self.anchorIds) == len(self.positiveIds) and len(self.positiveIds) == len(
            self.negativeIds
        )

    def getWeight(self):
        return self.weight

    def setActualLabels(self, newLabel):
        self.labelsMiniBatch = newLabel
        return

    def get_label_batch(label_data, batch_size, batch_index):

        nrof_examples = np.size(label_data, 0)
        j = batch_index * batch_size % nrof_examples

        if j + batch_size <= nrof_examples:
            batch = label_data[j : j + batch_size]
        else:
            x1 = label_data[j:nrof_examples]
            x2 = label_data[0 : nrof_examples - j]
            batch = np.vstack([x1, x2])
        batch_int = batch.astype(np.int64)

        return batch_int

    def forward(self, input):
        """
        计算 LT Loss：
          1. 计算正样本欧氏距离 d_p = ||a - p||_2 和负样本欧氏距离 d_n = ||a - n||_2；
          2. 根据三元组难易程度分组（默认 'all' 使用全部样本）；
          3. 自适应边界 margin = |dist_p + dist_n| / 2；
          4. 优化目标 max(0, dist_p + margin - dist_n)，促使 dist_p -> 0 且 dist_n > dist_p + margin。
        """
        margin_factor = 1
        filter_loss = False
        categories = ["all", "easy", "semihard", "hard"]
        triplet_categorie = categories[0]                      # 默认对 batch 内全部三元组计算损失
        loss = Variable(torch.zeros(1))
        zero = Variable(torch.zeros(1)).cuda()
        anchors = input[self.anchorIds, :]
        positives = input[self.positiveIds, :]
        negatives = input[self.negativeIds, :]

        # 计算锚点分别与正样本、负样本之间的 L2（欧氏）距离
        d_p = F.pairwise_distance(anchors, positives, 2)
        d_n = F.pairwise_distance(anchors, negatives, 2)

        # 动态边界（自适应 margin）：随当前正负样本距离均值变化
        self.margin = margin_factor * d_p
        self.margin = abs(d_p + d_n) / 2
        # 按三元组难度划分掩码（easy / semihard / hard）
        easy_samples = (d_p + self.margin < d_n).cpu().data.numpy().flatten()
        semihard_samples = (
            ((d_p < d_n) & (d_n < d_p + self.margin)).cpu().data.numpy().flatten()
        )
        hard_samples = (d_p > d_n).cpu().data.numpy().flatten()
        if triplet_categorie == "all":
            all = (d_p == d_p).cpu().data.numpy().flatten()
        if triplet_categorie == "easy":
            all = easy_samples
        if triplet_categorie == "semihard":
            # 半难样本：正样本距离小于负样本，但尚未拉开足够边界 (d_p < d_n < d_p + margin)
            all = semihard_samples
        if triplet_categorie == "hard":
            # 难样本：正样本反而比负样本更远 (d_n < d_p)
            all = hard_samples

        selected_triplets_ind = np.where(all == 1)
        if len(selected_triplets_ind[0]) == 0:
            return 0, [], []
        else:
            occ_selected_triplet = len(selected_triplets_ind[0])

        # 提取被选中的三元组样本重新计算距离与最终自适应边界损失
        all_a = anchors[selected_triplets_ind, :]
        all_p = positives[selected_triplets_ind, :]
        all_n = negatives[selected_triplets_ind, :]
        aa = all_a.view(occ_selected_triplet, -1)
        pp = all_p.view(occ_selected_triplet, -1)
        nn = all_n.view(occ_selected_triplet, -1)

        dist_p = F.pairwise_distance(aa, pp, 2).cuda()
        dist_n = F.pairwise_distance(aa, nn, 2).cuda()

        # 动态更新自适应边界 margin = |dist_p + dist_n| / 2
        self.margin = abs(dist_p + dist_n) / 2

        if triplet_categorie == "easy":
            loss2 = torch.max(zero, (dist_n - self.margin) - dist_p).cuda()
        else:
            loss2 = torch.max(
                zero, dist_p + self.margin.cuda() - dist_n).cuda()

        # 是否仅对非零激活的难三元组求平均损失
        if filter_loss:
            all = (loss2 != 0).cpu().data.numpy().flatten()
            valid_loss = np.where(all == 1)[0]
            valid_losses = [loss2[u] for u in valid_loss]
            loss = np.mean(valid_losses)
        else:
            loss = torch.mean(loss2).cuda()

        return dist_p, dist_n, loss
