"""
================================================================================
早期辅助损失类文件（保留兼容用，实际训练默认调用 triplet_loss_layers.py）
================================================================================
"""
import torch
import numpy as np


class HardNnetLoss(torch.nn.Module):
    """早期 HardNet 损失封装草稿（注意类名拼写，实际训练请见 triplet_loss_layers.py）。"""

    def __init__(self, anchorIds, positiveIds, negativeIds, lossweight=0.1, margin=0.2):
        torch.nn.Module.__init__(self)
        # 记录一个 batch 内锚点、正样本、负样本各自对应的切片索引
        self.anchorIds = anchorIds
        self.positiveIds = positiveIds
        self.negativeIds = negativeIds
        self.lenpositive = len(self.positiveIds)
        self.lennegative = len(self.negativeIds)
        self.weight = lossweight
        self.margin = margin
        self.labelsMiniBatch = None
        self.sizeMiniBatch = len(anchorIds)
        assert len(anchorIds) == len(positiveIds) and len(positiveIds) == len(
            negativeIds
        )

    def getWeight(self):
        """返回损失权重系数。"""
        return self.weight

    def setActualLabels(self, newLabel):
        """设置当前小批次标签缓存。"""
        self.labelsMiniBatch = newLabel
        return

    def get_label_batch(label_data, batch_size, batch_index):
        """按批次索引环形切片获取一个 batch 的标签数组。"""
        nrof_examples = np.size(label_data, 0)
        j = batch_index * batch_size % nrof_examples

        # 未越过数据集末尾直接切片；若越界则首尾拼接补足一个 batch
        if j + batch_size <= nrof_examples:
            batch = label_data[j : j + batch_size]
        else:
            x1 = label_data[j:nrof_examples]
            x2 = label_data[0 : nrof_examples - j]
            batch = np.vstack([x1, x2])
        batch_int = batch.astype(np.int64)

        return batch_int
