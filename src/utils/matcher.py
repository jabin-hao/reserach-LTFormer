"""
================================================================================
推理阶段特征提取、最近邻匹配与定量指标评估工具模块
================================================================================
核心函数说明：
  - feature_extraction()：批量将局部 Patch 送入模型提取 128 维描述子向量；
  - feature_match()     ：计算两组描述子两两欧氏距离，为左图每个点找右图距离最近的匹配点；
  - evaluate_matches()  ：根据距离阈值与像素位置误差统计 Precision 和 Matching Score 所需计数。
================================================================================
"""
import torch
from scipy.spatial import distance
import numpy as np
from math import sqrt


def evaluate_matches(
    gt_keypoint_src,
    gt_keypoint_dst,
    matches,
    distance_matching_threshold,
    distance_list,
    matching_threshold,
):
    """
    统计匹配评估指标（对应论文 Table 2 ~ Table 7）：
      - nb_rejected_matches ：描述子欧氏距离 > matching_threshold（被拒绝的点对）
      - nb_true_matches     ：未被拒绝 且 预测点与真值坐标空间误差 <= distance_matching_threshold
      - nb_false_matching   ：未被拒绝 但 空间误差超标（误匹配）
    """
    nb_false_matching = 0
    nb_true_matches = 0
    nb_rejected_matches = 0

    for j in range(0, len(gt_keypoint_src)):
        # gt_keypoint_dst[j] 是第 j 个源点的真实匹配坐标；gt_keypoint_dst[matches[j]] 是模型预测出的匹配坐标
        xp = int(gt_keypoint_dst[j][0])
        yp = int(gt_keypoint_dst[j][1])
        xp2 = int(gt_keypoint_dst[matches[j]][0])
        yp2 = int(gt_keypoint_dst[matches[j]][1])
        dist = sqrt((yp - yp2) ** 2 + (xp - xp2) ** 2)  # 空间定位误差（单位：像素）

        if distance_list[j] > matching_threshold:        # 1. 描述子距离太大，拒绝该匹配
            nb_rejected_matches += 1
        elif dist <= distance_matching_threshold:        # 2. 接受匹配且空间位置误差达标 -> 正确匹配
            nb_true_matches += 1
        else:                                            # 3. 接受匹配但实际找错了点 -> 误匹配
            nb_false_matching += 1
    return nb_false_matching, nb_true_matches, nb_rejected_matches


def feature_extraction(patches, model, imageSize):
    """推理提特征：逐个 Patch 转为 [1, 1, imageSize, imageSize] 送入模型并转回 NumPy 描述子列表。"""
    list_desc = []
    model.eval()

    inputs1_tensors = torch.stack([torch.Tensor(i) for i in patches])

    for (idx, data) in enumerate(inputs1_tensors):

        inputs = data
        input_var = torch.autograd.Variable(inputs)
        input_var = input_var.view(1, imageSize, imageSize)

        if not (list(input_var.size())[0] == 1):
            continue

        # 凑齐 4D 张量 [B=1, C=1, H, W] 送入 GPU 模型推理（关闭梯度以节省显存）
        inputs_var_batch = input_var.view(1, 1, imageSize, imageSize).to('cuda')
        with torch.no_grad():
            desc = model(inputs_var_batch).cuda()
        desc = torch.Tensor.cpu(desc).detach().numpy()
        list_desc.append(desc)

    return list_desc


def feature_match(feature_vectors_1, feature_vectors_2, matching_threshold):
    """暴力最近邻匹配：计算两组描述子之间的两两欧氏距离，返回最小距离索引及距离值。"""
    matched_idx = []
    distance_list = []
    for i in range(0, len(feature_vectors_1)):
        distance_sim = []
        for j in range(0, len(feature_vectors_2)):
            sim = distance.euclidean(feature_vectors_1[i].flatten(), feature_vectors_2[j].flatten()) #欧氏距离
            distance_sim.append(sim)
        # 按距离从小到大排序，取最近邻（索引 0）作为匹配结果
        candidate_index = np.argsort(distance_sim)[:2] #返回的是元素值从小到大排序后的索引值的数组
        distance_list.extend([distance_sim[candidate_index[0]]])
        matched_idx.extend([candidate_index[0]])

    return matched_idx, distance_list
