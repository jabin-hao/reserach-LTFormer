"""
================================================================================
自监督三元组数据集生成脚本（论文原版，单进程串行）
================================================================================
在项目流水线中的位置：【步骤 1 - 离线数据准备】
核心思路（为什么叫自监督）：
  1. 输入：预先空间对齐（像素级配准）的可见光(RGB)图像与近红外(NIRS)图像对；
  2. 选点：使用 SIFT 在可见光图像上提取稀疏关键点坐标；
  3. 造变换：对近红外图像随机施加已知的仿射/缩放变换（旋转 / 缩放 / 平移）；
  4. 切三元组：
     - a.png (Anchor 锚点)   ：可见光原图在关键点处切出的局部块；
     - p.png (Positive 正样本)：变换后近红外图在“同一地理位置的对应新坐标”切出的块；
     - n.png (Negative 负样本)：近红外原图在“另一个不同关键点”处切出的块。

     a.png（Anchor，锚点）：从可见光原图的关键点 b 处切出来；
     p.png（Positive，正样本）：从变换后的近红外图中，关键点 b 变换后的对应同一个地理位置切出来；
     n.png（Negative，负样本）：从近红外原图中，另一个不同关键点 b+1（代码里是取去重后的下一个关键点，代表不同地点）处切出来。

     这样训练时，网络就会被不断要求：
        让 a（可见光的某地）和 p（近红外+变形的同一地）的特征向量越接近越好，同时远离 n（近红外的另一地点）。

注意：若在 Colab 上处理 WHU-OPT-SAR 大图，推荐使用提速数十倍且支持断点续传的
      run_generate_dataset_fast.py。
================================================================================
"""
import glob
import os
import os.path as osp
import logging

try:
    from _version import __version__
except ImportError:
    __version__ = "1.0.0"
import hydra
import json
from omegaconf import OmegaConf

import cv2
from scipy.spatial import distance
import random
import numpy as np

from src.utils.image_keypoints_extractors import extract_image_keypoints, enhance_image_2

logger = logging.getLogger("Triplet-Dataset-Generator")
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config_dataset")
def main(cfg):
    """Hydra 入口函数：自动读取 config/config_dataset.yaml 注入 cfg 对象。"""
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)
    logger.info("Start Processing raw sequence")

    # 1. 遍历配置中的可见光与近红外序列目录
    input_rgb_image_lists = sorted(glob.glob(cfg.paths.rgb_data_dirs + "/*"))
    input_NIRS_image_lists = sorted(glob.glob(cfg.paths.NIRS_data_dirs + "/*"))

    assert input_rgb_image_lists
    assert input_NIRS_image_lists

    # 2. 逐个子序列、逐对图像进行处理
    idx = 0
    for image_path_list, image_path_NIRS_list in zip(input_rgb_image_lists, input_NIRS_image_lists):

        image_path_lists = sorted(glob.glob(image_path_list + "/*"))
        image_path_NIRS_lists = sorted(glob.glob(image_path_NIRS_list + "/*"))

        for image_path, image_path_NIRS in zip(image_path_lists, image_path_NIRS_lists):
            # 读取可见光(RGB)图像并做 LAB 空间 CLAHE 对比度增强
            # 遥感/医学影像/水下/传统特征点匹配 经常使用CLAHE对比度增强 因为更关注局部边缘和几何结构，使用CLAHE可以显著提升局部纹理清晰度
            image = cv2.imread(image_path, 3)
            enhanced_image = enhance_image_2(image)
            # 读取近红外图像并做相同增强
            image2 = cv2.imread(image_path_NIRS, 3)
            enhanced_image_NIRS = enhance_image_2(image2)

            # 提取可见光图像上的 SIFT 关键点（最多 400 个）
            keypoints, _ = extract_image_keypoints(enhanced_image, cfg.params.keypoint_extractor) # 这里丢弃了SIFT提供的手工描述子
            nb_keypoints = len(keypoints)
            if nb_keypoints == 0:
                print("nb_keypoints==0")
                continue
            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)
            # [el.pt for el in keypoints]（提取坐标）
            # np.float32(...)（转成 32 位浮点矩阵）
            # .reshape(-1, 1, 2)（调整成 OpenCV 点集标准形状）：其中 -1 表示让 NumPy 自动计算关键点总数 N，多夹一个 1：这是 OpenCV 处理二维点集（比如 perspectiveTransform 等函数）的习惯格式。所以你在后面代码里会看到它用 source_keypoints_coords[b][0][0] 取第 b 个点的 x、用 source_keypoints_coords[b][0][1] 取 y。

            # 贪婪过滤：删除像素距离小于阈值的过密关键点，保证点分布均匀
            # SIFT 经常在纹理特别丰富的地方 1-2个像素内同时检出好几个点，如果不清理的话，会产生大量高度重复的局部快，浪费算力；更关键的是会污染负样本：后面代码是直接拿“相邻下一个关键点 b + 1”当负样本（n.png）
            to_be_removed_ids = []
            for i in range(nb_keypoints):
                for j in range(nb_keypoints):
                    if i != j and j not in to_be_removed_ids:
                        dist = distance.euclidean(source_keypoints_coords[i].flatten(), source_keypoints_coords[j].flatten())
                        if dist < cfg.params.keypoint_dist_threshold:
                            to_be_removed_ids.append(j)

            keypoints = list(keypoints) # OpenCV 返回的 keypoints 是不可修改的元组（tuple），必须先转成列表（list）才能删元素。
            # 倒序删除：从最大下标往最小下标倒着删，前面元素的位置完全不会变动
            for el_idx in sorted(to_be_removed_ids, reverse=True):
                del keypoints[el_idx]
            # 防止极端情况下删完一个点都不剩，直接跳过当前图像。
            if len(keypoints) == 0:
                print("keypoints_len==0")
                continue

            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)
            nb_keypoints = len(source_keypoints_coords) # 记录过滤之后还剩下多少个关键点
            image_height = enhanced_image.shape[0] # OpenCV 读进来的数组形状永远是 (高, 宽, 通道)，所以 shape[0] 是高 height，shape[1] 是宽 width，和坐标 (x, y) 正好是反过来的
            image_width = enhanced_image.shape[1]
            (center_x, center_y) = (image_width // 2, image_height // 2)  # 算出图像正中心点的坐标

            # 随机从候选池中挑选一种自监督几何变换，并统一构造 2x3 仿射变换矩阵 M
            transformation = random.choice(cfg.params.transformation_list)

            if transformation == "rotation":
                # 绕图像中心旋转预设角度
                rotation_angle = random.choice(cfg.params.predefined_angle_degrees)
                M = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
            elif transformation == "zoom":
                # 绕图像中心缩放预设倍率（用 OpenCV 仿射矩阵统一实现，比 scipy.ndimage.zoom 快数十倍）
                zf = random.choice(cfg.params.predefined_zoom_factors)
                M = cv2.getRotationMatrix2D((image_width / 2.0, image_height / 2.0), 0.0, zf)
            elif transformation == "translation":
                # 向右下方平移预设像素偏移量
                shift = random.choice(cfg.params.predefined_translation_pixels)
                M = np.float32([[1, 0, shift], [0, 1, shift]])
            else:
                raise ValueError(f"Unsupported transformation: {transformation}")

            # 对近红外图像统一执行仿射变换
            warped_image = cv2.warpAffine(enhanced_image_NIRS, M, (image_width, image_height))

            # 统一遍历关键点：计算变换后正样本坐标并裁剪保存三元组 (a, p, n)
            triplet_counter = 0
            z = cfg.params.patch_size
            for b in range(0, len(keypoints) - 1):
                xa = int(source_keypoints_coords[b][0][0])              # 锚点坐标 x
                ya = int(source_keypoints_coords[b][0][1])              # 锚点坐标 y
                xn = int(source_keypoints_coords[b + 1][0][0])          # 负样本坐标 x（取下一个不同点）
                yn = int(source_keypoints_coords[b + 1][0][1])          # 负样本坐标 y

                # 使用齐次坐标 [xa, ya, 1] 乘以仿射矩阵 M，得到正样本在变换图中的新坐标
                warped_pt = M.dot(np.array([xa, ya, 1], dtype=np.float32))
                xp, yp = int(warped_pt[0]), int(warped_pt[1])

                # 统一边界检查（修复原版平移分支宽高写反的越界 Bug）
                if (
                    (yp - z) > 0 and (yp + z) < image_height and (xp - z) > 0 and (xp + z) < image_width
                    and (ya - z) > 0 and (ya + z) < image_height and (xa - z) > 0 and (xa + z) < image_width
                    and (yn - z) > 0 and (yn + z) < image_height and (xn - z) > 0 and (xn + z) < image_width
                ):
                    crop_img_p = warped_image[yp - z : yp + z, xp - z : xp + z]
                    crop_img_a = enhanced_image[ya - z : ya + z, xa - z : xa + z]
                    crop_img_n = enhanced_image_NIRS[yn - z : yn + z, xn - z : xn + z]

                    curr_output_folder = osp.join(cfg.paths.export_dir, f"{idx}_{triplet_counter}")
                    os.makedirs(curr_output_folder, exist_ok=True)
                    cv2.imwrite(osp.join(curr_output_folder, "p.png"), crop_img_p)
                    cv2.imwrite(osp.join(curr_output_folder, "a.png"), crop_img_a)
                    cv2.imwrite(osp.join(curr_output_folder, "n.png"), crop_img_n)
                    triplet_counter += 1
            idx += 1

if __name__ == "__main__":
    main()
