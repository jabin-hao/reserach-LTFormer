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
from scipy.ndimage import zoom
import random
import numpy as np

from src.utils.image_keypoints_extractors import extract_image_keypoints, enhance_image,enhance_image_2,zoom_coordinates,clipped_zoom

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
            image = cv2.imread(image_path, 3)
            enhanced_image = enhance_image_2(image)
            # 读取近红外图像并做相同增强
            image2 = cv2.imread(image_path_NIRS, 3)
            enhanced_image_NIRS = enhance_image_2(image2)

            # 提取可见光图像上的 SIFT 关键点（最多 400 个）
            keypoints, _ = extract_image_keypoints(enhanced_image, cfg.params.keypoint_extractor)
            nb_keypoints = len(keypoints)
            if nb_keypoints == 0:
                print("nb_keypoints==0")
                continue
            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)

            # 贪婪过滤：删除像素距离小于阈值的过密关键点，保证点分布均匀
            to_be_removed_ids = []
            for i in range(nb_keypoints):
                for j in range(nb_keypoints):
                    if i != j and j not in to_be_removed_ids:
                        dist = distance.euclidean(source_keypoints_coords[i].flatten(), source_keypoints_coords[j].flatten())
                        if dist < cfg.params.keypoint_dist_threshold:
                            to_be_removed_ids.append(j)

            keypoints = list(keypoints)

            for el_idx in sorted(to_be_removed_ids, reverse=True):
                del keypoints[el_idx]

            if len(keypoints) == 0:
                print("keypoints_len==0")
                continue

            source_keypoints_coords = np.float32([el.pt for el in keypoints]).reshape(-1, 1, 2)
            nb_keypoints = len(source_keypoints_coords)
            image_height = enhanced_image.shape[0]
            image_width = enhanced_image.shape[1]
            (center_x, center_y) = (image_width // 2, image_height // 2)

            # 随机从候选池中挑选一种自监督几何变换（缩放 / 旋转 / 平移）
            transformation = random.choice(cfg.params.transformation_list)

            if transformation == "rotation":
                # 绕图像中心旋转预设角度，同步计算旋转矩阵映射关键点坐标
                rotation_angle = random.choice(cfg.params.predefined_angle_degrees)
                rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
                warped_image = cv2.warpAffine(enhanced_image_NIRS, rotation_matrix, (image_width, image_height))

                triplet_counter = 0
                for b in range(0, len(keypoints) - 1):
                    rotated_point = rotation_matrix.dot(
                        np.array((int(source_keypoints_coords[b][0, 0]), int(source_keypoints_coords[b][0][1])) + (1,))
                    )
                    xp = int(rotated_point[0])                              # 正样本新坐标 x
                    yp = int(rotated_point[1])                              # 正样本新坐标 y
                    xa = int(source_keypoints_coords[b][0][0])              # 锚点坐标 x
                    ya = int(source_keypoints_coords[b][0][1])              # 锚点坐标 y
                    xn = int(source_keypoints_coords[b + 1][0][0])          # 负样本坐标 x（取下一个不同点）
                    yn = int(source_keypoints_coords[b + 1][0][1])          # 负样本坐标 y
                    z = cfg.params.patch_size                               # 裁剪半边长

                    # 检查三个裁剪窗口是否全部位于图像画布内部
                    if (
                        ((yp - z) > 0)
                        & ((xp - z) > 0)
                        & ((yp + z) < image_height)
                        & ((xp + z) < image_width)
                        & ((ya - z) > 0)
                        & ((xa - z) > 0)
                        & ((ya + z) < image_height)
                        & ((xa + z) < image_width)
                        & ((yn - z) > 0)
                        & ((xn - z) > 0)
                        & ((yn + z) < image_height)
                        & ((xn + z) < image_width)
                    ):
                        # 分别裁出正样本、锚点、负样本图像块并写入磁盘
                        crop_img_p = warped_image[yp - z : yp + z, xp - z : xp + z]
                        crop_img_a = enhanced_image[ya - z : ya + z, xa - z : xa + z]
                        crop_img_n = enhanced_image_NIRS[yn - z : yn + z, xn - z : xn + z]

                        curr_output_folder = osp.join(cfg.paths.export_dir,f"{idx}_{triplet_counter}")
                        filename_p = curr_output_folder + "/p.png"
                        filename_a = curr_output_folder + "/a.png"
                        filename_n = curr_output_folder + "/n.png"

                        if not osp.exists(curr_output_folder):
                            os.makedirs(curr_output_folder)

                        cv2.imwrite(filename_p, crop_img_p)
                        cv2.imwrite(filename_a, crop_img_a)
                        cv2.imwrite(filename_n, crop_img_n)

                        triplet_counter += 1
                idx += 1
            # ------------------------------- 缩放变换 (zoom) -------------------------------
            elif transformation == 'zoom':
                # 随机选择一个缩放倍率，对近红外图及对应关键点同步缩放
                zoom_tab = random.choice(cfg.params.predefined_zoom_factors)
                zf = zoom_tab
                transformed = clipped_zoom(enhanced_image_NIRS, zf)
                triplet_counter = 0
                for b in range(0, len(keypoints) - 1):
                    xa = int(source_keypoints_coords[b][0][0])
                    ya = int(source_keypoints_coords[b][0][1])
                    xp, yp = zoom_coordinates(enhanced_image_NIRS, xa, ya, zf)
                    xn = int(source_keypoints_coords[b + 1][0][0])
                    yn = int(source_keypoints_coords[b + 1][0][1])
                    z = cfg.params.patch_size

                    # 边界检查，防止越界切出残缺块
                    if  (
                        ((yp - z) > 0)
                        & ((xp - z) > 0)
                        & ((yp + z) < image_height)
                        & ((xp + z) < image_width)
                        & ((ya - z) > 0)
                        & ((xa - z) > 0)
                        & ((ya + z) < image_height)
                        & ((xa + z) < image_width)
                        & ((yn - z) > 0)
                        & ((xn - z) > 0)
                        & ((yn + z) < image_height)
                        & ((xn + z) < image_width)
                    ):
                        crop_img_p = transformed[yp - z: yp + z, xp - z: xp + z]
                        crop_img_a = enhanced_image[ya - z: ya + z, xa - z: xa + z]
                        crop_img_n = enhanced_image_NIRS[yn - z: yn + z, xn - z: xn + z]

                        curr_output_folder = osp.join(cfg.paths.export_dir, f"{idx}_{triplet_counter}")
                        filename_p = curr_output_folder + "/p.png"
                        filename_a = curr_output_folder + "/a.png"
                        filename_n = curr_output_folder + "/n.png"
                        if not osp.exists(curr_output_folder):
                            os.makedirs(curr_output_folder)

                        cv2.imwrite(filename_p, crop_img_p)
                        cv2.imwrite(filename_a, crop_img_a)
                        cv2.imwrite(filename_n, crop_img_n)
                        triplet_counter += 1
                idx += 1
            # ------------------------------- 平移变换 (translation) -------------------------------
            elif transformation == 'translation':
                # 构造向右下方平移 (8, 8) 像素的仿射矩阵
                M = np.float32([[1, 0, 8], [0, 1, 8]])
                transformed = cv2.warpAffine(enhanced_image_NIRS, M, (image_width, image_height))
                triplet_counter = 0
                for b in range(0, len(keypoints) - 1):
                    rotated_point = (M.dot(np.array((int(source_keypoints_coords[b][0][0]), int(source_keypoints_coords[b][0][1])) + (1,))))
                    xp = int(rotated_point[0])
                    yp = int(rotated_point[1])
                    xa = int(source_keypoints_coords[b][0][0])
                    ya = int(source_keypoints_coords[b][0][1])
                    xn = int(source_keypoints_coords[b + 1][0][0])
                    yn = int(source_keypoints_coords[b + 1][0][1])
                    z = cfg.params.patch_size

                    # 边界检查，确保裁剪块位于画布内
                    if (
                            ((yp - z) > 0)
                            & ((xp - z) > 0)
                            & ((yp + z) < image_width)
                            & ((xp + z) < image_height)
                            & ((ya - z) > 0)
                            & ((xa - z) > 0)
                            & ((ya + z) < image_width)
                            & ((xa + z) < image_height)
                            & ((yn - z) > 0)
                            & ((xn - z) > 0)
                            & ((yn + z) < image_width)
                            & ((xn + z) < image_height)
                    ):

                        crop_img_p = transformed[yp - z: yp + z, xp - z: xp + z]
                        crop_img_a = enhanced_image[ya - z: ya + z, xa - z: xa + z]
                        crop_img_n = enhanced_image_NIRS[yn - z: yn + z, xn - z: xn + z]

                        curr_output_folder = osp.join(cfg.paths.export_dir, f"{idx}_{triplet_counter}")
                        filename_p = curr_output_folder + "/p.png"
                        filename_a = curr_output_folder + "/a.png"
                        filename_n = curr_output_folder + "/n.png"
                        if not osp.exists(curr_output_folder):
                            os.makedirs(curr_output_folder)

                        cv2.imwrite(filename_p, crop_img_p)
                        cv2.imwrite(filename_a, crop_img_a)
                        cv2.imwrite(filename_n, crop_img_n)
                        triplet_counter += 1
                idx += 1



if __name__ == "__main__":
    main()
