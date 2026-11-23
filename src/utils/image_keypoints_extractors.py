"""
================================================================================
图像预处理、几何变换与 SIFT 关键点提取工具模块（供数据生成脚本使用）
================================================================================
主要功能：
  - enhance_image / enhance_image_2：使用高斯平滑 + CLAHE（自适应直方图均衡化）增强局部对比度；
  - zoom_coordinates / clipped_zoom：图像缩放及其对应的关键点坐标映射（原版使用 scipy.ndimage.zoom）；
  - extract_image_keypoints：调用 OpenCV SIFT 检测最多 400 个稀疏关键点。
================================================================================
"""
import cv2
from scipy.spatial import distance
from scipy.ndimage import zoom
import random
import numpy as np


def enhance_image(img):
    """灰度图模糊 + CLAHE 对比度增强。"""
    # 1. 将 BGR 彩色图转为单通道灰度图
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR->GRAY
    # 2. 使用 5x5 均值滤波去噪，减少高频椒盐/斑点噪声对 SIFT 检点的干扰
    gray2 = cv2.blur(gray2, (5, 5))
    # 3. 应用限制对比度自适应直方图均衡化（CLAHE），增强局部纹理细节
    clahe = cv2.createCLAHE(clipLimit=5)
    image_enhanced = clahe.apply(gray2)
    return image_enhanced


def enhance_image_2(img):
    """彩色图转 LAB 空间，仅对亮度通道 L 做 CLAHE 增强再转回 BGR（论文数据生成默认增强方式）。"""
    # 对彩色图像进行平滑处理
    blurred = cv2.blur(img, (5, 5))
    # 将平滑后的图像转换为 LAB 色彩空间
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    # 分割 LAB 色彩空间的通道
    l, a, b = cv2.split(lab)
    # 应用自适应直方图均衡化到 L 通道
    clahe = cv2.createCLAHE(clipLimit=5)
    enhanced_l = clahe.apply(l)
    # 合并增强后的 L 通道和原始 a、b 通道
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    # 转换回 BGR 色彩空间
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img




def zoom_coordinates(img, x, y, zoom_factor, **kwargs):
    """
    计算原图关键点 (x, y) 在经历以图像中心为原点的缩放变换 (zoom_factor) 后的新坐标 (x1, y1)。
    注：原代码内部也执行了 scipy.ndimage.zoom(img)（但未返回图像），因此性能开销较大。
    """
    h, w = img.shape[:2]

    # 多通道图像只缩放前两维 (H, W)，通道维缩放因子设为 1
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # 情况 1：缩小（zoom_factor < 1），四周补黑边（Zero-padding）保持输出尺寸仍为 (h, w)
    if zoom_factor < 1:

        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        # 按缩放后在画布上的居中偏移计算对应的新坐标
        y1 = (y + ((h - zh) // 2)) * zoom_factor
        x1 = (x + ((w - zw) // 2)) * zoom_factor
        out = np.zeros_like(img)
        out[top : top + zh, left : left + zw] = zoom(img, zoom_tuple, **kwargs)
        y1 = np.int32(y1)
        x1 = np.int32(x1)

    # 情况 2：放大（zoom_factor >= 1），先截取中心 (h/zf, w/zf) 区域，再放大回 (h, w)
    elif zoom_factor >= 1:

        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        # 减去裁剪起点的左上角偏移后再乘以放大倍率
        y1 = (y - ((h - zh) // 2)) * zoom_factor
        x1 = (x - ((w - zw) // 2)) * zoom_factor
        out = zoom(img[top : top + zh, left : left + zw], zoom_tuple, **kwargs)
        y1 = np.int32(y1)
        x1 = np.int32(x1)

    # 情况 3：倍率为 1 时坐标不变
    else:
        out = img
    return x1, y1


def clipped_zoom(img, zoom_factor, **kwargs):
    """
    保持输出图像分辨率仍为 (h, w) 的中心裁剪/填充缩放函数（原版调用 scipy.ndimage.zoom）。
    """
    h, w = img.shape[:2]

    # 仅对空间高宽维度缩放，通道维保持不变
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # 缩小分支：整图缩小后居中填入全零黑底画布
    if zoom_factor < 1:

        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(img)
        out[top : top + zh, left : left + zw] = zoom(img, zoom_tuple, **kwargs)

    # 放大分支：截取中心区域放大，若因取整导致略大于 (h, w) 再做中心裁剪对齐
    elif zoom_factor >= 1:

        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top : top + zh, left : left + zw], zoom_tuple, **kwargs)

        trim_top = (out.shape[0] - h) // 2
        trim_left = (out.shape[1] - w) // 2
        out = out[trim_top : trim_top + h, trim_left : trim_left + w]

    else:
        out = img
    return out


def extract_image_keypoints(image, extractor_id):
    """使用 SIFT 算法提取最多 400 个关键点（只用其坐标位置，不用于描述子匹配）。"""
    assert extractor_id == "SIFT"
    sift = cv2.SIFT_create(400)
    keypoints, features = sift.detectAndCompute(image, None) # 产生关键点位置跟特征向量
    return keypoints, features
