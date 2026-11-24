"""
================================================================================
图像预处理与 SIFT 关键点提取工具模块（供数据生成脚本使用）
================================================================================
主要功能：
  - enhance_image / enhance_image_2：使用均值平滑 + CLAHE（自适应直方图均衡化）增强局部对比度；
  - extract_image_keypoints：调用 OpenCV SIFT 检测最多 400 个稀疏关键点。
================================================================================
"""
import cv2


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




def extract_image_keypoints(image, extractor_id):
    """使用 SIFT 算法提取最多 400 个关键点（只用其坐标位置，不用于描述子匹配）。"""
    assert extractor_id == "SIFT"
    sift = cv2.SIFT_create(400) # OpenCV已经把SIFT复杂的底层C++算法封装成了现成的接口
    keypoints, features = sift.detectAndCompute(image, None) # 产生关键点对象列表（包含坐标(x,y)、尺度、方向等）以及128维的手工描述子向量
    return keypoints, features
