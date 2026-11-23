"""
================================================================================
自监督三元组数据集极速生成脚本（多进程并行 + 分片打包断点续传版）
================================================================================
相比原版 run_generate_dataset.py 的主要工程优化（值得在写数据预处理脚本时借鉴）：
  1. 兼容 WHU-OPT-SAR 4波段大图：自动剥离前3通道(RGB)与第4通道(NIR)并按 926x926 切块；
  2. 多进程绕过 GIL：使用 ProcessPoolExecutor 跑满 CPU 所有核心，并设 cv2.setNumThreads(1) 避免抢核；
  3. 计算提速：用 OpenCV 中心仿射变换 (warpAffine) 取代单线程超慢的 scipy.ndimage.zoom；
     用 NumPy 广播距离矩阵取代 O(N^2) Python 双重循环；
  4. I/O 提速与防断连：先并发拷贝源图到本地 SSD，再按 chunk 打包成 .tar 落盘到云盘，
     程序中断重跑时自动跳过已生成的 chunk_*.tar，实现断点续传。
================================================================================
"""
import argparse
import math
import os
import random
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    tifffile = None

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TRANS_LIST = ["zoom", "rotation", "translation"]
PREDEFINED_ANGLES = [5, 10, 15]
PREDEFINED_ZOOMS = [0.9, 0.95, 1.05, 1.1, 1.15]
PREDEFINED_TRANSLATION = 8


def collect_images(root_dir):
    """递归收集 root_dir 下所有受支持格式的图像路径并按字典序排序。"""
    files = []
    for root, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(root, fname))
    return sorted(files)


def read_image_any(path):
    """优先使用 OpenCV 原深度读取；若为特殊压缩的多波段 GeoTIFF 则回退到 tifffile。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img
    if tifffile is not None:
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[2] > 16:
            # 将通道在前的 (C, H, W) 格式转为 OpenCV 使用的 (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
        return arr
    raise RuntimeError(f"Failed to read image: {path}")


def to_uint8(arr):
    """将 16 位整型或浮点遥感影像安全线性拉伸为 uint8（SIFT 与 CLAHE 均要求 uint8）。"""
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    max_val = float(arr_f.max())
    min_val = float(arr_f.min())
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)
    if max_val <= 255.0 and min_val >= 0.0:
        return arr_f.astype(np.uint8)
    scaled = (arr_f - min_val) / (max_val - min_val) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def to_bgr_uint8(arr):
    """统一转换为 3 通道 uint8 BGR 图像，供后续 LAB 空间 CLAHE 增强使用。"""
    arr = to_uint8(arr)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if arr.shape[2] >= 3:
        return arr[:, :, :3]
    raise ValueError(f"Unsupported channel count: {arr.shape}")


def enhance_image_fast(bgr_img):
    """与原版 enhance_image_2 等价的快速增强函数（5x5 滤波 + LAB 空间 L 通道 CLAHE）。"""
    blurred = cv2.blur(bgr_img, (5, 5))
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5)
    enhanced_l = clahe.apply(l_ch)
    enhanced_lab = cv2.merge((enhanced_l, a_ch, b_ch))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def filter_sparse_keypoints_fast(coords, dist_threshold):
    """向量化贪婪去重：利用广播距离平方矩阵快速剔除间距小于 dist_threshold 的过密关键点。"""
    n = len(coords)
    if n <= 1:
        return coords
    # 利用 NumPy 广播一次算出 N x N 成对距离平方矩阵，替代双重 Python 循环
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    th_sq = dist_threshold * dist_threshold
    removed = np.zeros(n, dtype=bool)
    for i in range(n):
        if removed[i]:
            continue
        # 保留当前点 i，标记排在其后且距离过近的所有点
        removed[i + 1 :] |= dist_sq[i, i + 1 :] < th_sq
    return coords[~removed]


def iter_tiles(img_a, img_b, tile_size):
    """按固定网格 (tile_size x tile_size) 将原始遥感大图切分为互不重叠的小块。"""
    h, w = img_a.shape[:2]
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        yield 0, img_a, img_b
        return

    tile_id = 0
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            yield tile_id, img_a[y : y + tile_size, x : x + tile_size], img_b[y : y + tile_size, x : x + tile_size]
            tile_id += 1


def process_source_pair(task):
    """单个工作进程任务：读取一对源图 -> 切分网格 -> SIFT 检点 -> 随机仿射变换 -> 写出三元组 PNG。"""
    src_idx, rgb_path, pair_path, chunk_dir, cfg = task
    # 子进程内强制 OpenCV 单线程，避免多进程+多线程争抢 CPU 导致调度开销暴增
    cv2.setNumThreads(1)

    raw_rgb = read_image_any(rgb_path)
    mode = cfg["mode"]
    if mode == "rgb_nir":
        if raw_rgb.ndim < 3 or raw_rgb.shape[2] < 4:
            raise ValueError(
                f"rgb_nir mode requires >=4 channels (RGB+NIR), got shape {raw_rgb.shape} at {rgb_path}"
            )
        # 剥离前 3 通道作为可见光图，第 4 通道作为近红外图
        vis_full = to_bgr_uint8(raw_rgb[:, :, :3])
        nir_full = to_bgr_uint8(raw_rgb[:, :, 3])
    else:
        raw_pair = read_image_any(pair_path)
        vis_full = to_bgr_uint8(raw_rgb)
        nir_full = to_bgr_uint8(raw_pair)

    z = cfg["patch_half"]
    dist_th = cfg["dist_thresh"]
    tile_size = cfg["tile_size"]
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, cfg["png_compression"]]

    sift = cv2.SIFT_create(400)
    saved_count = 0

    # 1. 根据图像尺寸或参数切分网格（5556x3704 -> 24 块 926x926）并逐块生成三元组
    for tile_id, vis_tile, nir_tile in iter_tiles(vis_full, nir_full, tile_size):
        enh_vis = enhance_image_fast(vis_tile)
        enh_nir = enhance_image_fast(nir_tile)
        h, w = enh_vis.shape[:2]

        # 使用 SIFT 在增强后的可见光图像块中提取稀疏关键点
        kps, _ = sift.detectAndCompute(enh_vis, None)
        if not kps:
            continue

        coords = np.float32([kp.pt for kp in kps])
        coords = filter_sparse_keypoints_fast(coords, dist_th)  # 向量化剔除过密点
        if len(coords) < 2:
            continue

        # 随机采样几何变换，构建变换矩阵 M（旋转 / 中心缩放 / 平移）
        trans = random.choice(TRANS_LIST)
        if trans == "rotation":
            angle = random.choice(PREDEFINED_ANGLES)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        elif trans == "zoom":
            zf = random.choice(PREDEFINED_ZOOMS)
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, zf)
        else:
            M = np.float32([[1, 0, PREDEFINED_TRANSLATION], [0, 1, PREDEFINED_TRANSLATION]])

        warped_nir = cv2.warpAffine(enh_nir, M, (w, h), flags=cv2.INTER_LINEAR)

        # 利用齐次坐标矩阵乘法批量计算所有关键点变换后的精确坐标 [N, 2]
        ones = np.ones((len(coords), 1), dtype=np.float32)
        warped_coords = (M @ np.hstack([coords, ones]).T).T

        triplet_counter = 0
        for b in range(len(coords) - 1):
            xa, ya = int(coords[b, 0]), int(coords[b, 1])           # Anchor（可见光原图坐标）
            xp, yp = int(warped_coords[b, 0]), int(warped_coords[b, 1])  # Positive（变换后近红外坐标）
            xn, yn = int(coords[b + 1, 0]), int(coords[b + 1, 1])   # Negative（近红外不同位置坐标）

            # 确保三个局部窗口都在图像边界内，避免切出黑边或越界
            if (
                xp > z and xp + z < w and yp > z and yp + z < h
                and xa > z and xa + z < w and ya > z and ya + z < h
                and xn > z and xn + z < w and yn > z and yn + z < h
            ):
                crop_p = warped_nir[yp - z : yp + z, xp - z : xp + z]
                crop_a = enh_vis[ya - z : ya + z, xa - z : xa + z]
                crop_n = enh_nir[yn - z : yn + z, xn - z : xn + z]

                out_folder = os.path.join(chunk_dir, f"s{src_idx:05d}_t{tile_id:02d}_{triplet_counter}")
                os.makedirs(out_folder, exist_ok=True)
                cv2.imwrite(os.path.join(out_folder, "p.png"), crop_p, png_params)
                cv2.imwrite(os.path.join(out_folder, "a.png"), crop_a, png_params)
                cv2.imwrite(os.path.join(out_folder, "n.png"), crop_n, png_params)
                triplet_counter += 1
                saved_count += 1

    return saved_count


def stage_to_local_ssd(src_files, dst_dir, desc="Caching source files to local SSD"):
    """多线程并发将云盘源文件缓存到本地 SSD，减少多进程重复读网盘的延迟。"""
    os.makedirs(dst_dir, exist_ok=True)
    local_paths = [os.path.join(dst_dir, f"{i:06d}{os.path.splitext(p)[1]}") for i, p in enumerate(src_files)]
    pending = [(s, d) for s, d in zip(src_files, local_paths) if not os.path.exists(d)]
    if not pending:
        print(f"Found all {len(local_paths)} cached files in {dst_dir}, skipping copy.")
        return local_paths

    def _copy(item):
        shutil.copy2(item[0], item[1])

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(tqdm(ex.map(_copy, pending), total=len(pending), desc=desc))
    return local_paths


def detect_mode_and_shape(sample_rgb_path, pair_dir):
    """自动检测样本图像通道数与尺寸：>=4 通道走 rgb_nir 剥离模式，否则走 opt_sar 配对模式。"""
    sample = read_image_any(sample_rgb_path)
    h, w = sample.shape[:2]
    ch = 1 if sample.ndim == 2 else sample.shape[2]
    has_pair = bool(pair_dir and os.path.exists(pair_dir) and len(collect_images(pair_dir)) > 0)
    if ch >= 4:
        mode = "rgb_nir"
    elif has_pair:
        mode = "opt_sar"
    else:
        raise RuntimeError(
            f"Sample image {sample_rgb_path} has shape {sample.shape} (<4 channels) and no valid --pair-dir provided."
        )
    return mode, (h, w, ch), sample.dtype


def main():
    """命令行主入口：解析参数 -> 缓存源文件 -> 分片多进程生成并打包断点保存。"""
    parser = argparse.ArgumentParser(description="Fast multi-process & resumable LTFormer triplet generator")
    parser.add_argument("--rgb-dir", required=True, help="Directory containing RGB (or 4-band optical) images")
    parser.add_argument("--pair-dir", default="", help="Directory containing paired NIR/SAR images (for opt_sar mode)")
    parser.add_argument("--out-dir", required=True, help="Output directory (saved as chunk_*.tar when chunk-size > 0)")
    parser.add_argument("--cache-dir", default="/tmp/ltformer_cache", help="Local fast SSD cache directory")
    parser.add_argument("--mode", choices=["auto", "rgb_nir", "opt_sar"], default="auto")
    parser.add_argument("--tile-size", type=int, default=926, help="Crop tile size for large raw images (0 to disable)")
    parser.add_argument("--patch-half", type=int, default=128, help="Half patch size (crop is 2*patch_half)")
    parser.add_argument("--dist-thresh", type=float, default=3.0, help="Min distance between SIFT keypoints")
    parser.add_argument("--chunk-size", type=int, default=0, help="Source images per tar chunk (0=auto, -1=no tar)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes")
    parser.add_argument("--png-compression", type=int, default=1, help="PNG compression 0-9 (1 is fast and compact)")
    parser.add_argument("--no-local-cache", action="store_true", help="Read directly from --rgb-dir without copying to SSD")
    args = parser.parse_args()

    # 1. 扫描输入目录并自动判断数据模式
    rgb_files = collect_images(args.rgb_dir)
    if not rgb_files:
        raise FileNotFoundError(f"No images found under --rgb-dir: {args.rgb_dir}")

    detected_mode, shape_info, dtype_info = detect_mode_and_shape(rgb_files[0], args.pair_dir)
    mode = detected_mode if args.mode == "auto" else args.mode
    print(f"Detected sample shape={shape_info}, dtype={dtype_info} -> running in '{mode}' mode")

    pair_files = []
    if mode == "opt_sar":
        pair_files = collect_images(args.pair_dir)
        assert len(rgb_files) == len(pair_files), (
            f"Mismatched image counts: rgb={len(rgb_files)} vs pair={len(pair_files)}"
        )

    # 2. 将源数据预热到本地 SSD
    if not args.no_local_cache:
        rgb_files = stage_to_local_ssd(rgb_files, os.path.join(args.cache_dir, "raw_rgb"), "Caching RGB/Optical to SSD")
        if mode == "opt_sar":
            pair_files = stage_to_local_ssd(pair_files, os.path.join(args.cache_dir, "raw_pair"), "Caching NIR/SAR to SSD")
    if mode != "opt_sar":
        pair_files = [""] * len(rgb_files)

    os.makedirs(args.out_dir, exist_ok=True)

    # 3. 自动决定每个分片处理几张源图：大图每片 5 张（约 120 块），小图每片 100 张
    chunk_size = args.chunk_size
    if chunk_size == 0:
        is_large = args.tile_size > 0 and (shape_info[0] > args.tile_size or shape_info[1] > args.tile_size)
        chunk_size = 5 if is_large else 100

    cfg = {
        "mode": mode,
        "tile_size": args.tile_size,
        "patch_half": args.patch_half,
        "dist_thresh": args.dist_thresh,
        "png_compression": args.png_compression,
    }

    total_imgs = len(rgb_files)
    t_start = time.time()

    if chunk_size < 0:
        tasks = [(i, rgb_files[i], pair_files[i], args.out_dir, cfg) for i in range(total_imgs)]
        total_saved = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_source_pair, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Generating triplets"):
                total_saved += fut.result()
        print(f"Done! Generated {total_saved} triplets in {time.time() - t_start:.1f}s -> {args.out_dir}")
        return

    num_chunks = math.ceil(total_imgs / chunk_size)
    print(f"Workers={args.workers}, total_source_images={total_imgs}, chunk_size={chunk_size}, num_chunks={num_chunks}")

    work_root = os.path.join(args.cache_dir, "work_chunks")
    os.makedirs(work_root, exist_ok=True)

    grand_total = 0
    for c_idx in range(num_chunks):
        tar_name = f"chunk_{c_idx:04d}.tar"
        out_tar = os.path.join(args.out_dir, tar_name)
        if os.path.exists(out_tar) and os.path.getsize(out_tar) > 0:
            print(f"[Skip {c_idx + 1}/{num_chunks}] {tar_name} already exists.")
            continue

        start = c_idx * chunk_size
        end = min(start + chunk_size, total_imgs)
        chunk_dir = os.path.join(work_root, f"chunk_{c_idx:04d}")
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)

        tasks = [(i, rgb_files[i], pair_files[i], chunk_dir, cfg) for i in range(start, end)]
        chunk_saved = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_source_pair, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"Chunk {c_idx + 1}/{num_chunks}"):
                chunk_saved += fut.result()

        grand_total += chunk_saved
        tmp_tar = os.path.join(work_root, f"{tar_name}.tmp")
        with tarfile.open(tmp_tar, "w") as tar:
            for folder in os.listdir(chunk_dir):
                tar.add(os.path.join(chunk_dir, folder), arcname=folder)

        shutil.move(tmp_tar, out_tar)
        shutil.rmtree(chunk_dir)
        print(f"[Saved {c_idx + 1}/{num_chunks}] {chunk_saved} triplets -> {out_tar}")

    print(f"All chunks complete! New triplets={grand_total}, elapsed={time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
