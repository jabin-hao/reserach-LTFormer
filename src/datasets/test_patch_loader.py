"""
================================================================================
测试/评估阶段图像对与关键点局部 Patch 加载器（PyTorch Dataset）
================================================================================
目录结构约定：
  root_path/
    └── 序列名/
         ├── rgb/<子文件夹>/<图像名>.png     (源图像：可见光)
         ├── NIRS/<子文件夹>/<图像名>.png    (目标图像：近红外)
         └── matches/<子文件夹>/<图像名>.txt (真值匹配坐标：每行 x_src y_src x_dst y_dst)
作用：读取成对图像及真值关键点，按 patch_size 裁剪局部小块供 run_matching_demo.py 评估。
================================================================================
"""
import os
import os.path as osp
import glob
import cv2 as cv
from numpy import loadtxt
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Sparse correspondences dataset. 稀疏对应点评估数据集"""

    def __init__(self, root_path, patch_size, image_size=None, use_blur=False):
        self.image_name = []
        self.keypoints_GT = []
        self.root_path = root_path
        self.patch_size = patch_size # 默认 64（裁剪 128x128）；设为 128 时裁剪 256x256
        self.image_size = image_size # 若指定且与 2*patch_size 不同，则缩放到 image_size（对齐训练尺度）
        self.use_blur = use_blur     # 是否在 CLAHE 前先做 5x5 均值滤波（对齐训练预处理）

        self.all_frames_per_sequence = []
        self.all_keypoints = []
        self.data = []
        self.load_data_path()

    def __len__(self):
        return len(self.data)

    def load_data_path(self):
        """扫描测试目录下所有序列，把匹配点文件 (.txt) 与对应的 rgb / NIRS 图像路径配对存入 self.data。"""
        sequence_list = glob.glob(self.root_path + "/*")

        # 逐个测试序列目录扫描
        for sequence in sequence_list:
            # 读取当前序列 matches/ 下的子目录列表
            curr_matches_list = sorted(glob.glob(sequence+"/matches/*"))
            for match_path in curr_matches_list:
                folder_name = os.path.basename(match_path)
                curr_matches = sorted(glob.glob(match_path +"/*"))
                for match in curr_matches:
                    # 根据匹配文件名反推对应的可见光图像和近红外图像路径
                    file_path = os.path.splitext(match)[0]
                    file_name = os.path.basename(file_path)
                    src_filename = file_name+".png"
                    dst_filename = file_name+".png"
                    src_abs_path = osp.join(sequence,"rgb",folder_name, src_filename)
                    dst_abs_path = osp.join(sequence,"NIRS",folder_name, dst_filename)

                    # 若真值点文件为空，跳过该图像对
                    matchpoint = loadtxt(match, dtype='float')
                    if matchpoint.size == 0:
                        print("******")
                        continue
                    # 只有源图和目标图都存在时才加入评估列表
                    if osp.exists(src_abs_path) and osp.exists(dst_abs_path):
                        self.data.append({"match": match, "src_frame": src_abs_path, "dst_frame": dst_abs_path})

    def enhance(self, img):
        """
        将测试图像转为单通道灰度图并做 CLAHE 增强。
          - use_blur=False：原作者测试口径（灰度 -> CLAHE）；
          - use_blur=True ：严格复刻训练三元组生成口径（彩色 5x5 均值滤波 -> LAB 空间 L 通道 CLAHE -> BGR -> 灰度），
                            与 run_generate_dataset_fast.enhance_image_fast + TripletDataset 的 convert('L') 完全一致。
        """
        if self.use_blur:
            blurred = cv.blur(img, (5, 5))
            lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv.split(lab)
            l_ch = cv.createCLAHE(clipLimit=5).apply(l_ch)
            enhanced_bgr = cv.cvtColor(cv.merge((l_ch, a_ch, b_ch)), cv.COLOR_LAB2BGR)
            return cv.cvtColor(enhanced_bgr, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=5)
        image_enhanced = clahe.apply(gray2)
        return image_enhanced
    
    def __getitem__(self, idx):
        """读取第 idx 对测试图像和真值点对文件，并按 patch_size 裁剪成对的局部 Patch。"""
        frame_src = cv.imread(self.data[idx]["src_frame"], 3) #flag=3 原深度 3通道
        Next_frame = cv.imread(self.data[idx]["dst_frame"], 3)

        # 保存一份未增强的 RGB 原图，用于最后可视化连线展示
        ori_src = cv.cvtColor(frame_src, cv.COLOR_BGR2RGB)
        ori_next = cv.cvtColor(Next_frame, cv.COLOR_BGR2RGB)

        # 转灰度并做 CLAHE 增强，提高异源图像局部纹理对比度
        frame_src = self.enhance(frame_src)
        Next_frame = self.enhance(Next_frame)

        image_path = self.data[idx]["src_frame"]
        file_path = os.path.splitext(image_path)[0]
        folder_name = os.path.basename(os.path.dirname(image_path))  # 跨平台取子文件夹名（如 sample）

        # 读取真值匹配点文件：每行 4 个浮点数 (x_src, y_src, x_dst, y_dst)
        matches = self.data[idx]["match"]

        list_matches = loadtxt(matches, dtype='float') #match为读取地址 有2个点 四个值
        if list_matches.ndim == 1:
            list_matches = list_matches.reshape(1, -1)

        list_keypoints_src = []
        list_keypoints_next_frame = []

        for i in range(0, len(list_matches)):
            list_keypoints_src.append((list_matches[i][0], list_matches[i][1]))
            list_keypoints_next_frame.append((list_matches[i][2], list_matches[i][3]))
        h, w = frame_src.shape

        # 逐个关键点遍历裁剪成对局部 Patch
        gt_key_src = []
        gt_key_next_frame = []
        patches_src = []
        patches_next_frame = []

        for b in range(0, len(list_keypoints_src)):

            xa = int(list_keypoints_src[b][0])
            ya = int(list_keypoints_src[b][1])
            xp = int(list_keypoints_next_frame[b][0])
            yp = int(list_keypoints_next_frame[b][1])

            # 边界检查：确保以关键点为中心裁剪 [2*patch_size, 2*patch_size] 不越界
            if (
                    ((ya - self.patch_size) > 0)
                    & ((xa - self.patch_size) > 0)
                    & ((ya + self.patch_size) < h)
                    & ((xa + self.patch_size) < w)
                    & ((yp - self.patch_size) > 0)
                    & ((xp - self.patch_size) > 0)
                    & ((yp + self.patch_size) < h)
                    & ((xp + self.patch_size) < w)
            ):
                crop_patches_src = frame_src[
                    ya - self.patch_size : ya + self.patch_size, xa - self.patch_size : xa + self.patch_size
                ]
                crop_patches_next_frame = Next_frame[
                    yp - self.patch_size : yp + self.patch_size, xp - self.patch_size : xp + self.patch_size
                ]
                if self.image_size is not None and (
                    crop_patches_src.shape[0] != self.image_size
                    or crop_patches_src.shape[1] != self.image_size
                ):
                    # 256->128 下采样用 INTER_AREA（带抗混叠，更接近训练时 PIL resize 的效果）
                    crop_patches_src = cv.resize(
                        crop_patches_src, (self.image_size, self.image_size), interpolation=cv.INTER_AREA
                    )
                    crop_patches_next_frame = cv.resize(
                        crop_patches_next_frame, (self.image_size, self.image_size), interpolation=cv.INTER_AREA
                    )

                patches_next_frame.append(crop_patches_next_frame)
                patches_src.append(crop_patches_src)
                gt_key_src.append(list_keypoints_src[b])
                gt_key_next_frame.append(list_keypoints_next_frame[b])
        return {
            # 增强后的单通道图像
            "image_src_name": frame_src,
            "image_dst_name": Next_frame,
            # patch 列表
            "patch_src": patches_src,
            "patch_dst": patches_next_frame,
            # 真值关键点坐标列表
            "keypoint_src": gt_key_src,
            "keypoint_dst": gt_key_next_frame,
            # 图像所在子文件夹名与原始 RGB 图像
            "image_name":folder_name,
            "ori_src": ori_src,
            "ori_dst": ori_next
        }