#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# ====== 配置 ======
dataset_dir = Path("/mnt/e/datasets/PSCC/Training Dataset/CASIA2.0/Au")
output_txt = Path("casia1_Au_images.txt")
img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")  # 支持格式

# ====== 遍历并收集 ======
all_images = []

for root, dirs, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith(img_exts):
            all_images.append(str(Path(root) / f))

# ====== 写入 txt ======
all_images.sort()  # 可选：排序
with open(output_txt, 'w', encoding='utf-8') as f:
    for img_path in all_images:
        f.write(img_path + "\n")

print(f"完成，收集 {len(all_images)} 张图像，保存到 {output_txt}")
