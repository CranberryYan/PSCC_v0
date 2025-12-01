import os
import random
from pathlib import Path

# ================== 路径配置 ==================
auth_dir = Path("/mnt/e/datasets/PSCC/Training Dataset/authentic/authentic.txt")
out_list = Path("/mnt/e/datasets/PSCC/authentic_test_500.txt")

# ================== 参数 ==================
NUM_SAMPLES = 500
RANDOM_SEED = 2026
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ================== 扫描图像 ==================
all_imgs = [
    p for p in auth_dir.rglob("*")
    if p.suffix.lower() in IMG_EXTS and p.is_file()
]

print(f"Found {len(all_imgs)} authentic images.")

assert len(all_imgs) >= NUM_SAMPLES, "原图数量不足 500 张！"

# ================== 随机采样 ==================
random.seed(RANDOM_SEED)
selected_imgs = random.sample(all_imgs, NUM_SAMPLES)

# ================== 写入 test list ==================
with open(out_list, "w") as f:
    for p in selected_imgs:
        f.write(str(p) + "\n")

print(f"Saved {NUM_SAMPLES} authentic images to:")
print(out_list)
