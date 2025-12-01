#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把一个“全是原图(authentic)”的文件夹，转换成你的数据集结构：
  <OUT_ROOT>/fake/  保存原图（直接拷贝，不改内容）
  <OUT_ROOT>/mask/  生成同名 .png 全黑 mask（与原图尺寸一致）

✅ 不需要你提前有 mask
✅ 支持 Windows 路径（含空格）/ WSL 路径
✅ 默认递归扫描子目录
✅ 默认保持“真实文件名”（不加编号）
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Iterable, Tuple, Optional

from PIL import Image  # pip install pillow

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def iter_images(src_dir: Path, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        for p in src_dir.rglob("*"):
            if is_image(p):
                yield p
    else:
        for p in src_dir.iterdir():
            if is_image(p):
                yield p


def safe_relpath(p: Path, root: Path) -> Path:
    """获取相对路径（用于 keep_rel_dir）。失败则只返回文件名。"""
    try:
        return p.relative_to(root)
    except Exception:
        return Path(p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=str,
        default=r"/mnt/c/Users/Administrator/Downloads/Columbia Uncompressed Image Splicing Detection/4cam_auth",
        help="源目录（全是原图）",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=r"/mnt/e/datasets/PSCC/Training Dataset/splice_columbia",
        help="输出数据集根目录（会在里面创建 fake/ 和 mask/）",
    )
    ap.add_argument("--recursive", action="store_true", default=True, help="递归扫描子目录（默认 True）")
    ap.add_argument(
        "--keep_rel_dir",
        action="store_true",
        help="保持源目录相对结构，避免不同子目录同名冲突（推荐）",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出已存在同名文件，是否覆盖（默认不覆盖，跳过）",
    )
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_root = Path(args.out)

    if not src_dir.exists():
        raise FileNotFoundError(f"src dir not found: {src_dir}")

    fake_dir = out_root / "fake"
    mask_dir = out_root / "mask"
    fake_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    made_mask = 0
    skipped = 0
    bad = 0
    bad_samples = []

    for img_path in iter_images(src_dir, recursive=args.recursive):
        total += 1

        rel = safe_relpath(img_path, src_dir)
        # 输出图片：保持原后缀、真实文件名（不加编号）
        if args.keep_rel_dir:
            out_img_path = fake_dir / rel
            out_mask_path = (mask_dir / rel).with_suffix(".png")
        else:
            out_img_path = fake_dir / img_path.name
            out_mask_path = mask_dir / (img_path.stem + ".png")

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)

        # 若不覆盖且已存在，跳过
        if (not args.overwrite) and (out_img_path.exists() or out_mask_path.exists()):
            skipped += 1
            continue

        try:
            # 1) 拷贝原图到 fake/
            shutil.copy2(str(img_path), str(out_img_path))
            copied += 1

            # 2) 生成全黑 mask（与原图同尺寸）
            with Image.open(img_path) as im:
                w, h = im.size
            mask = Image.new("L", (w, h), 0)  # 全黑
            mask.save(out_mask_path)          # png
            made_mask += 1

        except Exception as e:
            bad += 1
            if len(bad_samples) < 20:
                bad_samples.append((str(img_path), repr(e)))

    print("======== DONE ========")
    print(f"SRC: {src_dir}")
    print(f"OUT: {out_root}")
    print(f"Total images scanned : {total}")
    print(f"Copied to fake/      : {copied}")
    print(f"Masks generated      : {made_mask}")
    print(f"Skipped (exists)     : {skipped}")
    print(f"Failed               : {bad}")
    if bad_samples:
        print("\n---- Failed samples (up to 20) ----")
        for p, err in bad_samples:
            print(" ", p)
            print("   ", err)


if __name__ == "__main__":
    main()
