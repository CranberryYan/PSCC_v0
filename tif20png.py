#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把目录里的 .tif/.tiff 批量转换为 .png

特性：
- 支持递归子目录
- 默认保持相对目录结构
- 默认输出到单独 out_dir（更安全），也支持“原地输出 png”（out_dir==in_dir）
- 默认只取多页 TIFF 的第 1 页（常见情况）；需要全页可改参数

依赖：pip install pillow
"""

import os
import argparse
from pathlib import Path
from PIL import Image

TIF_EXTS = {".tif", ".tiff"}


def iter_tifs(root: Path, recursive: bool):
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in TIF_EXTS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in TIF_EXTS:
                yield p


def convert_one(src: Path, dst: Path, overwrite: bool, first_page_only: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (not overwrite):
        return "skip"

    with Image.open(src) as im:
        # 多页 TIFF 默认只取第 1 页
        if first_page_only:
            try:
                im.seek(0)
            except Exception:
                pass

        # 有些 tiff 是带 alpha 的/调色板的，转成更通用模式
        # 注意：若是 16-bit 灰度(I;16)，PIL 会保持为 I;16 存成 16-bit PNG
        if im.mode in ("P",):
            im = im.convert("RGBA")
        elif im.mode == "CMYK":
            im = im.convert("RGB")

        im.save(dst, format="PNG", optimize=True)

    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="/mnt/e/datasets/PSCC/Training Dataset/splice_columbia/fake", help="包含 tif/tiff 的输入目录")
    ap.add_argument("--out_dir", type=str, default="/mnt/e/datasets/PSCC/Training Dataset/splice_columbia/fake", help="输出目录（会生成 png）")
    ap.add_argument("--recursive", action="store_true", help="递归子目录")
    ap.add_argument("--keep_rel", action="store_true", default=True,
                    help="保持相对目录结构（默认 True）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的 png")
    ap.add_argument("--first_page_only", action="store_true", default=True,
                    help="多页 TIFF 只转第 1 页（默认 True）")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    skip = 0
    bad = 0
    bad_samples = []

    for src in iter_tifs(in_dir, args.recursive):
        total += 1
        if args.keep_rel:
            rel = src.relative_to(in_dir)
            dst = (out_dir / rel).with_suffix(".png")
        else:
            dst = out_dir / (src.stem + ".png")

        try:
            r = convert_one(src, dst, overwrite=args.overwrite, first_page_only=args.first_page_only)
            if r == "ok":
                ok += 1
            else:
                skip += 1
        except Exception as e:
            bad += 1
            if len(bad_samples) < 20:
                bad_samples.append((str(src), repr(e)))

    print("======== DONE ========")
    print(f"in_dir : {in_dir}")
    print(f"out_dir: {out_dir}")
    print(f"total  : {total}")
    print(f"ok     : {ok}")
    print(f"skip   : {skip}")
    print(f"bad    : {bad}")
    if bad_samples:
        print("\n---- bad samples (up to 20) ----")
        for p, err in bad_samples:
            print(" ", p)
            print("   ", err)


if __name__ == "__main__":
    main()
