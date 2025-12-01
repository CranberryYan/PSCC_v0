#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取:
  E:\datasets\PSCC\Training Dataset\splice_columbia\fake
生成 txt，每行是:
  /mnt/e/datasets/PSCC/Training Dataset/splice_columbia/fake_256/<filename>.png

说明：
- 只输出“fake_256 下的目标路径”，不输出 mask、不输出 label
- 支持输入为 Windows 路径(E:\...) 或 WSL 路径(/mnt/e/...)
- 默认不递归；需要递归加 --recursive
- 默认输出文件名保持原文件名；若原后缀不是 .png，会强制改为 .png（你要的示例就是 .png）
"""

import os
import re
import argparse
from pathlib import Path
from typing import Iterable

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def windows_to_wsl_path(p: str) -> str:
    """
    Windows 路径 -> WSL 路径:
      E:\A\B -> /mnt/e/A/B
    已是 /mnt/... 则原样返回
    """
    p = p.strip().strip('"').strip("'")
    if p.startswith("/mnt/"):
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="/mnt/e/datasets/PSCC/Training Dataset/splice_columbia/fake/",
                    help=r"输入目录，例如: E:\datasets\PSCC\Training Dataset\splice_columbia\fake")
    ap.add_argument("--dst_root", type=str, default="/mnt/e/datasets/PSCC/Training Dataset/splice_columbia/fake/",
                    help=r"输出路径前缀，例如: /mnt/e/datasets/PSCC/Training Dataset/splice_columbia/fake_256")
    ap.add_argument("--out_txt", type=str, default="./datasets_Columbia.txt",
                    help="输出 txt 路径")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    ap.add_argument("--force_png", action="store_true", default=True,
                    help="强制输出后缀为 .png（默认 True）")
    args = ap.parse_args()

    src_dir = Path(windows_to_wsl_path(args.src))
    dst_root = Path(windows_to_wsl_path(args.dst_root))
    out_txt = Path(windows_to_wsl_path(args.out_txt))

    if not src_dir.exists():
        raise FileNotFoundError(f"src not found: {src_dir}")

    files = sorted(list(iter_images(src_dir, args.recursive)))
    if not files:
        print("未找到任何图片文件。")
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("", encoding="utf-8")
        return

    lines = []
    for p in files:
        name = p.name
        if args.force_png and p.suffix.lower() != ".png":
            name = p.stem + ".png"
        lines.append((dst_root / name).as_posix())

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("======== DONE ========")
    print("SRC:", src_dir)
    print("DST_ROOT:", dst_root)
    print("Count:", len(lines))
    print("OUT_TXT:", out_txt.as_posix())


if __name__ == "__main__":
    main()
