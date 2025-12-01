#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def windows_to_wsl_path(p: str) -> str:
    """C:\\A\\B -> /mnt/c/A/B ; /mnt/... 原样返回"""
    p = p.strip().strip('"').strip("'")
    if p.startswith("/mnt/"):
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")


def to_wsl_like(p: Path) -> str:
    """把 Path 尽量转成 /mnt/x/... 形式（如果是 Windows 盘符路径）"""
    s = str(p.resolve())
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p.resolve().as_posix()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def norm_key_from_stem(stem: str) -> str:
    """
    归一化 key，用于 fake <-> mask 配对
    规则：小写，去掉常见 gt/mask 后缀
    """
    s = stem.strip().lower()

    # 去掉常见后缀（按从长到短）
    suffixes = [
        "_groundtruth", "-groundtruth", "groundtruth",
        "_gt", "-gt", ".gt",
        "_mask", "-mask", ".mask",
    ]
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                s = s.rstrip("_- .")
                changed = True
                break

    return s


def index_images(root: Path) -> Dict[str, List[Path]]:
    """
    遍历 root 递归建立：norm_key -> [paths...]
    """
    d: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        key = norm_key_from_stem(p.stem)
        d.setdefault(key, []).append(p)
    return d


def pick_one(cands: List[Path]) -> Path:
    """
    多个候选时：优先选 png，其次按路径字典序
    """
    if len(cands) == 1:
        return cands[0]
    cands_sorted = sorted(cands, key=lambda x: (0 if x.suffix.lower() == ".png" else 1, str(x)))
    return cands_sorted[0]


def find_mask_for_fake(fake_key: str, masks_index: Dict[str, List[Path]]) -> Optional[Path]:
    """
    先精确 key 匹配；再做包含关系的唯一匹配（保守）
    """
    if fake_key in masks_index:
        return pick_one(masks_index[fake_key])

    # 包含关系的唯一匹配（只有一个候选才用）
    contains = []
    for k, ps in masks_index.items():
        if (fake_key in k) or (k in fake_key):
            contains.append((k, ps))
    if len(contains) == 1:
        return pick_one(contains[0][1])

    return None


def save_png(src: Path, dst: Path, as_mask: bool, binarize: bool = False, thresh: int = 0):
    from PIL import Image
    ensure_dir(dst.parent)

    im = Image.open(src)

    if as_mask:
        # mask：强制单通道，插值问题与 resize 无关（这里只做保存格式统一）
        im = im.convert("L")
        if binarize:
            import numpy as np
            arr = (np.array(im) > thresh).astype("uint8") * 255
            im = Image.fromarray(arr, mode="L")
        im.save(dst, format="PNG")
    else:
        # fake：统一 RGB
        if im.mode == "RGBA":
            im = im.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        im.save(dst, format="PNG")


def write_lines(path: Path, lines: List[str]):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in lines:
            f.write(s + "\n")


def main():
    ap = argparse.ArgumentParser(description="Make CASIA2.0 dataset (fake/mask) + txt lists.")
    ap.add_argument("--fake-root", default=r"/mnt/c/Users/Administrator/Downloads/CASIA2.0/Tp",
                    help="CASIA fake(Tp) 根目录")
    ap.add_argument("--mask-root", default=r"/mnt/c/Users/Administrator/Downloads/CASIA2.0_GT/",
                    help="CASIA GT(mask) 根目录")
    ap.add_argument("--out-root", default=r"/mnt/e/datasets/PSCC/Training Dataset/CASIA2.0",
                    help="输出根目录（会生成 fake/ mask/ 以及 txt）")
    ap.add_argument("--to-png", action="store_true", help="输出统一转成 png（默认开启）")
    ap.add_argument("--no-to-png", action="store_true", help="不转 png，保留原后缀复制")
    ap.add_argument("--mode", choices=["copy"], default="copy", help="写入方式：目前仅 copy")
    ap.add_argument("--binarize-mask", action="store_true", help="把 mask 二值化为 0/255（可选）")
    ap.add_argument("--mask-thresh", type=int, default=0, help="mask 二值化阈值，>thresh 为前景")
    ap.add_argument("--list-format", choices=["wsl", "native"], default="wsl",
                    help="txt 里路径格式：wsl(/mnt/x/...) 或 native(系统原生)")
    args = ap.parse_args()

    to_png = True
    if args.no_to_png:
        to_png = False
    if args.to_png:
        to_png = True

    fake_root = Path(windows_to_wsl_path(args.fake_root) if os.name != "nt" else args.fake_root).expanduser().resolve()
    mask_root = Path(windows_to_wsl_path(args.mask_root) if os.name != "nt" else args.mask_root).expanduser().resolve()
    out_root = Path(windows_to_wsl_path(args.out_root) if os.name != "nt" else args.out_root).expanduser().resolve()

    out_fake = out_root / "fake"
    out_mask = out_root / "mask"
    ensure_dir(out_fake)
    ensure_dir(out_mask)

    # 建索引
    print("[IN ] fake:", fake_root)
    print("[IN ] mask:", mask_root)
    print("[OUT] root:", out_root)

    masks_index = index_images(mask_root)

    # 遍历 fake，配对 mask
    fake_paths = [p for p in fake_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    fake_paths = sorted(fake_paths, key=lambda x: str(x))

    used_names = set()
    missing = []

    fake_list: List[str] = []
    pair_list: List[str] = []

    for fp in fake_paths:
        key = norm_key_from_stem(fp.stem)
        mp = find_mask_for_fake(key, masks_index)
        if mp is None:
            missing.append(fp)
            continue

        # 输出文件名：用 fake 的 stem（若冲突，加序号）
        stem_out = fp.stem
        if stem_out in used_names:
            i = 2
            while f"{stem_out}_{i}" in used_names:
                i += 1
            stem_out = f"{stem_out}_{i}"
        used_names.add(stem_out)

        if to_png:
            out_fp = out_fake / f"{stem_out}.png"
            out_mp = out_mask / f"{stem_out}.png"
            save_png(fp, out_fp, as_mask=False)
            save_png(mp, out_mp, as_mask=True, binarize=args.binarize_mask, thresh=args.mask_thresh)
        else:
            out_fp = out_fake / f"{stem_out}{fp.suffix.lower()}"
            out_mp = out_mask / f"{stem_out}{mp.suffix.lower()}"
            ensure_dir(out_fp.parent)
            ensure_dir(out_mp.parent)
            shutil.copy2(fp, out_fp)
            shutil.copy2(mp, out_mp)

        if args.list_format == "wsl":
            fake_str = to_wsl_like(out_fp)
            mask_str = to_wsl_like(out_mp)
        else:
            fake_str = str(out_fp.resolve())
            mask_str = str(out_mp.resolve())

        fake_list.append(fake_str)
        pair_list.append(fake_str)
        pair_list.append(mask_str)

    # 写 txt
    fake_list_path = out_root / "fake_list.txt"
    pair_list_path = out_root / "pair_list.txt"
    write_lines(fake_list_path, sorted(fake_list))
    # pair_list 要与 fake_list 顺序一致：按 fake_str 排序后再写两行
    order = sorted(range(len(fake_list)), key=lambda i: fake_list[i])
    pair_sorted = []
    for i in order:
        pair_sorted.append(pair_list[2 * i])
        pair_sorted.append(pair_list[2 * i + 1])
    write_lines(pair_list_path, pair_sorted)

    print(f"[DONE] paired={len(fake_list)}  missing={len(missing)}")
    print("[TXT ]", fake_list_path)
    print("[TXT ]", pair_list_path)

    if missing:
        print("\n[WARN] 缺少 mask 的 fake（仅展示前 30 个）：")
        for p in missing[:30]:
            print("  ", str(p))


if __name__ == "__main__":
    main()
