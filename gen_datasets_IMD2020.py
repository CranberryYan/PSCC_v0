#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def windows_to_wsl_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    if p.startswith("/mnt/"):
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_copy_or_link(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return
    if mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return
    raise ValueError(f"Unknown mode: {mode}")

def save_png_via_pillow(src: Path, dst: Path, as_mask: bool = False, binarize_mask: bool = False, thresh: int = 0):
    from PIL import Image
    ensure_dir(dst.parent)
    im = Image.open(src)

    if as_mask:
        im = im.convert("L")
        if binarize_mask:
            import numpy as np
            arr = (np.array(im) > thresh).astype("uint8") * 255
            im = Image.fromarray(arr, mode="L")
        im.save(dst, format="PNG")
    else:
        if im.mode == "RGBA":
            im = im.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        im.save(dst, format="PNG")

def is_orig_stem(stem: str) -> bool:
    return stem.endswith("_orig")

def build_pairs(in_root: Path):
    masks = {}
    images = {}

    for p in in_root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in IMG_EXTS:
            continue

        rel = p.relative_to(in_root)
        stem = p.stem

        # ===== 忽略 _orig（关键）=====
        if stem.endswith("_mask"):
            base_stem = stem[:-5]  # remove "_mask"
            if is_orig_stem(base_stem):
                continue
            key = (rel.parent / base_stem).as_posix()
            masks.setdefault(key, p)
        else:
            if is_orig_stem(stem):
                continue
            key = (rel.parent / stem).as_posix()
            images.setdefault(key, p)

    pairs = []
    missing_mask = []
    for key, img_path in images.items():
        m = masks.get(key, None)
        if m is None:
            missing_mask.append(img_path)
            continue
        pairs.append((key, img_path, m))

    return pairs, missing_mask

def make_unique_stem(stem: str, rel_parent: str, used: set):
    if stem not in used:
        used.add(stem)
        return stem
    parent_slug = rel_parent.replace("/", "__").replace("\\", "__")
    candidate = f"{parent_slug}__{stem}" if parent_slug else f"dup__{stem}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    i = 2
    while True:
        c2 = f"{candidate}__{i}"
        if c2 not in used:
            used.add(c2)
            return c2
        i += 1

def write_list(path: Path, lines):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in lines:
            f.write(s)
            f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="IMD2020 -> PSCC-style dataset maker (fake/mask) + txt lists.")
    ap.add_argument("--in-root", default="/mnt/c/Users/Administrator/Downloads/IMD2020", help=r'输入根目录，例如 "C:\Users\Administrator\Downloads\IMD2020" 或 "/mnt/c/Users/Administrator/Downloads/IMD2020"')
    ap.add_argument("--out-root", default=r"/mnt/e/datasets/PSCC/Training Dataset", help='输出根目录，默认 "/mnt/e/datasets/PSCC/Training Dataset"')
    ap.add_argument("--tamper", default="IMD2020", help='篡改类型子目录名，默认 copymove')
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy", help="写入方式：copy/symlink/hardlink，默认 copy")
    ap.add_argument("--flatten", action="store_true", help="扁平化输出到 fake/ 和 mask/ 下（默认：扁平化）")
    ap.add_argument("--keep-subdirs", action="store_true", help="保留输入子目录结构（开启则不扁平化）")
    ap.add_argument("--to-png", action="store_true", help="把 fake/mask 都转成 PNG")
    ap.add_argument("--binarize-mask", action="store_true", help="把 mask 二值化为 0/255（默认不做）")
    ap.add_argument("--mask-thresh", type=int, default=0, help="mask 二值化阈值，>thresh 视为前景，默认 0")

    # ===== list 输出开关与文件名 =====
    ap.add_argument("--write-lists", action="store_true", help="生成 fake_list.txt / pair_list.txt（默认开启）")
    ap.add_argument("--lists-prefix", default="", help="list 文件名前缀，例如 train_ -> train_fake_list.txt")
    args = ap.parse_args()

    # 默认开启写 list（用户不传 --write-lists 也会写）
    write_lists = True if (not args.write_lists) else True  # 兼容用户习惯：总是写

    flatten = True
    if args.keep_subdirs:
        flatten = False
    if args.flatten:
        flatten = True

    in_root = Path(windows_to_wsl_path(args.in_root)).expanduser().resolve()
    out_root = Path(windows_to_wsl_path(args.out_root)).expanduser().resolve()

    fake_dir = out_root / args.tamper / "fake"
    mask_dir = out_root / args.tamper / "mask"
    ensure_dir(fake_dir)
    ensure_dir(mask_dir)

    pairs, missing = build_pairs(in_root)
    print(f"[IN ] {in_root}")
    print(f"[OUT] {out_root / args.tamper}")
    print(f"[FOUND] pairs={len(pairs)}  missing_mask={len(missing)}")

    used_names = set()

    # ===== 收集 list =====
    fake_list = []
    pair_list = []

    for key, img_path, mask_path in pairs:
        rel_parent = str(Path(key).parent).replace("\\", "/")
        img_stem = img_path.stem

        if flatten:
            out_stem = make_unique_stem(img_stem, rel_parent, used_names)
            out_fake = fake_dir / (out_stem + (".png" if args.to_png else img_path.suffix.lower()))
            out_mask = mask_dir / (out_stem + (".png" if args.to_png else mask_path.suffix.lower()))
        else:
            out_fake = fake_dir / Path(key).parent / (img_stem + (".png" if args.to_png else img_path.suffix.lower()))
            out_mask = mask_dir / Path(key).parent / (img_stem + (".png" if args.to_png else mask_path.suffix.lower()))

        if args.to_png:
            save_png_via_pillow(img_path, out_fake, as_mask=False)
            save_png_via_pillow(mask_path, out_mask, as_mask=True,
                                binarize_mask=args.binarize_mask, thresh=args.mask_thresh)
        else:
            safe_copy_or_link(img_path, out_fake, args.mode)
            safe_copy_or_link(mask_path, out_mask, args.mode)

        # ===== 写入 list：用 WSL 绝对路径 =====
        fake_posix = out_fake.resolve().as_posix()
        mask_posix = out_mask.resolve().as_posix()
        fake_list.append(fake_posix)
        pair_list.append(fake_posix)
        pair_list.append(mask_posix)

    # ===== sort 保持稳定（尤其 flatten 时）=====
    # fake_list 与 pair_list 需要按 fake 的顺序排
    order = sorted(range(len(fake_list)), key=lambda i: fake_list[i])
    fake_list_sorted = [fake_list[i] for i in order]
    pair_list_sorted = []
    for i in order:
        pair_list_sorted.append(fake_list[i])
        pair_list_sorted.append(pair_list[2*i + 1])  # 对应 mask

    if write_lists:
        out_tamper_dir = out_root / args.tamper
        prefix = args.lists_prefix
        fake_list_path = out_tamper_dir / f"{prefix}fake_list.txt"
        pair_list_path = out_tamper_dir / f"{prefix}pair_list.txt"
        write_list(fake_list_path, fake_list_sorted)
        write_list(pair_list_path, pair_list_sorted)
        print(f"[LIST] wrote: {fake_list_path}")
        print(f"[LIST] wrote: {pair_list_path}")

    if missing:
        print("\n[WARN] 以下图片未找到对应 mask（仅展示前 20 个）：")
        for p in missing[:20]:
            print("  ", str(p))

    print("\n[DONE] dataset prepared.")

if __name__ == "__main__":
    main()
