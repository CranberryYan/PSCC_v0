#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def safe_copy(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """复制文件：成功复制返回 True；已存在且不覆盖返回 False。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (not overwrite):
        return False
    shutil.copy2(src, dst)
    return True


def normalize_stem_for_mask(stem: str) -> str:
    """
    把 mask 的 stem 归一化成可能对应 fake 图的 id：
    去掉 forge/forged、mask、下划线等。
    """
    s = stem
    s = re.sub(r"(?i)forged?", "", s)         # 去 forge / forged
    s = re.sub(r"(?i)mask", "", s)            # 去 mask
    s = re.sub(r"[_\-\s]+", "", s)            # 去 _ - 空格
    return s.strip()


def main():
    root = Path(r"/mnt/c/datasets/COVERAGE")

    img_dir = root / "image_256"
    msk_dir = root / "mask_256"

    out_fake = root / "fake_train"
    out_au = root / "au_train"
    out_mask = root / "mask_train"
    out_fake.mkdir(parents=True, exist_ok=True)
    out_au.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    overwrite = False  # True=覆盖已存在文件

    if not img_dir.exists():
        raise FileNotFoundError(f"image folder not found: {img_dir}")
    if not msk_dir.exists():
        raise FileNotFoundError(f"mask folder not found: {msk_dir}")

    # -----------------------
    # 1) 处理 image：t 结尾 => 原图(au)，并删掉末尾 t
    # -----------------------
    au_map = {}      # base_id -> image_path (原图)
    fake_map = {}    # base_id -> image_path (篡改图)

    imgs = [p for p in img_dir.rglob("*") if is_image_file(p)]
    for p in imgs:
        stem = p.stem

        if stem.lower().endswith("t"):
            base_id = stem[:-1]  # 删除末尾 t
            au_map[base_id] = p

            # 输出到 au_train：文件名去掉 t
            dst_name = base_id + p.suffix.lower()
            safe_copy(p, out_au / dst_name, overwrite=overwrite)
        else:
            base_id = stem
            fake_map[base_id] = p

            # 输出到 fake_train：保持原名
            safe_copy(p, out_fake / p.name, overwrite=overwrite)

    # -----------------------
    # 2) 处理 mask：挑 forge，并尽量对齐 fake 名称；输出统一 .png
    # -----------------------
    masks = [p for p in msk_dir.rglob("*") if is_image_file(p)]
    forge_masks = [p for p in masks if "forge" in p.stem.lower() or "forge" in p.name.lower()]

    copied_mask = 0
    matched_mask = 0
    unmatched_masks = []

    for mp in forge_masks:
        norm = normalize_stem_for_mask(mp.stem)

        target_id = None
        # mask 对齐篡改图 fake_map（而不是 au_map）
        if norm in fake_map:
            target_id = norm
        else:
            norm2 = re.sub(r"[^0-9a-zA-Z]+", "", norm)
            if norm2 in fake_map:
                target_id = norm2

        if target_id is not None:
            # 关键改动：输出统一 .png
            dst_name = target_id + ".png"
            matched_mask += 1
        else:
            # 不匹配也统一 .png（避免保留奇怪后缀）
            dst_name = mp.stem + ".png"
            unmatched_masks.append(mp)

        # 写 png：如果原本不是 png，也没关系，直接 copy2 改名即可（内容格式仍是原图格式）
        # 如果你想“强制转码为 PNG”，我也可以给你改成 cv2 读写方式。
        if safe_copy(mp, out_mask / dst_name, overwrite=overwrite):
            copied_mask += 1

    # -----------------------
    # 3) 统计 fake 是否缺 mask（以 base_id 对齐）
    # -----------------------
    out_mask_files = [p for p in out_mask.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    out_mask_ids = {p.stem for p in out_mask_files}

    missing_mask_for_fake = []
    for fid in sorted(fake_map.keys()):
        if fid not in out_mask_ids:
            missing_mask_for_fake.append(fid)

    # -----------------------
    # 4) 报告
    # -----------------------
    report = root / "coverage_split_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"ROOT: {root}\n")
        f.write(f"Found images: {len(imgs)}\n")
        f.write(f"AU images (t removed): {len(au_map)} -> {out_au}\n")
        f.write(f"FAKE images: {len(fake_map)} -> {out_fake}\n\n")

        f.write(f"Found masks: {len(masks)}\n")
        f.write(f"Forge masks: {len(forge_masks)}\n")
        f.write(f"Copied masks: {copied_mask} -> {out_mask}\n")
        f.write(f"Matched masks renamed to fake id: {matched_mask}\n")
        f.write(f"Unmatched forge masks: {len(unmatched_masks)} (saved as <stem>.png)\n\n")

        f.write(f"Missing mask for fake images: {len(missing_mask_for_fake)}\n")
        if missing_mask_for_fake:
            f.write("List (fake base_id without mask):\n")
            for fid in missing_mask_for_fake[:200]:
                f.write(fid + "\n")
            if len(missing_mask_for_fake) > 200:
                f.write(f"... truncated, total {len(missing_mask_for_fake)}\n")

    print("Done.")
    print("AU   ->", out_au, "(filenames: t removed)")
    print("FAKE ->", out_fake)
    print("MASK ->", out_mask, "(all .png)")
    print("Report:", report)


if __name__ == "__main__":
    main()
