#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.array(im)


def load_mask_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.array(im)


def save_png_rgb(arr: np.ndarray, out_path: Path):
    ensure_dir(out_path.parent)
    Image.fromarray(arr, mode="RGB").save(out_path, format="PNG", compress_level=1)


def save_png_mask_0255(arr_u8: np.ndarray, out_path: Path):
    ensure_dir(out_path.parent)
    Image.fromarray(arr_u8, mode="L").save(out_path, format="PNG", compress_level=1)


def find_mask_for_fake(fake_path: Path, mask_dir: Path, mask_index: dict[str, Path]) -> Path | None:
    """
    依次尝试：
    1) 同名同 stem：stem.png / stem.bmp / stem.jpg ...
    2) stem + _mask / _gt / _edgemask
    3) 预建索引：mask_index[stem]
    """
    stem = fake_path.stem

    # 1) 直接同 stem
    for ext in [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"]:
        cand = mask_dir / (stem + ext)
        if cand.exists():
            return cand

    # 2) 常见后缀
    for suf in ["_mask", "_gt", "_edgemask", "_groundtruth"]:
        for ext in [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"]:
            cand = mask_dir / (stem + suf + ext)
            if cand.exists():
                return cand

    # 3) 索引
    if stem in mask_index:
        return mask_index[stem]

    return None


def build_mask_index(mask_dir: Path) -> dict[str, Path]:
    """
    建一个 stem -> path 的索引（若 stem 冲突，保留第一个）。
    """
    idx = {}
    for p in mask_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        s = p.stem
        if s not in idx:
            idx[s] = p
    return idx


def choose_crop_xy(mask: np.ndarray, crop: int) -> tuple[int, int]:
    """
    根据 mask 前景 bbox 选择 256x256 裁剪窗口左上角 (x0, y0)
    - 若 mask 无前景：居中裁剪
    - 若有前景：尽量让 bbox 落入 crop 内（或以 bbox 中心为中心）
    """
    h, w = mask.shape[:2]
    assert w >= crop and h >= crop, f"image too small for crop: {(w, h)} < {crop}"

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # 无前景：居中
        x0 = (w - crop) // 2
        y0 = (h - crop) // 2
        return x0, y0

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    # 以 bbox 中心为目标
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    x0 = int(round(cx - crop / 2))
    y0 = int(round(cy - crop / 2))

    # clamp
    x0 = max(0, min(x0, w - crop))
    y0 = max(0, min(y0, h - crop))
    return x0, y0


def pad_to_at_least(arr: np.ndarray, min_h: int, min_w: int, is_mask: bool) -> np.ndarray:
    """
    万一出现小于 256 的情况：不 resize，采用边缘 padding（不会插值破坏纹理）。
    你这个数据通常不会触发，但加上更稳。
    """
    h, w = arr.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if arr.ndim == 2:
        mode = "edge"
        return np.pad(arr, ((top, bottom), (left, right)), mode=mode)
    else:
        mode = "edge"
        return np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode=mode)


def process_one(fake_path: Path, fake_dir: Path, mask_dir: Path,
                out_fake: Path, out_mask: Path,
                mask_index: dict[str, Path],
                crop: int) -> tuple[str, str]:
    """
    返回 (status, message)
    """
    rel = fake_path.relative_to(fake_dir)
    mask_path = find_mask_for_fake(fake_path, mask_dir, mask_index)
    if mask_path is None:
        return ("MISS_MASK", f"{fake_path}")

    img = load_rgb(fake_path)
    m = load_mask_gray(mask_path)

    # 对齐大小：必须一致，否则无法裁剪同区域
    if img.shape[0] != m.shape[0] or img.shape[1] != m.shape[1]:
        return ("SIZE_MISMATCH", f"fake={fake_path} ({img.shape[1]}x{img.shape[0]}) mask={mask_path} ({m.shape[1]}x{m.shape[0]})")

    # padding（极少触发）
    img = pad_to_at_least(img, crop, crop, is_mask=False)
    m = pad_to_at_least(m, crop, crop, is_mask=True)

    h, w = m.shape[:2]
    if w < crop or h < crop:
        return ("TOO_SMALL", f"{fake_path} size={(w,h)}")

    # 选择裁剪窗口
    x0, y0 = choose_crop_xy(m, crop=crop)

    img_c = img[y0:y0 + crop, x0:x0 + crop, :]
    m_c = m[y0:y0 + crop, x0:x0 + crop]

    # 输出 mask 转 {0,255}
    m_out = (m_c > 0).astype(np.uint8) * 255

    # 输出路径：保持相对结构；统一保存 PNG 避免二次 JPEG 压缩破坏纹理
    out_fake_path = (out_fake / rel).with_suffix(".png")
    out_mask_path = (out_mask / rel).with_suffix(".png")

    save_png_rgb(img_c, out_fake_path)
    save_png_mask_0255(m_out, out_mask_path)

    return ("OK", f"{fake_path} -> {out_fake_path} | {mask_path} -> {out_mask_path} | crop@({x0},{y0})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-dir", default="/mnt/e/datasets/PSCC/Training Dataset/CASIA2.0/fake")
    ap.add_argument("--mask-dir", default="/mnt/e/datasets/PSCC/Training Dataset/CASIA2.0/mask")
    ap.add_argument("--out-fake", default="/mnt/e/datasets/PSCC/Training Dataset/CASIA2.0/fake_256")
    ap.add_argument("--out-mask", default="/mnt/e/datasets/PSCC/Training Dataset/CASIA2.0/mask_256")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report", default="crop_report.txt")
    args = ap.parse_args()

    fake_dir = Path(args.fake_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    out_fake = Path(args.out_fake).resolve()
    out_mask = Path(args.out_mask).resolve()
    crop = int(args.crop)

    if not fake_dir.exists():
        raise FileNotFoundError(fake_dir)
    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)

    ensure_dir(out_fake)
    ensure_dir(out_mask)

    # 收集 fake 文件
    fake_files = [p for p in fake_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    fake_files.sort()
    print(f"[SCAN] fake={fake_dir} | n={len(fake_files)}")
    print(f"[SCAN] mask={mask_dir}")

    # 预建 mask 索引（加速匹配）
    mask_index = build_mask_index(mask_dir)
    print(f"[INDEX] mask files indexed: {len(mask_index)}")

    ok = miss = mismatch = small = err = 0
    report_path = out_fake.parent / args.report  # 放 CASIA1.0 目录下也方便
    lines = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(process_one, fp, fake_dir, mask_dir, out_fake, out_mask, mask_index, crop)
            for fp in fake_files
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status, msg = fut.result()
                if status == "OK":
                    ok += 1
                elif status == "MISS_MASK":
                    miss += 1
                elif status == "SIZE_MISMATCH":
                    mismatch += 1
                elif status == "TOO_SMALL":
                    small += 1
                else:
                    err += 1
                lines.append(f"{status}\t{msg}")
            except Exception as e:
                err += 1
                lines.append(f"EXCEPTION\t{e}")

            if i % 500 == 0 or i == len(fake_files):
                print(f"[PROG] {i}/{len(fake_files)} | ok={ok} miss={miss} mismatch={mismatch} small={small} err={err}")

    lines.sort()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"fake_dir: {fake_dir}\n")
        f.write(f"mask_dir: {mask_dir}\n")
        f.write(f"out_fake: {out_fake}\n")
        f.write(f"out_mask: {out_mask}\n")
        f.write(f"crop: {crop}\n")
        f.write(f"count: {len(fake_files)} | ok={ok} miss={miss} mismatch={mismatch} small={small} err={err}\n\n")
        for ln in lines:
            f.write(ln + "\n")

    print(f"[DONE] ok={ok} miss={miss} mismatch={mismatch} small={small} err={err}")
    print(f"[DONE] report -> {report_path}")
    print(f"[DONE] outputs -> {out_fake} , {out_mask}")


if __name__ == "__main__":
    main()
