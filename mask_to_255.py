#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_mask_like_path(p: Path) -> bool:
    """默认只处理 mask：避免把 fake 图也二值化"""
    s = str(p).lower()
    stem = p.stem.lower()
    # 路径包含 /mask/ 或文件名包含 mask/edgemask/gt 等关键词
    return ("/mask/" in s) or ("mask" in stem) or ("edgemask" in s) or ("groundtruth" in s) or ("gt" in stem)


def read_list(txt_path: Path):
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip().strip('"').strip("'")
            if not x or x.startswith("#"):
                continue
            lines.append(x)
    return lines


def otsu_threshold_u8(x: np.ndarray) -> int:
    """x: uint8 2D"""
    hist = np.bincount(x.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thr = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thr = t
    return int(thr)


def pick_threshold_and_polarity(mask_u8: np.ndarray, target_ratio: float = 0.2):
    """
    对“灰度 mask”(比如 min>0, max<255)做二值化：
    - 候选阈值：Otsu / (众数 + 最大值)/2
    - 候选极性：前景=高值(>) 或 前景=低值(<)
    选择一个前景比例不极端且更接近 target_ratio 的方案
    """
    hist = np.bincount(mask_u8.ravel(), minlength=256)
    mode = int(hist.argmax())
    vmax = int(mask_u8.max())
    vmin = int(mask_u8.min())

    t_otsu = otsu_threshold_u8(mask_u8)
    t_mid = int((mode + vmax) * 0.5)

    candidates = []
    for name, t in [("otsu", t_otsu), ("mode_mid", t_mid)]:
        # 高值为前景
        fg_hi = mask_u8 > t
        r_hi = float(fg_hi.mean())
        candidates.append((name, t, "hi", fg_hi, r_hi))

        # 低值为前景
        fg_lo = mask_u8 < t
        r_lo = float(fg_lo.mean())
        candidates.append((name, t, "lo", fg_lo, r_lo))

    # 过滤极端比例
    usable = [c for c in candidates if 0.001 < c[4] < 0.999]
    if not usable:
        # 兜底：用中点阈值，高值前景
        fg = mask_u8 > t_mid
        return ("mode_mid", t_mid, "hi", fg, float(fg.mean()), mode, vmin, vmax)

    def score(r):
        # 越接近 target_ratio 越好，太大/太小都差
        return abs(r - target_ratio)

    best = sorted(usable, key=lambda x: score(x[4]))[0]
    return (*best, mode, vmin, vmax)  # (method, thr, polarity, fg, ratio, mode, vmin, vmax)


def to_u8_2d(arr):
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    # float/16bit 统一裁剪到 0~255
    a = arr.astype(np.float32)
    a = np.clip(a, 0.0, 255.0)
    return a.astype(np.uint8)


def save_png(arr_u8: np.ndarray, out_path: Path, compress_level: int = 1):
    ensure_dir(out_path.parent)
    Image.fromarray(arr_u8, mode="L").save(out_path, format="PNG", compress_level=compress_level)


def process_one(p: Path, out_mode: str, compress_level: int, target_ratio: float):
    # 读取
    with Image.open(p) as im:
        im = im.convert("L")
        m = np.array(im)

    mmin = float(m.min())
    mmax = float(m.max())

    # 统计 unique（只取 top 10）
    uniq, cnt = np.unique(m, return_counts=True)
    order = np.argsort(-cnt)
    uniq_top = uniq[order][:10].tolist()
    cnt_top = cnt[order][:10].tolist()

    # 决策：如果本来就接近二值(含 0/255)，走简单路径；否则按灰度 mask 处理
    m_u8 = to_u8_2d(m)

    # 判断是否已经是“近似二值”
    u = np.unique(m_u8)
    if len(u) <= 3 and (0 in u) and (255 in u):
        fg = m_u8 > 0
        method, thr, pol, ratio, mode, vmin, vmax = ("binary_like", -1, "hi", float(fg.mean()), int(u[0]), int(mmin), int(mmax))
    else:
        method, thr, pol, fg, ratio, mode, vmin, vmax = pick_threshold_and_polarity(m_u8, target_ratio=target_ratio)

    out = (fg.astype(np.uint8) * 255)

    # 输出路径
    if out_mode == "suffix":
        out_path = p.with_name(p.stem + "_0255.png")
    elif out_mode == "inplace":
        out_path = p.with_suffix(".png")  # 强制 png
    else:
        raise ValueError("out_mode must be suffix or inplace")

    save_png(out, out_path, compress_level=compress_level)

    return {
        "path": str(p),
        "dtype": str(m.dtype),
        "shape": tuple(m.shape),
        "min": mmin,
        "max": mmax,
        "uniq_top": list(zip(map(int, uniq_top), map(int, cnt_top))),
        "method": method,
        "thr": int(thr) if thr is not None else -1,
        "pol": pol,
        "fg_ratio": float(ratio),
        "out_unique": np.unique(out).tolist(),
        "out": str(out_path),
        "mode": int(mode),
        "vmin": int(vmin),
        "vmax": int(vmax),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="./datasets_Columbia.txt", help="txt 文件路径（里面每行一个路径）")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--compress", type=int, default=1, help="PNG compress_level: 0最快,9最慢")
    ap.add_argument("--target-ratio", type=float, default=0.2, help="期望前景比例（用于阈值方案选择）")
    ap.add_argument("--out-mode", choices=["suffix", "inplace"], default="inplace",
                    help="suffix: 生成 *_0255.png；inplace: 覆盖（不建议）")
    ap.add_argument("--process-all-lines", action="store_true",
                    help="默认只处理像 mask 的路径；加这个会处理 txt 中所有存在的图片路径")
    ap.add_argument("--report", default="mask_convert_report.txt")
    args = ap.parse_args()

    txt_path = Path(args.txt).resolve()
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    raw = read_list(txt_path)
    paths = []
    for s in raw:
        p = Path(s)
        # 允许相对路径
        if not p.is_absolute():
            p = (txt_path.parent / p).resolve()
        if not p.exists():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if (not args.process_all_lines) and (not is_mask_like_path(p)):
            continue
        paths.append(p)

    paths = sorted(set(paths))
    print(f"[SCAN] txt={txt_path} | candidates={len(raw)} | masks_to_process={len(paths)}")

    ok, err = 0, 0
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, p, args.out_mode, args.compress, args.target_ratio) for p in paths]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                results.append(r)
                ok += 1
            except Exception as e:
                err += 1
                if err <= 20:
                    print("[ERR]", e)

            if i % 200 == 0 or i == len(paths):
                print(f"[PROG] {i}/{len(paths)} | ok={ok} err={err}")

    # 写报告
    report_path = txt_path.parent / args.report
    results.sort(key=lambda x: x["path"])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"txt: {txt_path}\n")
        f.write(f"processed: {len(results)}, ok={ok}, err={err}\n")
        f.write(f"out_mode: {args.out_mode}, compress: {args.compress}, workers: {args.workers}\n")
        f.write(f"target_ratio: {args.target_ratio}\n\n")
        for r in results:
            f.write(
                f'{r["path"]} | dtype={r["dtype"]} shape={r["shape"]} '
                f'min={r["min"]} max={r["max"]} mode={r["mode"]} vmin={r["vmin"]} vmax={r["vmax"]} '
                f'| method={r["method"]} thr={r["thr"]} pol={r["pol"]} fg_ratio={r["fg_ratio"]:.6f} '
                f'| uniq_top={r["uniq_top"]} | out_unique={r["out_unique"]} | out={r["out"]}\n'
            )

    print(f"[DONE] ok={ok} err={err}")
    print(f"[DONE] report -> {report_path}")
    if ok:
        print("[DONE] example:")
        eg = results[0]
        print(" ", eg["path"], "->", eg["out"], "|", eg["method"], eg["thr"], eg["pol"], eg["fg_ratio"])


if __name__ == "__main__":
    main()
