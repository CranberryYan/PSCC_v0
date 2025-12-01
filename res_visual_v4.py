#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import imageio.v2 as imageio


# ========================= 工具函数 =========================
_THRESH = 255 * 0.90

def load_binary_mask(path: str) -> np.ndarray:
    """读取并二值化 mask, 返回 uint8 的 2D (0/1)"""
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask >= _THRESH).astype(np.uint8)


def make_black_mask_like(ref: np.ndarray) -> np.ndarray:
    """生成与 ref 同尺寸的全黑 mask"""
    return np.zeros_like(ref, dtype=np.uint8)


def _to_three_channel_uint8(img: np.ndarray) -> np.ndarray:
    """灰度 / RGBA → H×W×3 uint8"""
    if img.ndim == 2:
        img = np.broadcast_to(img[:, :, None], (img.shape[0], img.shape[1], 3))
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _align_min_hw(a2d: np.ndarray, b2d: np.ndarray):
    h = min(a2d.shape[0], b2d.shape[0])
    w = min(a2d.shape[1], b2d.shape[1])
    return a2d[:h, :w], b2d[:h, :w]


def _metrics_from_tp_fp_fn(tp: int, fp: int, fn: int, *, gt_pos: int, pred_pos: int):
    """
    像素级 P/R/F1/IoU
    - 若 GT 无正样本(gt_pos==0):
        * pred_pos==0 => 视为完全正确, P=R=F1=IoU=1
        * pred_pos>0  => 纯 FP, P=R=F1=IoU=0
    """
    if gt_pos == 0:
        if pred_pos == 0:
            return 1.0, 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0, 0.0

    # GT 有正样本：按常规定义
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return p, r, f1, iou


# ========================= 主流程 =========================
def compute_f1_scores(
    ref_txt,
    pred_root,
    save_path='f1_results.txt',
    case_root='cases',
    bad_threshold=0.55,
    good_threshold=0.85,
    max_workers=None,
):
    """
    - 若 GT mask 不存在 → 视为原图，自动生成全黑 mask
    - 统计像素级 P / R / F1 / IoU
    - good / normal / bad 仅按 F1 划分
    """

    pred_root = Path(pred_root)
    case_root = Path(case_root)

    if case_root.exists():
        shutil.rmtree(case_root)

    good_dir = case_root / 'good_cases'
    normal_dir = case_root / 'normal_cases'
    bad_dir = case_root / 'bad_cases'
    good_dir.mkdir(parents=True)
    normal_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)

    with open(ref_txt, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if max_workers is None:
        cpu = os.cpu_count() or 8
        max_workers = min(16, cpu * 2)

    def _process_one(idx_line):
        idx, line = idx_line
        img_path = Path(line)
        stem = img_path.stem

        pred_path = pred_root / f"{stem}.png"
        if not pred_path.exists():
            return idx, None, f"预测 mask 不存在: {pred_path}"

        y_pred = load_binary_mask(str(pred_path))

        ref_path = Path(str(img_path).replace('fake', 'mask')).with_suffix('.png')
        if ref_path.exists():
            y_true = load_binary_mask(str(ref_path))
        else:
            y_true = make_black_mask_like(y_pred)

        y_true, y_pred = _align_min_hw(y_true, y_pred)

        t = y_true.astype(bool)
        p = y_pred.astype(bool)

        tp = int(np.count_nonzero(t & p))
        fp = int(np.count_nonzero((~t) & p))
        fn = int(np.count_nonzero(t & (~p)))

        gt_pos = int(np.count_nonzero(t))
        pred_pos = int(np.count_nonzero(p))

        prec, rec, f1, iou = _metrics_from_tp_fp_fn(tp, fp, fn, gt_pos=gt_pos, pred_pos=pred_pos)

        # 同时返回 total_pixels 和 tn，便于全体像素统计(包含TN的像素准确率等)
        total_pixels = int(t.size)
        tn = total_pixels - tp - fp - fn

        if f1 < bad_threshold:
            target_dir, tag = bad_dir, 'bad'
        elif f1 >= good_threshold:
            target_dir, tag = good_dir, 'good'
        else:
            target_dir, tag = normal_dir, 'normal'

        if img_path.exists():
            orig = _to_three_channel_uint8(imageio.imread(str(img_path)))
            gt = _to_three_channel_uint8((y_true * 255).astype(np.uint8))
            pd = _to_three_channel_uint8((y_pred * 255).astype(np.uint8))

            h = min(orig.shape[0], gt.shape[0], pd.shape[0])
            w = min(orig.shape[1], gt.shape[1], pd.shape[1])
            concat = np.concatenate([orig[:h, :w], gt[:h, :w], pd[:h, :w]], axis=1)
            imageio.imwrite(str(target_dir / f"concat_{stem}.png"), concat)

        return idx, (line, prec, rec, f1, iou, tp, fp, fn, tn, total_pixels, tag), None

    results, errors = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one, (i, l)) for i, l in enumerate(lines)]
        for fu in as_completed(futures):
            idx, payload, err = fu.result()
            if err:
                errors.append(err)
            elif payload:
                results.append((idx, payload))

    results.sort(key=lambda x: x[0])

    micro_tp = micro_fp = micro_fn = micro_tn = 0
    total_pix = 0

    out_lines = []

    for _, (line, p, r, f1, iou, tp, fp, fn, tn, n_pix, tag) in results:
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        micro_tn += tn
        total_pix += n_pix

        out_lines.append(
            f"{line}\tP:{p:.4f}\tR:{r:.4f}\tF1:{f1:.4f}\tIoU:{iou:.4f}\n"
        )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

        f.write("\n")
        mp = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        mr = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        mf = (2 * micro_tp) / (2 * micro_tp + micro_fp + micro_fn) if (2 * micro_tp + micro_fp + micro_fn) > 0 else 0.0
        miou = micro_tp / (micro_tp + micro_fp + micro_fn) if (micro_tp + micro_fp + micro_fn) > 0 else 0.0

        f.write(
            f"Global(Pixel-Micro) P/R/F1/IoU: "
            f"{mp:.4f} / {mr:.4f} / {mf:.4f} / {miou:.4f}\n"
        )

        # 可选：像素准确率(包含TN)，真正“以全体像素数”为分母
        if total_pix > 0:
            acc = (micro_tp + micro_tn) / total_pix
            f.write(f"Global PixelAcc: {acc:.4f}\n")

        if errors:
            f.write("\n[WARN]\n")
            for e in errors:
                f.write(e + "\n")

        print("完成评估")
        print(f"样本数: {len(results)}, 错误: {len(errors)}")


# ========================= 入口 =========================
if __name__ == "__main__":
    compute_f1_scores(
        # ref_txt="./datasets_CASIA1.0.txt",
        ref_txt="./datasets_CASIA1.0.txt",
        pred_root="mask_results/result/",
        save_path="f1_results.txt",
        case_root="cases_all",
        bad_threshold=0.55,
        good_threshold=0.85,
        max_workers=16,
    )
