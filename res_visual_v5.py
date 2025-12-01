#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import imageio.v2 as imageio


# ============== 跟你的 val 保持一致的阈值 ==============
MASK_F1_THRESH = 0.90  # TODO: 改成你 val 里同一个 MASK_F1_THRESH


# ========================= 工具函数 =========================
def _read_gray(path: str) -> np.ndarray:
    """读为 2D 灰度"""
    m = imageio.imread(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def load_prob_mask_01(path: str) -> np.ndarray:
    """
    读取预测 mask，并转成 0~1 的概率图 (float32)
    兼容:
      - uint8 0~255
      - float 0~1
    """
    m = _read_gray(path)
    m = m.astype(np.float32)

    # 若是 0~255 的 uint8/float，归一化到 0~1
    if m.max() > 1.5:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)
    return m


def load_gt_mask_01(path: str) -> np.ndarray:
    """
    读取 GT mask，转成 0~1 (float32)
    兼容:
      - 0/255
      - 0/1
      - 其它灰度
    """
    m = _read_gray(path).astype(np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)
    return m


def make_black_mask_like(ref_2d: np.ndarray) -> np.ndarray:
    """生成与 ref 同尺寸的全黑 mask (float32 0~1)"""
    return np.zeros_like(ref_2d, dtype=np.float32)


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


def f1_binary_special(pred01: np.ndarray, gt01: np.ndarray) -> float:
    """
    对齐你 val 里 per-image f1 的 zero_division=1 逻辑：
      - gt 全 0 且 pred 全 0 => f1=1
      - gt 全 0 且 pred 有 1 => f1=0
      - 否则常规 f1
    """
    gt_pos = int(gt01.sum())
    pred_pos = int(pred01.sum())
    if gt_pos == 0:
        return 1.0 if pred_pos == 0 else 0.0

    tp = int(np.count_nonzero((gt01 == 1) & (pred01 == 1)))
    fp = int(np.count_nonzero((gt01 == 0) & (pred01 == 1)))
    fn = int(np.count_nonzero((gt01 == 1) & (pred01 == 0)))
    denom = (2 * tp + fp + fn)
    return (2 * tp) / denom if denom > 0 else 0.0


# ========================= 主流程 =========================
def compute_val_style_metrics(
    ref_txt,
    pred_root,
    save_path='val_style_results.txt',
    case_root='cases',
    bad_threshold=0.55,
    good_threshold=0.85,
    max_workers=None,
):
    """
    完全对齐你 val 的统计口径：
      - pred: 读概率图(0~1) -> pred = prob > MASK_F1_THRESH
      - gt:   读 0~1 -> gt = gt > 0.5
      - micro: 全体像素 TP/FP/FN/TN
      - fake-only: 只统计 gt_pos>0 的图的 TP/FP/FN
      - auth fp rate: gt_pos==0 的图上 pred 为 1 的像素 / 总像素
      - pixel_acc: (TP+TN)/total_pixels
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

    def infer_gt_path(img_path: Path) -> Path:
        """
        尽量兼容你项目里常见映射：
          - .../fake/... -> .../mask/...
          - 文件名中 fake->mask
          - 后缀改为 .png
        """
        s = str(img_path)

        # 优先替换目录段
        if "/fake/" in s:
            s2 = s.replace("/fake/", "/mask/")
        else:
            s2 = s.replace("fake", "mask")

        return Path(s2).with_suffix(".png")

    def _process_one(idx_line):
        idx, line = idx_line
        img_path = Path(line)
        stem = img_path.stem

        # pred
        pred_path = pred_root / f"{stem}.png"
        if not pred_path.exists():
            return idx, None, f"预测 mask 不存在: {pred_path}"

        prob = load_prob_mask_01(str(pred_path))              # 0~1
        pred01 = (prob > MASK_F1_THRESH).astype(np.uint8)     # 0/1

        # gt
        gt_path = infer_gt_path(img_path)
        if gt_path.exists():
            gt01 = (load_gt_mask_01(str(gt_path)) > 0.5).astype(np.uint8)
        else:
            gt01 = make_black_mask_like(prob).astype(np.uint8)

        # align
        gt01, pred01 = _align_min_hw(gt01, pred01)

        t = gt01.astype(bool)
        p = pred01.astype(bool)

        tp = int(np.count_nonzero(t & p))
        fp = int(np.count_nonzero((~t) & p))
        fn = int(np.count_nonzero(t & (~p)))
        total_pixels = int(t.size)
        tn = total_pixels - tp - fp - fn

        gt_pos = int(np.count_nonzero(t))
        pred_pos = int(np.count_nonzero(p))

        # per-image f1（用于 good/normal/bad 分桶，与 val 的 zero_division=1 对齐）
        f1_i = f1_binary_special(pred01, gt01)

        if f1_i < bad_threshold:
            target_dir, tag = bad_dir, 'bad'
        elif f1_i >= good_threshold:
            target_dir, tag = good_dir, 'good'
        else:
            target_dir, tag = normal_dir, 'normal'

        # 保存 concat（原图/GT/Pred）
        if img_path.exists():
            orig = _to_three_channel_uint8(imageio.imread(str(img_path)))
            gt_u8 = _to_three_channel_uint8((gt01 * 255).astype(np.uint8))
            pd_u8 = _to_three_channel_uint8((pred01 * 255).astype(np.uint8))

            h = min(orig.shape[0], gt_u8.shape[0], pd_u8.shape[0])
            w = min(orig.shape[1], gt_u8.shape[1], pd_u8.shape[1])
            concat = np.concatenate([orig[:h, :w], gt_u8[:h, :w], pd_u8[:h, :w]], axis=1)
            imageio.imwrite(str(target_dir / f"concat_{stem}.png"), concat)

        return idx, (line, tp, fp, fn, tn, total_pixels, gt_pos, tag, f1_i), None

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

    # ======= 对齐你 val 的累积统计 =======
    tp = fp = fn = tn = 0
    total_pix = 0

    tp_f = fp_f = fn_f = 0          # fake-only
    fp_auth = 0
    tot_auth = 0

    out_lines = []

    for _, (line, _tp, _fp, _fn, _tn, n_pix, gt_pos, tag, f1_i) in results:
        tp += _tp
        fp += _fp
        fn += _fn
        tn += _tn
        total_pix += n_pix

        # fake-only vs auth-only
        if gt_pos > 0:
            tp_f += _tp
            fp_f += _fp
            fn_f += _fn
        else:
            fp_auth += _fp
            tot_auth += n_pix

        out_lines.append(f"{line}\tF1_img:{f1_i:.4f}\ttag:{tag}\n")

    # micro 指标
    seg_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    seg_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    seg_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    seg_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # fake-only F1
    seg_f1_fake = (2 * tp_f) / (2 * tp_f + fp_f + fn_f) if (2 * tp_f + fp_f + fn_f) > 0 else 0.0

    # auth 上误检率
    auth_fp_rate = fp_auth / tot_auth if tot_auth > 0 else 0.0

    # pixel acc
    pixel_acc = (tp + tn) / total_pix if total_pix > 0 else 0.0

    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
        f.write("\n")
        f.write(
            f"Validation: SegF1(micro)={seg_f1:.4f}, P={seg_p:.4f}, R={seg_r:.4f}, IoU={seg_iou:.4f} | "
            f"FakeF1={seg_f1_fake:.4f}, AuthFP={auth_fp_rate:.6f}, PixelAcc={pixel_acc:.4f}\n"
        )
        if errors:
            f.write("\n[WARN]\n")
            for e in errors:
                f.write(e + "\n")

    print("完成评估")
    print(f"样本数: {len(results)}, 错误: {len(errors)}")
    print(
        f"Validation: SegF1(micro)={seg_f1:.4f}, P={seg_p:.4f}, R={seg_r:.4f}, IoU={seg_iou:.4f} | "
        f"FakeF1={seg_f1_fake:.4f}, AuthFP={auth_fp_rate:.6f}, PixelAcc={pixel_acc:.4f}"
    )


if __name__ == "__main__":
    compute_val_style_metrics(
        ref_txt="./datasets_CASIA2.0_allTemp.txt",
        pred_root="mask_results/result/",
        save_path="val_style_results.txt",
        case_root="cases_all",
        bad_threshold=0.55,
        good_threshold=0.85,
        max_workers=16,
    )
