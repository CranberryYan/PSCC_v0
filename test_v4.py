#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from utils.config import get_pscc_args
from utils.load_vdata import TestData

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============== 你 val 里用的阈值（保持一致） ==============
MASK_F1_THRESH = 0.95     # 像素级二值化阈值：pred_prob > MASK_F1_THRESH
CLS_THRESH = 0.50         # 图像级判别阈值：forged_prob >= CLS_THRESH -> forged(1)

# 可视化分桶阈值（按 per-image F1）
BAD_THRESHOLD = 0.55
GOOD_THRESHOLD = 0.85


# ===================== checkpoint =====================
def load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet):
    ckpt = torch.load(ckpt_path, map_location=device)
    print("Loaded checkpoint:", ckpt_path)
    FENet.load_state_dict(ckpt["FENet"])
    SegNet.load_state_dict(ckpt["SegNet"])
    ClsNet.load_state_dict(ckpt["ClsNet"], strict=False)


# ========================= I/O utils =========================
def _read_gray(path: str) -> np.ndarray:
    m = imageio.imread(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def load_gt_mask_01(path: str) -> np.ndarray:
    """读取GT mask -> float32 0~1"""
    m = _read_gray(path).astype(np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


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


def infer_gt_path(img_path: Path) -> Path:
    """
    尽量兼容常见映射：
      - .../fake/... -> .../mask/...
      - 否则把字符串里的 fake 替换为 mask
      - 后缀改为 .png
    """
    s = str(img_path).replace("\\", "/")
    if "/fake/" in s:
        s2 = s.replace("/fake/", "/mask/")
    else:
        s2 = s.replace("fake", "mask")
    return Path(s2).with_suffix(".png")


def save_prob_png(prob01: np.ndarray, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(prob01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    imageio.imwrite(str(out_png), u8)


# ========================= 主流程 =========================
def test_joint(args):
    # ---------- 输出目录 ----------
    out_root = Path("./test_joint_out")
    pred_root = out_root / "mask_results" / "result"   # 与你第二段脚本的 pred_root 习惯一致
    case_root = out_root / "cases_all"
    log_path = out_root / "val_style_results.txt"

    if out_root.exists():
        shutil.rmtree(out_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    good_dir = case_root / "good_cases"
    normal_dir = case_root / "normal_cases"
    bad_dir = case_root / "bad_cases"
    good_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 网络 ----------
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg).to(device)
    SegNet = NLCDetection(args).to(device)
    ClsNet = DetectionHead(args).to(device)

    # ---------- checkpoint（按你的路径） ----------
    ckpt_path = (
        "./total=15000_epochs=100_batch=6_with_MoE=False(MoE_attn=False_K=8)_with_HiLo=False/checkpoint/NLCDetection_checkpoint/NLCDetection_best.pth"
    )
    load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet)

    FENet.eval()
    SegNet.eval()
    ClsNet.eval()

    # ---------- DataLoader ----------
    test_loader = DataLoader(
        TestData(args),
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
    )

    # ---------- 图像级指标缓存 ----------
    y_true = []
    y_score = []
    y_pred = []

    # ---------- 像素级统计（micro + fake-only + auth fp rate） ----------
    TP = FP = FN = TN = 0
    TP_f = FP_f = FN_f = 0
    FP_auth = 0
    TOT_auth_pix = 0
    TOTAL_pix = 0

    missing_imgs = []
    missing_gts = []

    out_lines = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing+Visualizing", ncols=100):
            images, _, names = batch
            images = images.to(device)
            B = images.size(0)

            # ===== forward =====
            feat = FENet(images)
            # pred_masks, feats, _ = SegNet(feat)
            pred_masks, feats = SegNet(feat)
            pred_logits = ClsNet(feats)
            pred_probs = torch.softmax(pred_logits, dim=1)  # [B,2]
            forged_probs = pred_probs[:, 1]                 # [B]

            # 取分割输出（兼容 list/tuple）
            if isinstance(pred_masks, (list, tuple)):
                seg_prob = pred_masks[0]   # [B,1,h,w]
            else:
                seg_prob = pred_masks      # [B,1,h,w]

            for i in range(B):
                name_i = names[i]
                name_str = str(name_i)

                # ---- 图像级 score / pred ----
                prob_i = float(forged_probs[i].item())
                if not np.isfinite(prob_i):
                    print(f"[WARN] NaN/Inf score, skip: {name_str}")
                    continue

                pred_cls = int(prob_i >= CLS_THRESH)  # 1 forged, 0 authentic

                # ---- 图像级 GT（按你原逻辑）----
                lower = name_str.lower()
                if ("authentic" in lower) or ("/au" in lower) or ("\\au" in lower) or ("au_" in lower) or ("4cam_auth_256" in lower) or ("au_train" in lower):
                    gt_cls = 0
                else:
                    gt_cls = 1

                y_true.append(gt_cls)
                y_score.append(prob_i)
                y_pred.append(pred_cls)

                # ---- 像素级：若图像级判别为原图，则直接全黑 mask ----
                # 先把 seg_prob up 到当前输入尺寸
                prob_map = seg_prob[i:i + 1].float()  # [1,1,h,w]
                prob_up = F.interpolate(
                    prob_map,
                    size=(images.size(2), images.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )  # [1,1,H,W]

                if pred_cls == 0:
                    prob_up.zero_()  # 关键：图像级判别为原图 -> 像素级全黑

                prob_up_01 = prob_up[0, 0].detach().cpu().numpy().astype(np.float32)  # [H,W], 0~1(约)

                # ---- 保存预测 mask（按 stem.png 平铺）----
                stem = Path(name_str).stem
                pred_png = pred_root / f"{stem}.png"
                save_prob_png(prob_up_01, pred_png)

                # ---- GT mask 读取（不存在则全黑）----
                img_path = Path(name_str)
                gt_path = infer_gt_path(img_path)
                if gt_path.exists():
                    gt01 = (load_gt_mask_01(str(gt_path)) > 0.5).astype(np.uint8)
                else:
                    # GT 不存在：视为全黑
                    gt01 = np.zeros_like(prob_up_01, dtype=np.uint8)
                    missing_gts.append(str(gt_path))

                pred01 = (prob_up_01 > MASK_F1_THRESH).astype(np.uint8)

                # ---- 对齐尺寸 ----
                gt01, pred01 = _align_min_hw(gt01, pred01)
                t = gt01.astype(bool)
                p = pred01.astype(bool)

                _tp = int(np.count_nonzero(t & p))
                _fp = int(np.count_nonzero((~t) & p))
                _fn = int(np.count_nonzero(t & (~p)))
                n_pix = int(t.size)
                _tn = n_pix - _tp - _fp - _fn

                TP += _tp
                FP += _fp
                FN += _fn
                TN += _tn
                TOTAL_pix += n_pix

                gt_pos = int(np.count_nonzero(t))
                if gt_pos > 0:
                    TP_f += _tp
                    FP_f += _fp
                    FN_f += _fn
                else:
                    FP_auth += _fp
                    TOT_auth_pix += n_pix

                # ---- per-image F1 分桶 ----
                f1_i = f1_binary_special(pred01, gt01)
                if f1_i < BAD_THRESHOLD:
                    target_dir, tag = bad_dir, "bad"
                elif f1_i >= GOOD_THRESHOLD:
                    target_dir, tag = good_dir, "good"
                else:
                    target_dir, tag = normal_dir, "normal"

                out_lines.append(
                    f"{name_str}\tcls_gt:{gt_cls}\tcls_prob:{prob_i:.6f}\tcls_pred:{pred_cls}"
                    f"\tF1_img:{f1_i:.4f}\ttag:{tag}\n"
                )

                # ---- 保存 concat 可视化（原图/GT/Pred）----
                if img_path.exists():
                    try:
                        orig = _to_three_channel_uint8(imageio.imread(str(img_path)))
                        gt_u8 = _to_three_channel_uint8((gt01 * 255).astype(np.uint8))
                        pd_u8 = _to_three_channel_uint8((pred01 * 255).astype(np.uint8))

                        h = min(orig.shape[0], gt_u8.shape[0], pd_u8.shape[0])
                        w = min(orig.shape[1], gt_u8.shape[1], pd_u8.shape[1])
                        concat = np.concatenate(
                            [orig[:h, :w], gt_u8[:h, :w], pd_u8[:h, :w]],
                            axis=1
                        )
                        imageio.imwrite(str(target_dir / f"concat_{stem}.png"), concat)
                    except Exception as e:
                        print(f"[WARN] concat save failed: {img_path} ({e})")
                        missing_imgs.append(str(img_path))
                else:
                    missing_imgs.append(str(img_path))

    # ===================== 计算图像级指标 =====================
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float32)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    img_metrics_ok = (y_true_np.size > 0) and (np.unique(y_true_np).size == 2)

    if img_metrics_ok:
        auc = roc_auc_score(y_true_np, y_score_np)
        acc = accuracy_score(y_true_np, y_pred_np)
        p_img = precision_score(y_true_np, y_pred_np)
        r_img = recall_score(y_true_np, y_pred_np)
        f1_img = f1_score(y_true_np, y_pred_np)
    else:
        auc = acc = p_img = r_img = f1_img = float("nan")

    # ===================== 计算像素级指标（micro） =====================
    seg_p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    seg_r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    seg_f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    seg_iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    seg_f1_fake = (2 * TP_f) / (2 * TP_f + FP_f + FN_f) if (2 * TP_f + FP_f + FN_f) > 0 else 0.0
    auth_fp_rate = FP_auth / TOT_auth_pix if TOT_auth_pix > 0 else 0.0
    pixel_acc = (TP + TN) / TOTAL_pix if TOTAL_pix > 0 else 0.0

    # ===================== 写日志 =====================
    out_root = Path("./test_joint_out")
    out_root.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
        f.write("\n========== Image-level Evaluation ==========\n")
        f.write(f"AUC : {auc:.4f}\n")
        f.write(f"ACC : {acc:.4f}\n")
        f.write(f"P   : {p_img:.4f}\n")
        f.write(f"R   : {r_img:.4f}\n")
        f.write(f"F1  : {f1_img:.4f}\n")
        f.write("===========================================\n\n")

        f.write(
            "========== Pixel-level Evaluation (val-style) ==========\n"
            f"SegF1(micro)={seg_f1:.4f}, P={seg_p:.4f}, R={seg_r:.4f}, IoU={seg_iou:.4f} | "
            f"FakeF1={seg_f1_fake:.4f}, AuthFP={auth_fp_rate:.6f}, PixelAcc={pixel_acc:.4f}\n"
            "======================================================\n"
        )

        if missing_imgs:
            f.write("\n[WARN] missing images for concat:\n")
            for x in missing_imgs:
                f.write(x + "\n")

        # gt 缺失可能很多（auth 天然没gt），这里给一个去重后的数量提示
        if missing_gts:
            f.write(f"\n[INFO] missing gt masks count (including authentic): {len(missing_gts)}\n")

    # ===================== 打印汇总 =====================
    print("\n========== Image-level Evaluation ==========")
    if img_metrics_ok:
        print(f"AUC : {auc:.4f}")
        print(f"ACC : {acc:.4f}")
        print(f"P   : {p_img:.4f}")
        print(f"R   : {r_img:.4f}")
        print(f"F1  : {f1_img:.4f}")
    else:
        print("[WARN] Image-level metrics skipped (need both classes in test set).")
    print("===========================================\n")

    print("========== Pixel-level Evaluation (val-style) ==========")
    print(
        f"SegF1(micro)={seg_f1:.4f}, P={seg_p:.4f}, R={seg_r:.4f}, IoU={seg_iou:.4f} | "
        f"FakeF1={seg_f1_fake:.4f}, AuthFP={auth_fp_rate:.6f}, PixelAcc={pixel_acc:.4f}"
    )
    print("======================================================\n")

    print("输出目录：")
    print(f"  - 预测mask: {pred_root}")
    print(f"  - 可视化cases: {case_root}")
    print(f"  - 日志: {log_path}")


if __name__ == "__main__":
    args = get_pscc_args()
    test_joint(args)
