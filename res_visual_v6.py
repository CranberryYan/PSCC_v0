import cv2
import numpy as np
from pathlib import Path
import imageio.v2 as imageio

MASK_F1_THRESH = 0.5  # 和训练 val 一致

def load_prob_01(path: str) -> np.ndarray:
    m = imageio.imread(path)
    if m.ndim == 3: m = m[:, :, 0]
    m = m.astype(np.float32)
    if m.max() > 1.5: m = m / 255.0
    return np.clip(m, 0.0, 1.0)

def load_gt01(path: str) -> np.ndarray:
    m = imageio.imread(path)
    if m.ndim == 3: m = m[:, :, 0]
    m = m.astype(np.float32)
    if m.max() > 1.5: m = m / 255.0
    return (m > 0.5).astype(np.uint8)

def resize_prob_to(prob: np.ndarray, h: int, w: int) -> np.ndarray:
    if prob.shape[:2] == (h, w):
        return prob
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

def eval_from_hard_val_lists(hard_fake_list, hard_auth_list, pred_root):
    pred_root = Path(pred_root)

    # 读取 hard fake: cls \t img \t mask
    fake_items = []
    with open(hard_fake_list, "r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split("\t")
            if len(sp) < 3: 
                continue
            _, img_p, gt_p = sp[0], sp[1], sp[2]
            fake_items.append((img_p, gt_p))

    # 读取 hard auth: 0 \t img
    auth_items = []
    with open(hard_auth_list, "r", encoding="utf-8") as f:
        for line in f:
            sp = line.strip().split("\t")
            if len(sp) < 2:
                continue
            _, img_p = sp[0], sp[1]
            auth_items.append(img_p)

    tp=fp=fn=tn=0
    tp_f=fp_f=fn_f=0
    fp_auth=0
    tot_auth=0
    total_pix=0

    # ---------- fake ----------
    for img_p, gt_p in fake_items:
        stem = Path(img_p).stem
        pred_p = pred_root / f"{stem}.png"
        if not pred_p.exists():
            continue

        prob = load_prob_01(str(pred_p))
        gt01 = load_gt01(gt_p)
        H, W = gt01.shape

        prob = resize_prob_to(prob, H, W)
        pred01 = (prob > MASK_F1_THRESH).astype(np.uint8)

        t = gt01.astype(bool)
        p = pred01.astype(bool)

        _tp = int(np.count_nonzero(t & p))
        _fp = int(np.count_nonzero((~t) & p))
        _fn = int(np.count_nonzero(t & (~p)))
        n_pix = int(t.size)
        _tn = n_pix - _tp - _fp - _fn

        tp += _tp; fp += _fp; fn += _fn; tn += _tn
        tp_f += _tp; fp_f += _fp; fn_f += _fn
        total_pix += n_pix

    # ---------- auth ----------
    for img_p in auth_items:
        stem = Path(img_p).stem
        pred_p = pred_root / f"{stem}.png"
        if not pred_p.exists():
            continue

        prob = load_prob_01(str(pred_p))
        H, W = prob.shape
        # gt 全 0
        pred01 = (prob > MASK_F1_THRESH).astype(np.uint8)

        _fp = int(pred01.sum())
        n_pix = int(pred01.size)
        _tn = n_pix - _fp

        fp += _fp; tn += _tn
        fp_auth += _fp; tot_auth += n_pix
        total_pix += n_pix

    seg_p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    seg_r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    seg_f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
    seg_iou = tp/(tp+fp+fn) if (tp+fp+fn)>0 else 0.0

    fake_f1 = (2*tp_f)/(2*tp_f+fp_f+fn_f) if (2*tp_f+fp_f+fn_f)>0 else 0.0
    auth_fp = fp_auth/tot_auth if tot_auth>0 else 0.0
    pix_acc = (tp+tn)/total_pix if total_pix>0 else 0.0

    print(f"Validation: SegF1(micro)={seg_f1:.4f}, P={seg_p:.4f}, R={seg_r:.4f}, IoU={seg_iou:.4f} | "
          f"FakeF1={fake_f1:.4f}, AuthFP={auth_fp:.6f}, PixelAcc={pix_acc:.4f}")

# 用法：
# eval_from_hard_val_lists(
#   "./debug_epoch0_filelists/val_hard_fake_epoch0.txt",
#   "./debug_epoch0_filelists/val_hard_auth_epoch0.txt",
#   "mask_results/result/"
# )
if __name__ == "__main__":
    eval_from_hard_val_lists(
        "./total=0+6206(C1_C2_Colu)+1000(Au_C1.0_C2.0)_epochs=100_batch=6_with_MoE=True(MoE_attn=CBAM_K=8)_with_HiLo=True/debug_epoch0_filelists/val_hard_fake_epoch0.txt",
        "./total=0+6206(C1_C2_Colu)+1000(Au_C1.0_C2.0)_epochs=100_batch=6_with_MoE=True(MoE_attn=CBAM_K=8)_with_HiLo=True/debug_epoch0_filelists/val_hard_auth_epoch0.txt",
        "mask_results/result/"
    )