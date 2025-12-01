import os
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils.utils import save_image
from utils.config import get_pscc_args
from utils.load_vdata import TestData

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ===================== checkpoint =====================
def load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet):
    ckpt = torch.load(ckpt_path, map_location=device)
    print("Loaded checkpoint:", ckpt_path)

    FENet.load_state_dict(ckpt["FENet"])
    SegNet.load_state_dict(ckpt["SegNet"])
    ClsNet.load_state_dict(ckpt["ClsNet"], strict=False)


# ===================== test =====================
def test(args):
    # ---------- 输出目录 ----------
    result_root = "./result"
    if os.path.exists(result_root):
        shutil.rmtree(result_root)
    os.makedirs(result_root, exist_ok=True)

    # ---------- 网络 ----------
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg).to(device)
    SegNet = NLCDetection(args).to(device)
    ClsNet = DetectionHead(args).to(device)

    # ---------- checkpoint ----------
    ckpt_path = (
        "./total=0+6206(C1_C2_Colu)+1000(Au_C1.0_C2.0)_epochs=100_batch=6_with_MoE=True(MoE_attn=CBAM_K=8)_with_HiLo=True/checkpoint/NLCDetection_checkpoint/NLCDetection_25.pth"
    )
    load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet)

    FENet.eval()
    SegNet.eval()
    ClsNet.eval()

    # ---------- DataLoader ----------
    test_loader = DataLoader(
        TestData(args),
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
    )

    # ---------- 指标缓存 ----------
    y_true = []
    y_score = []
    y_pred = []

    # ---------- 推理 ----------
    with torch.inference_mode():
        for test_data in tqdm(
            test_loader,
            desc="Testing",
            ncols=100,
        ):
            images, _, names = test_data
            images = images.to(device)
            B = images.size(0)

            # ===== forward (FP32，稳定优先) =====
            feat = FENet(images)
            pred_masks, feats, _ = SegNet(feat)
            # pred_masks, feats = SegNet(feat)
            pred_logits = ClsNet(feats)
            pred_probs = torch.softmax(pred_logits, dim=1)  # [B,2]

            forged_probs = pred_probs[:, 1]  # [B]

            # ===== 逐样本处理 =====
            for i in range(B):
                name_i = names[i]
                prob_i = forged_probs[i].item()

                # ---- NaN / Inf 防御 ----
                if not torch.isfinite(torch.tensor(prob_i)):
                    print(f"[WARN] NaN/Inf score detected, skip: {name_i}")
                    continue

                pred_cls = int(prob_i >= 0.5)

                # ---- GT ----
                if "authentic" in name_i.lower() or "au" in name_i.lower():
                    gt = 0
                    # print(name_i)
                else:
                    gt = 1

                y_true.append(gt)
                y_score.append(prob_i)
                y_pred.append(pred_cls)

                # ---- 可选：保存 mask ----
                pred_mask = pred_masks[0][i:i + 1]  # [1,1,h,w]
                pred_mask_up = F.interpolate(
                    pred_mask.float(),
                    size=(images.size(2), images.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )

                save_image(
                    torch.clamp(pred_mask_up, 0, 1),
                    os.path.join(
                        result_root,
                        os.path.splitext(os.path.basename(name_i))[0],
                    ),
                    "mask",
                )

    # ===================== 计算指标 =====================
    y_true = torch.tensor(y_true)
    y_score = torch.tensor(y_score)
    y_pred = torch.tensor(y_pred)

    assert y_true.numel() > 0, "No valid samples!"
    assert len(torch.unique(y_true)) == 2, "Test set must contain both classes!"

    auc = roc_auc_score(y_true.numpy(), y_score.numpy())
    acc = accuracy_score(y_true.numpy(), y_pred.numpy())
    p = precision_score(y_true.numpy(), y_pred.numpy())
    r = recall_score(y_true.numpy(), y_pred.numpy())
    f1 = f1_score(y_true.numpy(), y_pred.numpy())

    print("\n========== Image-level Evaluation ==========")
    print(f"AUC : {auc:.4f}")
    print(f"ACC : {acc:.4f}")
    print(f"P   : {p:.4f}")
    print(f"R   : {r:.4f}")
    print(f"F1  : {f1:.4f}")
    print("===========================================\n")


if __name__ == "__main__":
    args = get_pscc_args()
    test(args)
