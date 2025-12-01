import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from utils.config import get_pscc_args
from utils.load_tdata import TrainData, ValData
from utils.utils import findLastCheckpoint, save_image, adjust_learning_rate
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead

# TODO: MoE 不适用于多卡训练，会有 Bug，这里默认单卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_MoE = True
is_HiLo = True
MoE_K = 8
MoE_attn = 'CBAM'
MASK_F1_THRESH = 0.50

HARD_VIS_ROOT = "./hard_val_vis"   # 输出根目录
HARD_GOOD_T = 0.60
HARD_NORMAL_T = 0.30
SAVE_HARD_VAL_EVERY_EPOCH = True   # 你要求每个 epoch 都保存

from pathlib import Path

_hard_train_seen = set()  # 全局：防止重复保存

# ============ Hard Train Mining ============
HARD_TRAIN_ROOT = "./hard_train_mining"
HARD_TRAIN_F1_T = 0.60
ENABLE_HARD_TRAIN_MINING = True

import re
from pathlib import Path
from PIL import Image
import numpy as np

_hard_train_seen = set()

def build_pos_weight_mask(mask, pos_w=5.0, max_w=10.0):
    # mask: 0/1
    w = torch.ones_like(mask)
    pos = (mask == 1)
    if pos.any():
        w[pos] = min(float(pos_w), float(max_w))
    return w

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred_prob, target):
        # pred_prob/target: [B,1,H,W] 或 [B,H,W]
        if pred_prob.ndim == 4 and pred_prob.size(1) == 1:
            pred_prob = pred_prob[:, 0]
        if target.ndim == 4 and target.size(1) == 1:
            target = target[:, 0]

        pred = pred_prob.float().reshape(pred_prob.size(0), -1)
        gt   = target.float().reshape(target.size(0), -1)

        inter = (pred * gt).sum(dim=1)
        denom = pred.sum(dim=1) + gt.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth + self.eps)
        return 1.0 - dice.mean()

def sanitize_filename(name: str) -> str:
    """
    将路径或名字转为合法的文件名：
    - 去掉盘符、斜杠，换成下划线
    - 去掉非法字符
    """
    # 替换 / \ | 空格等为下划线
    s = re.sub(r"[\\/| :]", "_", name)
    # 去掉多余连续下划线
    s = re.sub(r"_+", "_", s)
    # 去掉开头结尾下划线
    s = s.strip("_")
    return s

def save_hard_train_sample(name, image, mask, cls, f1):
    """
    name: 样本名（字符串，建议 dataset 返回原始路径）
    image: [3,H,W] tensor, 0~1
    mask:  [H,W] tensor, 0/1
    cls:   scalar tensor
    f1:    float
    """
    global _hard_train_seen

    # 如果重复，直接返回
    if name in _hard_train_seen:
        return
    _hard_train_seen.add(name)

    # ---------------------- 目录 ----------------------
    root = Path(HARD_TRAIN_ROOT)
    img_dir = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- 文件名 ----------------------
    safe_name = sanitize_filename(name)

    # ---------------------- 保存 image ----------------------
    img_np = image.detach().cpu().clamp(0, 1).numpy()
    img_u8 = (img_np.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(img_u8).save(img_dir / f"{safe_name}.png")

    # ---------------------- 保存 mask ----------------------
    mask_np = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask_np).save(mask_dir / f"{safe_name}.png")

    # ---------------------- 记录 meta ----------------------
    with open(root / "meta.txt", "a") as f:
        f.write(f"{safe_name} cls={int(cls)} f1={f1:.4f}\n")


from PIL import Image
from pathlib import Path

def _f1_binary_np(pred01: np.ndarray, gt01: np.ndarray, zero_division: float = 1.0) -> float:
    """pred01/gt01: 0/1 的 numpy 数组"""
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return float(zero_division)
    return float(2 * tp / denom)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _save_u8_png(arr_u8: np.ndarray, path: Path, mode: str):
    _ensure_dir(path.parent)
    Image.fromarray(arr_u8, mode=mode).save(str(path), format="PNG", compress_level=1)

def _save_hard_val_sample(epoch: int, bucket: str, base_name: str,
                          img_u8: np.ndarray, gt_u8: np.ndarray,
                          pred_bin_u8: np.ndarray, pred_prob_u8: np.ndarray,
                          f1: float):
    """
    保存为:
      hard_val_vis/epoch_XXX/{good|normal|bad}/<base>__f1=0.1234_{img|gt|pred_bin|pred_prob}.png
    """
    out_dir = Path(HARD_VIS_ROOT) / f"epoch_{epoch:03d}" / bucket
    tag = f"{base_name}__f1={f1:.4f}"
    _save_u8_png(img_u8,       out_dir / f"{tag}_img.png",       mode="RGB")
    _save_u8_png(gt_u8,        out_dir / f"{tag}_gt.png",        mode="L")
    _save_u8_png(pred_bin_u8,  out_dir / f"{tag}_pred_bin.png",  mode="L")
    _save_u8_png(pred_prob_u8, out_dir / f"{tag}_pred_prob.png", mode="L")

def train(args):
    filename = (
        f'total={args["train_num"]}+180(Colu)+183(Au_Colu)_epochs={args["num_epochs"]}'
        f'_batch={args["train_bs"]}'
        f'_with_MoE={is_MoE}(MoE_attn={MoE_attn}_K={MoE_K})_with_HiLo={is_HiLo}.log'
    )

    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ================== 构建网络 ==================
    FENet_name = "HRNet"
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg).to(device)

    SegNet_name = "NLCDetection"
    SegNet = NLCDetection(args).to(device)

    ClsNet_name = "DetectionHead"
    ClsNet = DetectionHead(args).to(device)

    # ================== DataLoader ==================
    # args["hard_path"] = args.get("hard_path", "/mnt/c/datasets/datasets_hard")
    # args["hard_path"] = args.get("hard_path", "/mnt/c/datasets/datasets_hard_hard")
    # args["hard_train_pick"] = int(args.get("hard_train_pick", 1737))
    # args["hard_val_pick"] = int(args.get("hard_val_pick", 1737))
    # args["hard_cls_fixed"] = int(args.get("hard_cls_fixed", 1))  # hard 当作 fake(=1)

    # ================== DataLoader（只创建一次） ==================
    train_dataset = TrainData(args)
    val_dataset = ValData(args)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args["train_bs"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True if 32 > 0 else False,
    )

    # ================== 优化器 ==================
    params = list(FENet.parameters()) + list(SegNet.parameters()) + list(
        ClsNet.parameters()
    )
    optimizer = torch.optim.Adam(params, lr=args["learning_rate"])

    # ================== checkpoint 目录 ==================
    FENet_dir = "./checkpoint/{}_checkpoint".format(FENet_name)
    SegNet_dir = "./checkpoint/{}_checkpoint".format(SegNet_name)
    ClsNet_dir = "./checkpoint/{}_checkpoint".format(ClsNet_name)

    os.makedirs(FENet_dir, exist_ok=True)
    os.makedirs(SegNet_dir, exist_ok=True)
    os.makedirs(ClsNet_dir, exist_ok=True)

    # ================== 预训练权重（如果存在） ==================
    try:
        FENet_weight_path = "{}/{}.pth".format(FENet_dir, FENet_name)
        FENet_state_dict = torch.load(FENet_weight_path, map_location=device)
        FENet.load_state_dict(FENet_state_dict)
        logging.info(
            "{} weight-loading succeed: {}".format(FENet_name, FENet_weight_path)
        )
    except Exception as e:
        logging.info("{} weight-loading fails: {}".format(FENet_name, e))

    try:
        SegNet_weight_path = "{}/{}.pth".format(SegNet_dir, SegNet_name)
        SegNet_state_dict = torch.load(SegNet_weight_path, map_location=device)
        SegNet.load_state_dict(SegNet_state_dict)
        logging.info(
            "{} weight-loading succeed: {}".format(SegNet_name, SegNet_weight_path)
        )
    except Exception as e:
        logging.info("{} weight-loading fails: {}".format(SegNet_name, e))

    try:
        ClsNet_weight_path = "{}/{}.pth".format(ClsNet_dir, ClsNet_name)
        ClsNet_state_dict = torch.load(ClsNet_weight_path, map_location=device)
        ClsNet.load_state_dict(ClsNet_state_dict)
        logging.info(
            "{} weight-loading succeed: {}".format(ClsNet_name, ClsNet_weight_path)
        )
    except Exception as e:
        logging.info("{} weight-loading fails: {}".format(ClsNet_name, e))

    logging.info("length of traindata: {}".format(len(train_data_loader)))

    # ================== 损失函数 ==================
    authentic_ratio = args["train_ratio"][0]
    fake_ratio = 1 - authentic_ratio
    logging.info(
        "authentic_ratio: {}  fake_ratio: {}".format(authentic_ratio, fake_ratio)
    )

    weights = [1.0 / authentic_ratio, 1.0 / fake_ratio]
    weights = torch.tensor(weights).to(device)
    CE_loss = nn.CrossEntropyLoss(weight=weights).to(device)

    BCE_loss_full = nn.BCELoss(reduction='none').to(device)
    dice = DiceLoss().to(device)

    # ================== 断点续训 ==================
    # 统一使用 SegNet_dir / NLCDetection_{epoch}.pth 保存 / 恢复完整 checkpoint
    initial_epoch = findLastCheckpoint(save_dir=SegNet_dir)
    resume_ok = False

    if initial_epoch > 0:
        ckpt_path = "{0}/{1}_{2}.pth".format(SegNet_dir, SegNet_name, initial_epoch)
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            FENet.load_state_dict(checkpoint["FENet"])
            SegNet.load_state_dict(checkpoint["SegNet"])
            ClsNet.load_state_dict(checkpoint["ClsNet"])

            # ✅重置 optimizer：重新 new 一个
            initial_epoch = 0
            params = list(FENet.parameters()) + list(SegNet.parameters()) + list(ClsNet.parameters())
            optimizer = torch.optim.Adam(params, lr=args["learning_rate"])

            logging.info(f"Loaded weights from {ckpt_path}, optimizer RESET (fresh Adam).")
            resume_ok = True
            logging.info("==> Resume training from checkpoint: %s (epoch=%d)", ckpt_path, initial_epoch)
        except Exception as e:
            logging.info("Cannot load checkpoint %s: %s", ckpt_path, e)
            initial_epoch = 0
            logging.info("==> Restart training from epoch 0")


    if not resume_ok:
        logging.info("==> Start training from scratch (may load single-net weights if exist).")

    # ==================（可选）只在非断点续训时加载单独权重 ==================
    # 建议你把上面的三个 try-load（HRNet.pth / NLCDetection.pth / DetectionHead.pth）
    # 放到这里，并且仅当 resume_ok=False 时执行，避免“先加载旧权重又被checkpoint覆盖”的混乱。

    # ================== baseline validation（此时已经是正确状态） ==================
    seg_f1, cls_acc = validation(FENet, SegNet, ClsNet, args, -1, val_dataset)
    logging.info("seg_f1 before training %.4f, cls_acc before training %.4f", seg_f1, cls_acc)

    # ================== 训练 loop ==================
    f1_sum = 0.0
    batch_count = 0
    epoch_list = []
    loss_list = []
    diversity_history = []
    routing_history = []

    previous_seg_f1 = -1e9
    best_epoch = 0

    for epoch in range(initial_epoch, args["num_epochs"]):
        adjust_learning_rate(
            optimizer, epoch, args["lr_strategy"], args["lr_decay_step"]
        )

        seg_loss_sum = 0.0
        cls_loss_sum = 0.0
        aux_loss_sum = 0.0
        cls_total, cls_correct = 0, 0

        epoch_loss_sum = 0.0
        epoch_step_count = 0

        FENet.train()
        SegNet.train()
        ClsNet.train()

        train_dataset.set_epoch(epoch)
        # val_dataset.set_epoch(epoch)

        if epoch == 0:
            debug_dir = "./debug_epoch0_filelists"
            train_dataset.dump_epoch0_lists(debug_dir, epoch=0, print_n=30)
            val_dataset.dump_epoch0_lists(debug_dir, epoch=0, print_n=30)

        for batch_id, train_data in enumerate(train_data_loader):
            # image, [mask1, mask2, mask3, mask4], cls
            # image, masks, cls = train_data
            image, masks, cls, names = train_data
            # 只要不是 origin，全归为 fake
            cls[cls != 0] = 1
            mask1, mask2, mask3, mask4 = masks

            # ---------- to(device) ----------
            image = image.to(device)
            mask1 = mask1.float().to(device)
            mask2 = mask2.float().to(device)
            mask3 = mask3.float().to(device)
            mask4 = mask4.float().to(device)

            cls = cls.to(device)

            optimizer.zero_grad()

            # ---------- Feature extraction ----------
            feat = FENet(image)

            # ---------- Segmentation (MoE / HiLo) ----------
            # 温度退火策略
            current_temp = max(0.5, 2.0 * (1 - epoch / 100))
            # 这里已经不再使用 DataParallel，所以直接访问 SegNet.getmaskX
            SegNet.getmask4.temp = current_temp
            SegNet.getmask3.temp = current_temp
            SegNet.getmask2.temp = current_temp
            SegNet.getmask1.temp = current_temp

            pred_masks, feats, aux_losses = SegNet(feat)
            # pred_masks, feats = SegNet(feat)
            pred_mask1, pred_mask2, pred_mask3, pred_mask4 = pred_masks
            feat1, feat2, feat3, feat4 = feats

            if is_MoE:
                div4 = SegNet.getmask4.get_diversity()
                div3 = SegNet.getmask3.get_diversity()
                div2 = SegNet.getmask2.get_diversity()
                div1 = SegNet.getmask1.get_diversity()
                diversity_history.append([div4, div3, div2, div1])

                rout4 = SegNet.getmask4.get_routing_stats()
                rout3 = SegNet.getmask3.get_routing_stats()
                rout2 = SegNet.getmask2.get_routing_stats()
                rout1 = SegNet.getmask1.get_routing_stats()
                routing_history.append([rout4, rout3, rout2, rout1])

                if batch_id % 100 == 99:
                    moe_log = []
                    for k, div, rout, temp in zip(
                        ["getmask4", "getmask3", "getmask2", "getmask1"],
                        [div4, div3, div2, div1],
                        [rout4, rout3, rout2, rout1],
                        [
                            SegNet.getmask4.temp,
                            SegNet.getmask3.temp,
                            SegNet.getmask2.temp,
                            SegNet.getmask1.temp,
                        ],
                    ):
                        moe_log.append(
                            f"{k}:temp={temp:.4f}, div={div:.4f}, route={np.round(rout, 3)}"
                        )
                    moe_log_str = " | ".join(moe_log)
                    logging.info(
                        "[Epoch {0}, Batch {1}] MoE: {2}".format(
                            epoch + 1, batch_id + 1, moe_log_str
                        )
                    )

            # ---------- Classification ----------
            pred_logit = ClsNet(feats)

            # ---------- Segmentation loss ----------
            # pred_maskX: [B,1,H,W] -> [B,H,W]
            pred_mask1 = pred_mask1.squeeze(1)
            pred_mask2 = pred_mask2.squeeze(1)
            pred_mask3 = pred_mask3.squeeze(1)
            pred_mask4 = pred_mask4.squeeze(1)

            # 你当前 pred_maskX 基本是 prob(0~1)，为 BCELoss 做数值安全
            eps = 1e-6
            pred_mask1 = pred_mask1.clamp(eps, 1.0 - eps)
            pred_mask2 = pred_mask2.clamp(eps, 1.0 - eps)
            pred_mask3 = pred_mask3.clamp(eps, 1.0 - eps)
            pred_mask4 = pred_mask4.clamp(eps, 1.0 - eps)

            # GT -> 0/1 float
            mask1_01 = (mask1 > 0.5).float()
            mask2_01 = (mask2 > 0.5).float()
            mask3_01 = (mask3 > 0.5).float()
            mask4_01 = (mask4 > 0.5).float()

            # ========= 关键：只对正类加权（抑制 FP，比你原来的 balance 更稳） =========
            pos_w  = float(args.get("pos_w", 5.0))     # 建议 3~8 起试
            w_clip = float(args.get("w_clip", 10.0))  # 防止极端 mask 权重爆炸

            def make_pos_weight(gt01):
                w = torch.ones_like(gt01)
                w = w + gt01 * (pos_w - 1.0)          # 正类=pos_w，负类=1
                return w.clamp_(1.0, w_clip)

            w1 = make_pos_weight(mask1_01)
            w2 = make_pos_weight(mask2_01)
            w3 = make_pos_weight(mask3_01)
            w4 = make_pos_weight(mask4_01)

            # BCE（逐像素 weight）
            mask1_bce = (BCE_loss_full(pred_mask1, mask1_01) * w1).mean()
            mask2_bce = (BCE_loss_full(pred_mask2, mask2_01) * w2).mean()
            mask3_bce = (BCE_loss_full(pred_mask3, mask3_01) * w3).mean()
            mask4_bce = (BCE_loss_full(pred_mask4, mask4_01) * w4).mean()
            bce_loss = mask1_bce + mask2_bce + mask3_bce + mask4_bce

            # Dice：只在 fake 图上算（auth 全黑用 fp_pen 控制）
            dice_w = float(args.get("dice_w", 0.25))  # 你现在 P 低，先用 0.2~0.3
            fake = (cls != 0)

            if fake.any():
                d1 = dice(pred_mask1[fake], mask1_01[fake])
                d2 = dice(pred_mask2[fake], mask2_01[fake])
                d3 = dice(pred_mask3[fake], mask3_01[fake])
                d4 = dice(pred_mask4[fake], mask4_01[fake])
                dice_loss = d1 + d2 + d3 + d4
            else:
                dice_loss = torch.zeros((), device=device, dtype=pred_mask1.dtype)

            seg_loss = bce_loss + dice_w * dice_loss

            # aux_loss（MoE 路由正则）
            aux_loss = sum(aux_losses)

            # ---------- Classification loss ----------
            cls_loss = CE_loss(pred_logit, cls)

            # ========= auth FP penalty：只对 cls==0 的图压“预测面积” =========
            fp_w = float(args.get("fp_pen_w", 0.02))      # 建议 0.01~0.05
            warm = int(args.get("fp_pen_warmup", 0))      # 可设 3~5 个 epoch 预热再开
            if epoch < warm:
                fp_w = 0.0

            auth = (cls == 0)
            if auth.any():
                fp_pen = (
                    pred_mask1[auth].mean()
                    + pred_mask2[auth].mean()
                    + pred_mask3[auth].mean()
                    + pred_mask4[auth].mean()
                ) / 4.0
            else:
                fp_pen = torch.zeros((), device=device, dtype=pred_mask1.dtype)

            loss = seg_loss + cls_loss + aux_loss + fp_w * fp_pen

            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_step_count += 1

            # ---------- 训练阶段指标统计 ----------
            # seg F1（只用 mask1 做例子）
            with torch.no_grad():
                prob_mask1 = pred_mask1.clamp(0.0, 1.0)
                binary_mask1 = (prob_mask1 > MASK_F1_THRESH)

                # ✅ 用二值 GT
                gt01 = (mask1 > 0.5)

                if ENABLE_HARD_TRAIN_MINING:
                    B = image.size(0)
                    for i in range(B):
                        pred_i = binary_mask1[i].cpu().numpy().astype(np.uint8).ravel()
                        gt_i   = gt01[i].cpu().numpy().astype(np.uint8).ravel()   # ✅改这里

                        f1_i = f1_score(gt_i, pred_i, zero_division=1)
                        if f1_i < HARD_TRAIN_F1_T:
                            save_hard_train_sample(
                                name=names[i],
                                image=image[i],
                                mask=mask1[i],   # 保存原 mask float 也行
                                cls=cls[i],
                                f1=f1_i,
                            )

                # ✅ batch f1 也用二值 GT
                binary_mask1_np = binary_mask1.cpu().numpy().astype(np.uint8).ravel()
                mask1_np        = gt01.cpu().numpy().astype(np.uint8).ravel()
                batch_f1 = f1_score(mask1_np, binary_mask1_np, zero_division=1)

                f1_sum += batch_f1
                batch_count += 1

                # 分类 accuracy
                _, binary_cls = torch.max(pred_logit, 1)
                cls_correct += (binary_cls == cls).sum().item()
                cls_total += cls.size(0)

                seg_loss_sum += seg_loss.item()
                cls_loss_sum += cls_loss.item()
                aux_loss_sum += aux_loss.item()

            if batch_id % 100 == 99:
                mean_f1 = f1_sum / batch_count
                logging.info(
                    "[Epoch {0}, Batch {1}] F1 Score: {2:.4f}, seg_loss: {3:.4f}; "
                    "Classification Accuracy: [{4}/{5}] {6:.2f}%, cls_loss: {7:.4f}; "
                    "aux_loss: {8:.4f}".format(
                        epoch + 1,
                        batch_id + 1,
                        mean_f1,
                        seg_loss_sum / 100,
                        cls_correct,
                        cls_total,
                        cls_correct / cls_total * 100.0,
                        cls_loss_sum / 100,
                        aux_loss_sum / 100
                    )
                )
                # reset
                f1_sum, batch_count = 0.0, 0
                seg_loss_sum, cls_loss_sum, aux_loss_sum = 0.0, 0.0, 0.0
                cls_correct, cls_total = 0, 0

        # ---------- 每个 epoch 结束后保存 checkpoint ----------
        checkpoint = {
            "epoch": epoch + 1,
            "FENet": FENet.state_dict(),
            "SegNet": SegNet.state_dict(),
            "ClsNet": ClsNet.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if (epoch + 1) % 5 == 0:
            ckpt_path = "{0}/{1}_{2}.pth".format(
                SegNet_dir, SegNet_name, epoch + 1
            )
            torch.save(checkpoint, ckpt_path)
            logging.info("Saved checkpoint at epoch {0} -> {1}".format(
                epoch + 1, ckpt_path
            ))

        # last
        torch.save(
            checkpoint,
            "{0}/{1}_last.pth".format(SegNet_dir, SegNet_name),
        )

        # ---------- 每个 epoch 做一次验证 ----------
        seg_f1, cls_acc = validation(FENet, SegNet, ClsNet, args, epoch + 1, val_dataset)
        logging.info(
            "Epoch %d: seg_f1=%.4f, cls_acc=%.4f",
            epoch + 1, seg_f1, cls_acc
        )

        if seg_f1 >= previous_seg_f1:
            previous_seg_f1 = seg_f1
            best_epoch = epoch + 1

            torch.save(
                checkpoint,
                "{0}/{1}_best.pth".format(SegNet_dir, SegNet_name),
            )
            logging.info(
                "*** New best model at epoch %d, seg_f1=%.4f, cls_acc=%.4f ***",
                best_epoch, seg_f1, cls_acc
            )

        # 计算当前 epoch 的平均 loss
        epoch_avg_loss = epoch_loss_sum / max(1, epoch_step_count)

        epoch_list.append(epoch + 1)
        loss_list.append(epoch_avg_loss)

        logging.info(
            "Epoch {0}: epoch_avg_loss={1:.4f}".format(
                epoch + 1, epoch_avg_loss
            )
        )

    logging.info(
        "*** best model at epoch %d, seg_f1=%.4f, cls_acc=%.4f ***",
        best_epoch, seg_f1, cls_acc
    )


def validation(FENet, SegNet, ClsNet, args, epoch: int, val_dataset):
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args["val_bs"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # ======= 全体像素统计（micro）=======
    tp = fp = fn = 0
    tn = 0
    # ======= 只看 fake 图（gt 有正像素）=======
    tp_f = fp_f = fn_f = 0
    # ======= 只看 auth 图（gt 全 0）上的误检比例 =======
    fp_auth = 0
    tot_auth = 0

    # all_mask_pred = []
    # all_mask_gt = []
    cls_correct, cls_total = 0, 0

    # hard 分桶统计
    hard_cnt = {"good": 0, "normal": 0, "bad": 0}
    hard_seen = 0

    hard_key = str(args.get("hard_path", "")).strip()
    do_save_hard = (epoch >= 0) and SAVE_HARD_VAL_EVERY_EPOCH and (hard_key != "")

    FENet.eval()
    SegNet.eval()
    ClsNet.eval()

    with torch.no_grad():
        for batch_id, val_data in enumerate(val_data_loader):
            image, mask, cls, name = val_data
            image = image.to(device, non_blocking=True)
            mask = mask.float().to(device, non_blocking=True)

            cls[cls != 0] = 1
            cls = cls.to(device, non_blocking=True)

            feat = FENet(image)
            pred_masks, feats, aux_loss = SegNet(feat)
            # pred_masks, feats = SegNet(feat)
            pred_mask1, pred_mask2, pred_mask3, pred_mask4 = pred_masks
            # print("pred_mask1 stats:", pred_mask1.min().item(), pred_mask1.max().item(), pred_mask1.mean().item())


            pred_logit = ClsNet(feats)

            # 统一 pred_mask1 尺寸到 gt mask
            # mask 可能是 [B,H,W]；pred_mask1 是 [B,1,h,w]
            gt_hw = mask.shape[-2:]
            if pred_mask1.shape[-2:] != gt_hw:
                pred_mask1 = F.interpolate(pred_mask1, size=gt_hw, mode="bilinear", align_corners=True)

            # ======= 统一形状 =======
            # pred_mask1: [B,1,H,W] -> [B,H,W]
            prob = pred_mask1
            if prob.ndim == 4 and prob.size(1) == 1:
                prob = prob[:, 0, :, :]

            # gt mask: [B,H,W]（你的 mask 已经是 float）
            gt = (mask > 0.5)
            pred = (prob > MASK_F1_THRESH)

            # ======= micro: 全体像素 TP/FP/FN/TN =======
            tp += (pred & gt).sum().item()
            fp += (pred & (~gt)).sum().item()
            fn += ((~pred) & gt).sum().item()
            tn += ((~pred) & (~gt)).sum().item()

            # ======= fake-only vs auth-only 诊断 =======
            B = gt.size(0)
            gt_pos = gt.flatten(1).sum(dim=1)  # 每张图 gt 正像素数

            for i in range(B):
                if gt_pos[i] > 0:
                    # fake：gt 有正像素
                    tp_f += (pred[i] & gt[i]).sum().item()
                    fp_f += (pred[i] & (~gt[i])).sum().item()
                    fn_f += ((~pred[i]) & gt[i]).sum().item()
                else:
                    # auth：gt 全 0，统计误检比例
                    fp_auth += pred[i].sum().item()
                    tot_auth += pred[i].numel()

            # 分类准确率
            pred_prob = torch.softmax(pred_logit, dim=1)
            _, binary_cls = torch.max(pred_prob, 1)
            cls_correct += (binary_cls == cls).sum().item()
            cls_total += cls.size(0)

            # ============ 仅对 hard-val 保存 good/normal/bad ============
            if do_save_hard:
                # name 可能是 list/tuple，也可能是 tensor/list
                if isinstance(name, (list, tuple)):
                    names = list(name)
                else:
                    names = [name]

                B = image.size(0)
                for i in range(B):
                    ni = str(names[i])
                    if hard_key not in ni:
                        continue  # 只处理 hard 集

                    hard_seen += 1

                    # 取单张 pred / gt
                    # pred_mask1: [B,1,H,W]
                    prob_i = pred_mask1[i]
                    if prob_i.ndim == 3:  # [1,H,W]
                        prob_i = prob_i[0]
                    prob_np = prob_i.detach().float().cpu().clamp(0, 1).numpy()  # [H,W]

                    pred01 = (prob_np > MASK_F1_THRESH).astype(np.uint8)
                    # gt: mask[i] 可能 [H,W] 或 [1,H,W]
                    gt_i = mask[i]
                    if gt_i.ndim == 3:
                        gt_i = gt_i[0]
                    gt_np = (gt_i.detach().float().cpu().numpy() > 0.5).astype(np.uint8)

                    f1_i = _f1_binary_np(pred01, gt_np, zero_division=1.0)

                    if f1_i >= HARD_GOOD_T:
                        bucket = "good"
                    elif f1_i >= HARD_NORMAL_T:
                        bucket = "normal"
                    else:
                        bucket = "bad"
                    hard_cnt[bucket] += 1

                    # 保存图像/GT/Pred
                    img_i = image[i].detach().float().cpu().clamp(0, 1).numpy()      # [3,H,W]
                    img_u8 = (img_i.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)

                    gt_u8 = (gt_np * 255).astype(np.uint8)
                    pred_bin_u8 = (pred01 * 255).astype(np.uint8)
                    pred_prob_u8 = (prob_np * 255.0 + 0.5).astype(np.uint8)

                    base = os.path.splitext(os.path.basename(ni))[0]
                    _save_hard_val_sample(epoch=epoch, bucket=bucket, base_name=base,
                                          img_u8=img_u8, gt_u8=gt_u8,
                                          pred_bin_u8=pred_bin_u8, pred_prob_u8=pred_prob_u8,
                                          f1=f1_i)

    # ======= 由累计 TP/FP/FN 计算 micro 指标 =======
    seg_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    seg_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    seg_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    seg_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # fake-only F1（更反映“定位能力”）
    seg_f1_fake = (2 * tp_f) / (2 * tp_f + fp_f + fn_f) if (2 * tp_f + fp_f + fn_f) > 0 else 0.0

    # auth 上误检率（越低越好）
    auth_fp_rate = fp_auth / tot_auth if tot_auth > 0 else 0.0

    # 像素准确率（包含 TN，更“全体像素为分母”）
    pixel_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0

    print(
        f"Validation: SegF1(micro)={seg_f1:.4f}, P={seg_p:.4f}, R={seg_r:.4f}, IoU={seg_iou:.4f} | "
        f"FakeF1={seg_f1_fake:.4f}, AuthFP={auth_fp_rate:.6f}, PixelAcc={pixel_acc:.4f} | "
        f"ClsAcc={cls_acc:.4f}"
    )
    return seg_f1, cls_acc



if __name__ == "__main__":
    args = get_pscc_args()
    train(args)
