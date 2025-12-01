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
        f'total={args["train_num"]}+6206(C1_C2_Colu)+1000(Au_C1.0_C2.0)_epochs={args["num_epochs"]}'
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

    # 训练前 baseline（只做记录，不参与 best_score 逻辑）
    seg_f1, cls_acc = validation(FENet, SegNet, ClsNet, args, -1, val_dataset)
    logging.info("seg_f1 before training %.4f, cls_acc before training %.4f", seg_f1, cls_acc)

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

    # ================== 断点续训 ==================
    # 统一使用 SegNet_dir / NLCDetection_{epoch}.pth 保存 / 恢复完整 checkpoint
    initial_epoch = findLastCheckpoint(save_dir=SegNet_dir)
    if initial_epoch > 0:
        ckpt_path = "{0}/{1}_{2}.pth".format(SegNet_dir, SegNet_name, initial_epoch)
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            FENet.load_state_dict(checkpoint["FENet"])
            SegNet.load_state_dict(checkpoint["SegNet"])
            ClsNet.load_state_dict(checkpoint["ClsNet"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "Resuming all nets & optimizer by loading epoch {}".format(
                    initial_epoch
                )
            )
        except Exception as e:
            logging.info(
                "Cannot load checkpoint at epoch {}: {}".format(initial_epoch, e)
            )
            initial_epoch = 0
            logging.info("Restart training from epoch 0")

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

            # ---------- 构造 balance mask ----------
            def build_balance_mask(mask):
                balance = torch.ones_like(mask)
                pos = mask == 1
                num_pos = pos.sum().float()
                num_all = mask.numel()
                if num_pos > 0:
                    neg = mask == 0
                    balance[pos] = 0.5 / (num_pos / num_all)
                    balance[neg] = 0.5 / (neg.sum().float() / num_all)
                    return balance
                else:
                    # logging.info("Mask balance is not working! (no positive)")
                    return balance

            mask1_balance = build_balance_mask(mask1)
            mask2_balance = build_balance_mask(mask2)
            mask3_balance = build_balance_mask(mask3)
            mask4_balance = build_balance_mask(mask4)

            # ---------- to(device) ----------
            image = image.to(device)
            mask1 = mask1.float().to(device)
            mask2 = mask2.float().to(device)
            mask3 = mask3.float().to(device)
            mask4 = mask4.float().to(device)

            mask1_balance = mask1_balance.to(device)
            mask2_balance = mask2_balance.to(device)
            mask3_balance = mask3_balance.to(device)
            mask4_balance = mask4_balance.to(device)

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
            # 这里 pred_maskX 仍然是 logits，不再 clamp 到 [0,1]
            pred_mask1 = pred_mask1.squeeze(dim=1)
            pred_mask2 = pred_mask2.squeeze(dim=1)
            pred_mask3 = pred_mask3.squeeze(dim=1)
            pred_mask4 = pred_mask4.squeeze(dim=1)

            # 数值安全：保证落在 (0,1) 内，防止 BCELoss 里 log(0)
            eps = 1e-4
            pred_mask1 = pred_mask1.clamp(eps, 1.0 - eps)
            pred_mask2 = pred_mask2.clamp(eps, 1.0 - eps)
            pred_mask3 = pred_mask3.clamp(eps, 1.0 - eps)
            pred_mask4 = pred_mask4.clamp(eps, 1.0 - eps)

            mask1_loss = torch.mean(
                BCE_loss_full(pred_mask1, mask1) * mask1_balance
            )
            mask2_loss = torch.mean(
                BCE_loss_full(pred_mask2, mask2) * mask2_balance
            )
            mask3_loss = torch.mean(
                BCE_loss_full(pred_mask3, mask3) * mask3_balance
            )
            mask4_loss = torch.mean(
                BCE_loss_full(pred_mask4, mask4) * mask4_balance
            )
            seg_loss = mask1_loss + mask2_loss + mask3_loss + mask4_loss

            aux_loss = sum(aux_losses)

            # ---------- Classification loss ----------
            cls_loss = CE_loss(pred_logit, cls)
            loss = seg_loss + cls_loss + aux_loss
            # loss = seg_loss + cls_loss
            # print(seg_loss.item(), cls_loss.item(), aux_loss.item())

            loss.backward()
            optimizer.step()

            # 累计 epoch loss
            epoch_loss_sum += loss.item()
            epoch_step_count += 1

            # ---------- 训练阶段指标统计 ----------
            # seg F1（只用 mask1 做例子）
            with torch.no_grad():
                # pred_mask1 已经是概率（0~1），来自 MoE+CBAM
                prob_mask1 = torch.clamp(pred_mask1, 0.0, 1.0)
                binary_mask1 = prob_mask1 > MASK_F1_THRESH 

                if ENABLE_HARD_TRAIN_MINING:
                    B = image.size(0)
                    for i in range(B):
                        pred_i = binary_mask1[i].cpu().numpy().flatten()
                        gt_i = mask1[i].cpu().numpy().flatten()

                        f1_i = f1_score(gt_i, pred_i, zero_division=1)

                    if f1_i < HARD_TRAIN_F1_T:
                        sample_name = names[i]  # ✅ 直接从 Dataset 返回

                        save_hard_train_sample(
                            name=sample_name,
                            image=image[i],
                            mask=mask1[i],
                            cls=cls[i],
                            f1=f1_i,
                        )

                binary_mask1_np = binary_mask1.cpu().numpy().flatten()
                mask1_np = mask1.cpu().numpy().flatten()

                batch_f1 = f1_score(
                    mask1_np, binary_mask1_np, zero_division=1
                )
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

    all_mask_pred = []
    all_mask_gt = []
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

            pred_logit = ClsNet(feats)

            # 统一 pred_mask1 尺寸到 gt mask
            # mask 可能是 [B,H,W]；pred_mask1 是 [B,1,h,w]
            gt_hw = mask.shape[-2:]
            if pred_mask1.shape[-2:] != gt_hw:
                pred_mask1 = F.interpolate(pred_mask1, size=gt_hw, mode="bilinear", align_corners=True)

            # 预测二值 / GT 二值（用于整体 F1）
            pred_bin = (pred_mask1 > MASK_F1_THRESH).cpu().numpy().astype(np.uint8)  # [B,1,H,W] or [B,H,W]
            gt_bin = (mask > 0.5).cpu().numpy().astype(np.uint8)                     # [B,H,W] or [B,1,H,W]

            all_mask_pred.append(pred_bin.reshape(pred_bin.shape[0], -1))
            all_mask_gt.append(gt_bin.reshape(gt_bin.shape[0], -1))

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

    all_mask_pred = np.concatenate(all_mask_pred, axis=0).reshape(-1)
    all_mask_gt = np.concatenate(all_mask_gt, axis=0).reshape(-1)

    seg_f1 = f1_score(all_mask_gt, all_mask_pred, zero_division=1)
    cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0

    # 打印 / 记录 hard 分桶统计
    if do_save_hard:
        logging.info(
            "[HardVal][Epoch %d] saved=%d | good=%d normal=%d bad=%d | root=%s",
            epoch, hard_seen, hard_cnt["good"], hard_cnt["normal"], hard_cnt["bad"], HARD_VIS_ROOT
        )
        print(f"[HardVal][Epoch {epoch}] saved={hard_seen} good={hard_cnt['good']} normal={hard_cnt['normal']} bad={hard_cnt['bad']} -> {HARD_VIS_ROOT}/epoch_{epoch:03d}/")

    print(f"Validation: Seg F1={seg_f1:.4f}, Cls Acc={cls_acc:.4f}")
    return seg_f1, cls_acc


if __name__ == "__main__":
    args = get_pscc_args()
    train(args)
