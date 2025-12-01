import os
import logging
import numpy as np
import matplotlib.pyplot as plt
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
MASK_F1_THRESH = 0.5

def train(args):
    filename = (
        f'total={args["train_num"]}_epochs={args["num_epochs"]}'
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
    train_data_loader = DataLoader(
        TrainData(args),
        batch_size=args["train_bs"],
        shuffle=True,
        num_workers=8,
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
    seg_f1, cls_acc = validation(FENet, SegNet, ClsNet, args, -1)
    logging.info("seg_f1 before training {0:.4f}, cls_acc before training {0:.4f}".format(seg_f1, cls_acc))

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

    previous_seg_f1, previous_cls_acc = -1e9, -1e9
    best_epoch = 0

    for epoch in range(initial_epoch, args["num_epochs"]):
        adjust_learning_rate(
            optimizer, epoch, args["lr_strategy"], args["lr_decay_step"]
        )

        seg_loss_sum = 0.0
        cls_loss_sum = 0.0
        cls_total, cls_correct = 0, 0

        epoch_loss_sum = 0.0
        epoch_step_count = 0

        FENet.train()
        SegNet.train()
        ClsNet.train()

        for batch_id, train_data in enumerate(train_data_loader):
            # image, [mask1, mask2, mask3, mask4], cls
            image, masks, cls = train_data
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
                else:
                    logging.info("Mask balance is not working! (no positive)")
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

            pred_masks, feats = SegNet(feat)
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

                if batch_id % 10 == 9:
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

            # ---------- Classification loss ----------
            cls_loss = CE_loss(pred_logit, cls)
            loss = seg_loss + cls_loss

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

            if batch_id % 10 == 9:
                mean_f1 = f1_sum / batch_count
                logging.info(
                    "[Epoch {0}, Batch {1}] F1 Score: {2:.4f}, seg_loss: {3:.4f}; "
                    "Classification Accuracy: [{4}/{5}] {6:.2f}%, cls_loss: {7:.4f}".format(
                        epoch + 1,
                        batch_id + 1,
                        mean_f1,
                        seg_loss_sum / 10,
                        cls_correct,
                        cls_total,
                        cls_correct / cls_total * 100.0,
                        cls_loss_sum / 10,
                    )
                )
                # reset
                f1_sum, batch_count = 0.0, 0
                seg_loss_sum, cls_loss_sum = 0.0, 0.0
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
        seg_f1, cls_acc = validation(FENet, SegNet, ClsNet, args, epoch + 1)
        logging.info(
            "Epoch %d: seg_f1=%.4f, cls_acc=%.4f",
            epoch + 1, seg_f1, cls_acc
        )

        if seg_f1 >= previous_seg_f1 and cls_acc >= previous_cls_acc:
            previous_seg_f1 = seg_f1
            previous_cls_acc = cls_acc
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

    # ====== MoE 统计的可视化（按需开启） ======
    # div_arr = np.array(diversity_history)  # shape: [steps, 4]
    # for i, name in enumerate(["getmask4", "getmask3", "getmask2", "getmask1"]):
    #     plt.plot(div_arr[:, i], label=name)
    # plt.legend()
    # plt.title("Expert Diversity (Variance) at Different Scales")
    # plt.xlabel("Iteration")
    # plt.ylabel("Variance")
    # plt.grid()
    # plt.show()
    #
    # routing_arr = np.array(routing_history)  # shape: [steps, 4, num_experts]
    # for i, name in enumerate(["getmask4", "getmask3", "getmask2", "getmask1"]):
    #     plt.bar(
    #         np.arange(routing_arr.shape[2]),
    #         routing_arr[-1, i, :],
    #         alpha=0.7,
    #         label=name,
    #     )
    # plt.legend()
    # plt.title("Routing Distribution (last step)")
    # plt.xlabel("Expert ID")
    # plt.ylabel("Mean Routing")
    # plt.show()


def validation(FENet, SegNet, ClsNet, args, epoch: int):
    val_data_loader = DataLoader(
        ValData(args),
        batch_size=args["val_bs"],
        shuffle=False,
        num_workers=8,
    )

    all_mask_pred = []
    all_mask_gt = []
    cls_correct, cls_total = 0, 0

    FENet.eval()
    SegNet.eval()
    ClsNet.eval()

    with torch.no_grad():
        for batch_id, val_data in enumerate(val_data_loader):
            image, mask, cls, name = val_data
            image = image.to(device)
            mask = mask.float().to(device)
            cls[cls != 0] = 1
            cls = cls.to(device)

            feat = FENet(image)
            pred_masks, feats = SegNet(feat)
            pred_mask1, pred_mask2, pred_mask3, pred_mask4 = pred_masks
            feat1, feat2, feat3, feat4 = feats

            pred_logit = ClsNet(feats)

            # 保证 pred_mask1 和 mask 对齐
            if pred_mask1.shape != mask.shape:
                pred_mask1 = F.interpolate(
                    pred_mask1,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

            # SegNet 输出的 pred_mask 已经是 0~1 概率(MoE + CBAM 后的 sigmoid),
            # 因此这里直接阈值，不再做 sigmoid
            binary_mask = (
                pred_mask1 > MASK_F1_THRESH
            ).cpu().numpy().astype(np.uint8)
            gt_mask = mask.cpu().numpy().astype(np.uint8)

            all_mask_pred.append(binary_mask.flatten())
            all_mask_gt.append(gt_mask.flatten())

            # 分类准确率统计
            sm = nn.Softmax(dim=1)
            pred_prob = sm(pred_logit)
            _, binary_cls = torch.max(pred_prob, 1)
            cls_correct += (binary_cls == cls).sum().item()
            cls_total += cls.size(0)

            # ---------- 固定图像可视化部分 ----------
            # epoch=-1 时不保存图
            if epoch >= 0 and batch_id % 100 == 0:
                tensor_mask = torch.from_numpy(binary_mask).float()
                if tensor_mask.ndim == 2:
                    tensor_mask = tensor_mask.unsqueeze(0)  # (1, H, W)

                epoch_folder = f"epoch_{epoch:03d}"

                if isinstance(name, (list, tuple)):
                    name_with_epoch = [f"{epoch_folder}/{n}" for n in name]
                else:
                    name_with_epoch = f"{epoch_folder}/{name}"

                save_image(tensor_mask, name_with_epoch, "mask")
            # ---------- 可视化部分结束 ----------

    all_mask_pred = np.concatenate(all_mask_pred)
    all_mask_gt = np.concatenate(all_mask_gt)
    seg_f1 = f1_score(all_mask_gt, all_mask_pred, zero_division=1)

    cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
    print(f"Validation: Seg F1={seg_f1:.4f}, Cls Acc={cls_acc:.4f}")
    return seg_f1, cls_acc


if __name__ == "__main__":
    args = get_pscc_args()
    train(args)
