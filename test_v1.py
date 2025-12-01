import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as tv_utils

from utils.utils import save_image
from utils.config import get_pscc_args
from utils.load_vdata import TestData

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet):
    ckpt = torch.load(ckpt_path, map_location=device)
    print("Loaded checkpoint:", ckpt_path)
    print("Available keys:", ckpt.keys())

    FENet.load_state_dict(ckpt["FENet"])
    SegNet.load_state_dict(ckpt["SegNet"])

    missing, unexpected = ClsNet.load_state_dict(ckpt["ClsNet"], strict=False)
    print("ClsNet missing keys:", missing)
    print("ClsNet unexpected keys (ignored):", unexpected)


def _unwrap_dp(m: nn.Module) -> nn.Module:
    """兼容 DataParallel"""
    return m.module if isinstance(m, nn.DataParallel) else m


def _norm01(t: torch.Tensor) -> torch.Tensor:
    """把任意 tensor 归一化到 [0,1] 便于可视化保存"""
    t = t.detach().float()
    tmin = t.amin(dim=(-2, -1), keepdim=True)
    tmax = t.amax(dim=(-2, -1), keepdim=True)
    return (t - tmin) / (tmax - tmin + 1e-6)


def _save_grid(maps_echw: torch.Tensor, save_path: str, nrow: int):
    """
    maps_echw: [E,1,H,W] or [E,3,H,W]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid = tv_utils.make_grid(maps_echw.cpu(), nrow=nrow, padding=2)
    tv_utils.save_image(grid, save_path)


def _edge_energy_sobel(x_hw: torch.Tensor) -> float:
    """简单边缘能量(高频强度)：Sobel 梯度均值"""
    x = x_hw[None, None].float()  # [1,1,H,W]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)[None, None]
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)[None, None]
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return g.mean().item()


def _find_moe_layers(segnet: nn.Module):
    """
    自动扫描 SegNet 中的 MoEAttention 模块（通过接口判断更稳）
    返回 [(name, module), ...]
    """
    segnet = _unwrap_dp(segnet)
    moe = []
    for name, m in segnet.named_modules():
        if hasattr(m, "enable_debug") and hasattr(m, "get_last_debug"):
            moe.append((name, m))
    return moe


def test(args):
    # =================== 清空输出目录 ===================
    if os.path.exists("mask_results/result/"):
        shutil.rmtree("mask_results/result/")
    os.makedirs("mask_results/result/", exist_ok=True)

    vis_root = "mask_results/moe_vis"
    if os.path.exists(vis_root):
        shutil.rmtree(vis_root)
    os.makedirs(vis_root, exist_ok=True)

    # =================== 定义网络 ===================
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg).to(device)

    SegNet = NLCDetection(args).to(device)
    ClsNet = DetectionHead(args).to(device)

    # =================== 加载 checkpoint ===================
    ckpt_path = "./total=15000_epoch=100_batch=6_with_MoE=True(MoE_attn=CBAM_K=8)_With_HiLo=True/checkpoint/NLCDetection_checkpoint/NLCDetection_60.pth"
    load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet)

    # =================== 找到 MoEAttention 并开启 debug ===================
    moe_layers = _find_moe_layers(SegNet)
    if len(moe_layers) == 0:
        print("[WARN] 没找到带 enable_debug/get_last_debug 的 MoEAttention。请确认 MoEAttention 方法写在 class 里(不是 __init__ 里)。")
        target_moe_name, target_moe = None, None
    else:
        print("[INFO] Found MoE layers:")
        for n, m in moe_layers:
            print("  -", n, "|", m.__class__.__name__)
            m.enable_debug(True)  # 开启缓存
        target_moe_name, target_moe = moe_layers[-1]  # 默认取最后一个 MoE（更靠近输出）
        print("[INFO] Use target MoE:", target_moe_name)

    # =================== DataLoader ===================
    test_data_loader = DataLoader(
        TestData(args),
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    FENet.eval()
    SegNet.eval()
    ClsNet.eval()

    authentic_num = 0

    # 最多保存多少张 forged 样本的专家图
    VIS_MAX = 50
    vis_saved = 0

    with torch.no_grad():
        global_idx = 0

        for batch_id, test_data in enumerate(test_data_loader):
            image, cls, name = test_data
            image = image.to(device, non_blocking=True)
            B = image.size(0)

            # backbone
            feat = FENet(image)

            # SegNet forward（注意：此时 target_moe 会缓存 last_debug）
            pred_masks, feats = SegNet(feat)
            pred_mask1 = pred_masks[0]  # [B,1,h,w]

            # 取 MoE debug（必须在 forward 之后）
            moe_debug = None
            if target_moe is not None:
                moe_debug = target_moe.get_last_debug()  # dict or None

            # 上采样到原图大小
            pred_mask1_up = F.interpolate(
                pred_mask1,
                size=(image.size(2), image.size(3)),
                mode="bilinear",
                align_corners=True,
            )
            prob_mask = torch.clamp(pred_mask1_up, 0.0, 1.0)  # [B,1,H,W]

            # 分类 head
            pred_logit = ClsNet(feats)  # [B,2]
            pred_prob = torch.softmax(pred_logit, dim=1)
            binary_cls = torch.argmax(pred_prob, dim=1)  # [B]

            for i in range(B):
                global_idx += 1

                cls_i = binary_cls[i].item()
                pred_tag = "forged" if cls_i == 1 else "authentic"
                if pred_tag == "authentic":
                    authentic_num += 1

                name_i = name[i]
                print_name = os.path.splitext(os.path.basename(name_i))[0]

                if global_idx % 100 == 0:
                    print(f"[{global_idx}] The image {print_name} is {pred_tag}")

                # 保存概率图（你的原逻辑）
                save_image(
                    prob_mask[i:i + 1],
                    os.path.join("result", print_name),
                    "mask"
                )

                # =================== 保存专家可视化 ===================
                if (
                    pred_tag == "forged"
                    and vis_saved < VIS_MAX
                    and moe_debug is not None
                    and isinstance(moe_debug, dict)
                    and "expert_masks" in moe_debug
                ):
                    vis_saved += 1
                    H, W = image.size(2), image.size(3)

                    # expert_masks: [B,E,h,w] -> [E,h,w]
                    expert_masks = moe_debug["expert_masks"][i]  # [E,h,w]
                    routing = moe_debug["routing"][i]            # [E,h,w]
                    expert_res = moe_debug["expert_res"][i]      # [E,C,h,w]

                    # 1) expert mask grid
                    masks_e1hw = expert_masks.unsqueeze(1)  # [E,1,h,w]
                    masks_e1hw = torch.clamp(masks_e1hw, 0.0, 1.0)
                    masks_e1HW = F.interpolate(masks_e1hw, size=(H, W), mode="bilinear", align_corners=True)
                    masks_e1HW = _norm01(masks_e1HW)

                    # 2) routing grid
                    routing_e1hw = routing.unsqueeze(1)  # [E,1,h,w]
                    routing_e1HW = F.interpolate(routing_e1hw, size=(H, W), mode="bilinear", align_corners=True)
                    routing_e1HW = _norm01(routing_e1HW)

                    # 3) feature heatmap grid（|res| 的通道均值）
                    feat_e1hw = expert_res.abs().mean(dim=1, keepdim=True)  # [E,1,h,w]
                    feat_e1HW = F.interpolate(feat_e1hw, size=(H, W), mode="bilinear", align_corners=True)
                    feat_e1HW = _norm01(feat_e1HW)

                    # 保存目录
                    sample_dir = os.path.join(vis_root, print_name)
                    os.makedirs(sample_dir, exist_ok=True)

                    # 原图
                    tv_utils.save_image(_norm01(image[i].detach().cpu()), os.path.join(sample_dir, "image.png"))

                    E = masks_e1HW.shape[0]
                    _save_grid(masks_e1HW, os.path.join(sample_dir, f"grid_expert_masks_E{E}.png"), nrow=E)
                    _save_grid(routing_e1HW, os.path.join(sample_dir, f"grid_routing_E{E}.png"), nrow=E)
                    _save_grid(feat_e1HW, os.path.join(sample_dir, f"grid_expert_feat_E{E}.png"), nrow=E)

                    # 高频/边界强度排序（给论文提供量化证据）
                    energy = []
                    for e in range(E):
                        em = _edge_energy_sobel(expert_masks[e])  # expert_masks[e] 已在 GPU 上
                        energy.append((e, em))
                    energy.sort(key=lambda x: x[1], reverse=True)

                    with open(os.path.join(sample_dir, "expert_edge_energy.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Target MoE layer: {target_moe_name}\n")
                        f.write("Expert edge-energy ranking (mask Sobel mean):\n")
                        for e, em in energy:
                            f.write(f"  expert {e:02d}: {em:.6f}\n")

                    print(f"[VIS] saved expert maps for {print_name} -> {sample_dir}")

    print(f"The num of authentic is {authentic_num}")
    print(f"[DONE] moe vis saved: {vis_saved} samples -> {vis_root}")


if __name__ == "__main__":
    args = get_pscc_args()
    test(args)
