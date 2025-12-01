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


def _toggle_moe_debug(m, flag: bool):
    """兼容不同实现：enable_debug(True/False)"""
    if m is None:
        return
    if hasattr(m, "enable_debug"):
        try:
            m.enable_debug(flag)
        except TypeError:
            # 有些实现可能是 enable_debug() 无参
            if flag:
                m.enable_debug()
    # 尝试清空历史缓存（如果你的 MoE 实现提供的话）
    for fn in ("clear_debug", "reset_debug", "clear_last_debug"):
        if hasattr(m, fn):
            try:
                getattr(m, fn)()
            except Exception:
                pass


def test(args):
    # =================== 清空输出目录 ===================
    out_root = "./"
    result_root = os.path.join(out_root, "result")
    vis_root = os.path.join(out_root, "moe_vis")

    if os.path.exists(result_root):
        shutil.rmtree(result_root)
    os.makedirs(result_root, exist_ok=True)

    if os.path.exists(vis_root):
        shutil.rmtree(vis_root)
    os.makedirs(vis_root, exist_ok=True)

    # =================== 定义网络 ===================
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg).to(device)

    SegNet = NLCDetection(args).to(device)
    ClsNet = DetectionHead(args).to(device)

    # =================== 加载 checkpoint ===================
    ckpt_path = "./total=15000_epochs=100_batch=6_with_MoE=False(MoE_attn=False_K=8)_with_HiLo=False/checkpoint/NLCDetection_checkpoint/NLCDetection_35.pth"
    load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet)

    # =================== 找到 MoEAttention（默认关 debug，避免显存暴涨） ===================
    moe_layers = _find_moe_layers(SegNet)
    if len(moe_layers) == 0:
        print("[WARN] 没找到带 enable_debug/get_last_debug 的 MoEAttention。")
        target_moe_name, target_moe = None, None
    else:
        print("[INFO] Found MoE layers:")
        for n, m in moe_layers:
            print("  -", n, "|", m.__class__.__name__)
            _toggle_moe_debug(m, False)  # 默认关闭缓存
        target_moe_name, target_moe = moe_layers[-1]
        print("[INFO] Use target MoE:", target_moe_name)

    # =================== DataLoader ===================
    # pin_memory=True 有时会引发 host allocation failed（尤其 WSL/内存紧张）
    test_data_loader = DataLoader(
        TestData(args),
        batch_size=48,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
    )

    FENet.eval()
    SegNet.eval()
    ClsNet.eval()

    authentic_num = 0
    VIS_MAX = 0
    vis_saved = 0

    # 是否保存 expert_res 的特征热力图（它最吃显存/内存；建议先关掉跑通）
    SAVE_EXPERT_RES_HEATMAP = False

    from torch.cuda.amp import autocast

    # 用 inference_mode 更省显存/更快（比 no_grad 更强）
    with torch.inference_mode():
        global_idx = 0

        for batch_id, test_data in enumerate(test_data_loader):
            image, cls, name = test_data
            image = image.to(device)
            B = image.size(0)

            # backbone + segnet + clsnet：全部用 AMP 半精度推理，显存会降很多
            with autocast(enabled=(device.type == "cuda")):
                feat = FENet(image)
                # pred_masks, feats, aux_losses = SegNet(feat)
                pred_masks, feats = SegNet(feat)
                pred_mask1 = pred_masks[0]  # [B,1,h,w]

                pred_logit = ClsNet(feats)  # [B,2]
                pred_prob = torch.softmax(pred_logit, dim=1)
                binary_cls = torch.argmax(pred_prob, dim=1)  # [B]

            # 上采样到原图大小
            pred_mask1_up = F.interpolate(
                pred_mask1.float(),
                size=(image.size(2), image.size(3)),
                mode="bilinear",
                align_corners=True,
            )
            prob_mask = torch.clamp(pred_mask1_up, 0.0, 1.0)  # [B,1,H,W]

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

                # 保存概率图（修正路径到 mask_results/result）
                save_image(
                    prob_mask[i:i + 1],
                    os.path.join(result_root, print_name),
                    "mask"
                )

                # =================== 仅在需要可视化的 forged 样本上，临时打开 MoE debug 并重跑一次 SegNet ===================
                if (
                    pred_tag == "forged"
                    and vis_saved < VIS_MAX
                    and target_moe is not None
                ):
                    vis_saved += 1
                    H, W = image.size(2), image.size(3)

                    # 临时开 debug（避免每张图都缓存导致 OOM）
                    _toggle_moe_debug(target_moe, True)

                    with autocast(enabled=(device.type == "cuda")):
                        # 只需要 MoE debug，因此直接复用 feat 重跑 SegNet（比重跑 FENet 省）
                        _pred_masks_dbg, _feats_dbg, _aux_dbg = SegNet(feat)

                    moe_debug = target_moe.get_last_debug()

                    # 立刻关 debug，尽量释放缓存
                    _toggle_moe_debug(target_moe, False)

                    if not (isinstance(moe_debug, dict) and "expert_masks" in moe_debug and "routing" in moe_debug):
                        print(f"[VIS] skip {print_name}: moe_debug invalid")
                        continue

                    # ===== 把 debug 张量尽快搬到 CPU，避免 GPU 常驻 =====
                    expert_masks = moe_debug["expert_masks"][i].detach().float().cpu()  # [E,h,w]
                    routing = moe_debug["routing"][i].detach().float().cpu()            # [E,h,w]

                    expert_res = None
                    if SAVE_EXPERT_RES_HEATMAP and ("expert_res" in moe_debug):
                        # 这个非常大：只有你确实需要再开
                        expert_res = moe_debug["expert_res"][i].detach().float().cpu()  # [E,C,h,w]

                    # 释放本轮 debug 相关 GPU 引用
                    del _pred_masks_dbg, _feats_dbg, _aux_dbg, moe_debug

                    # ===== 生成可视化 =====
                    sample_dir = os.path.join(vis_root, print_name)
                    os.makedirs(sample_dir, exist_ok=True)

                    # 原图
                    tv_utils.save_image(_norm01(image[i].detach().cpu()), os.path.join(sample_dir, "image.png"))

                    # 1) expert mask grid
                    masks_e1hw = expert_masks.unsqueeze(1)  # [E,1,h,w]
                    masks_e1hw = torch.clamp(masks_e1hw, 0.0, 1.0)
                    masks_e1HW = F.interpolate(masks_e1hw, size=(H, W), mode="bilinear", align_corners=True)
                    masks_e1HW = _norm01(masks_e1HW)

                    # 2) routing grid
                    routing_e1hw = routing.unsqueeze(1)
                    routing_e1HW = F.interpolate(routing_e1hw, size=(H, W), mode="bilinear", align_corners=True)
                    routing_e1HW = _norm01(routing_e1HW)

                    # 3) feature heatmap grid（可选）
                    if expert_res is not None:
                        feat_e1hw = expert_res.abs().mean(dim=1, keepdim=True)  # [E,1,h,w]
                        feat_e1HW = F.interpolate(feat_e1hw, size=(H, W), mode="bilinear", align_corners=True)
                        feat_e1HW = _norm01(feat_e1HW)
                    else:
                        feat_e1HW = None

                    E = masks_e1HW.shape[0]
                    _save_grid(masks_e1HW, os.path.join(sample_dir, f"grid_expert_masks_E{E}.png"), nrow=E)
                    _save_grid(routing_e1HW, os.path.join(sample_dir, f"grid_routing_E{E}.png"), nrow=E)
                    if feat_e1HW is not None:
                        _save_grid(feat_e1HW, os.path.join(sample_dir, f"grid_expert_feat_E{E}.png"), nrow=E)

                    # ===== 边缘能量排序：在 CPU 上算，别占 GPU =====
                    energy = []
                    for e in range(E):
                        em = _edge_energy_sobel(expert_masks[e])  # expert_masks 已经是 CPU tensor
                        energy.append((e, em))
                    energy.sort(key=lambda x: x[1], reverse=True)

                    with open(os.path.join(sample_dir, "expert_edge_energy.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Target MoE layer: {target_moe_name}\n")
                        f.write("Expert edge-energy ranking (mask Sobel mean):\n")
                        for e, em in energy:
                            f.write(f"  expert {e:02d}: {em:.6f}\n")

                    print(f"[VIS] saved expert maps for {print_name} -> {sample_dir}")

                    # 尽量释放 CPU 大对象引用
                    del expert_masks, routing, expert_res, masks_e1HW, routing_e1HW, feat_e1HW, energy

            # 每轮释放 GPU 引用（避免你后面改代码时不小心 list 累积）
            # del pred_masks, feats, aux_losses, pred_mask1, pred_mask1_up, prob_mask, pred_logit, pred_prob, binary_cls, feat
            del pred_masks, feats, pred_mask1, pred_mask1_up, prob_mask, pred_logit, pred_prob, binary_cls, feat
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"The num of authentic is {authentic_num}")
    print(f"[DONE] moe vis saved: {vis_saved} samples -> {vis_root}")


if __name__ == "__main__":
    args = get_pscc_args()
    test(args)
