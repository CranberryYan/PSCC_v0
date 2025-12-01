import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import save_image
from utils.config import get_pscc_args
from utils.load_vdata import TestData

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead

import os
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet):
    ckpt = torch.load(ckpt_path, map_location=device)
    print("Loaded checkpoint:", ckpt_path)
    print("Available keys:", ckpt.keys())

    FENet.load_state_dict(ckpt["FENet"])
    SegNet.load_state_dict(ckpt["SegNet"])

    # 兼容老的 DetectionHead 结构，忽略多出来的 bias
    missing, unexpected = ClsNet.load_state_dict(ckpt["ClsNet"], strict=False)
    print("ClsNet missing keys:", missing)
    print("ClsNet unexpected keys (ignored):", unexpected)


def test(args):
    # 先清空 result
    if os.path.exists('mask_results/result/'):
        shutil.rmtree('mask_results/result/')
    os.makedirs('mask_results/result/', exist_ok=True)

    # 1. 定义网络
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg).to(device)

    SegNet = NLCDetection(args).to(device)
    ClsNet = DetectionHead(args).to(device)

    # 2. 加载 checkpoint（按你目前的路径）
    ckpt_path = "./checkpoint/NLCDetection_checkpoint/NLCDetection_best.pth"
    load_checkpoint_all(ckpt_path, FENet, SegNet, ClsNet)

    # 3. DataLoader
    test_data_loader = DataLoader(
        TestData(args),
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    authentic_num = 0

    FENet.eval()
    SegNet.eval()
    ClsNet.eval()

    with torch.no_grad():
        global_idx = 0  # 全局图片计数，用来控制每多少张打印一次

        for batch_id, test_data in enumerate(test_data_loader):
            image, cls, name = test_data        # name: 长度为 B 的 list
            image = image.to(device)
            B = image.size(0)

            # backbone
            feat = FENet(image)

            # SegNet: (pred_masks, feats)
            pred_masks, feats = SegNet(feat)
            pred_mask1 = pred_masks[0]  # [B,1,h,w]

            # 上采样到原图大小
            pred_mask1_up = F.interpolate(
                pred_mask1,
                size=(image.size(2), image.size(3)),
                mode="bilinear",
                align_corners=True,
            )

            # 概率图 0~1
            prob_mask = torch.clamp(pred_mask1_up, 0.0, 1.0)  # [B,1,H,W]

            # 分类 head
            pred_logit = ClsNet(feats)              # [B, 2]
            pred_prob = torch.softmax(pred_logit, dim=1)
            binary_cls = torch.argmax(pred_prob, dim=1)  # [B]

            # 按 batch 内每张图单独处理
            for i in range(B):
                global_idx += 1

                cls_i = binary_cls[i].item()  # 0 or 1
                pred_tag = "forged" if cls_i == 1 else "authentic"
                if pred_tag == "authentic":
                    authentic_num += 1

                # 当前样本的文件名
                name_i = name[i]
                print_name = os.path.splitext(os.path.basename(name_i))[0]

                # 每 100 张 或者你想要的间隔打印一次
                if global_idx % 100 == 0:
                    print(f"[{global_idx}] The image {print_name} is {pred_tag}")

                # 保存当前样本的概率图
                # 这里用 prob_mask[i:i+1] 保持形状 [1,1,H,W]，兼容你原来的 save_image
                save_image(
                    prob_mask[i:i+1],
                    os.path.join('result', print_name),
                    "mask"
                )

    print(f"The num of authentic is {authentic_num}")


if __name__ == "__main__":
    args = get_pscc_args()
    test(args)
