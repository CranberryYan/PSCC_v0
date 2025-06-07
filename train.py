import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import findLastCheckpoint, save_image, adjust_learning_rate
from utils.config import get_pscc_args
from utils.load_tdata import TrainData, ValData
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead

# TODO: MoE不适用于多卡训练, 会有Bug
# device_ids = list(range(torch.cuda.device_count()))
device_ids = [0]
device = torch.device('cuda:0')

logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train(args):
    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    SegNet_name = 'NLCDetection'
    SegNet = NLCDetection(args)

    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(args)

    train_data_loader = DataLoader(
        TrainData(args),
        batch_size=args['train_bs'],
        shuffle=True, num_workers=8)

    FENet = FENet.to(device)
    SegNet = SegNet.to(device)
    ClsNet = ClsNet.to(device)

    # 多卡并行
    FENet = nn.DataParallel(FENet, device_ids=device_ids)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
    ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)

    params = list(FENet.parameters()) + list(SegNet.parameters()) + list(ClsNet.parameters())
    optimizer = torch.optim.Adam(params, lr=args['learning_rate'])

    FENet_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
    if not os.path.exists(FENet_dir):
        os.mkdir(FENet_dir)
    SegNet_dir = './checkpoint/{}_checkpoint'.format(SegNet_name)
    if not os.path.exists(SegNet_dir):
        os.mkdir(SegNet_dir)
    ClsNet_dir = './checkpoint/{}_checkpoint'.format(ClsNet_name)
    if not os.path.exists(ClsNet_dir):
        os.mkdir(ClsNet_dir)

    try:
        FENet_weight_path = '{}/{}.pth'.format(FENet_dir, FENet_name)
        FENet_state_dict = torch.load(FENet_weight_path, map_location='cuda:0')
        FENet.load_state_dict(FENet_state_dict)
        logging.info('{} weight-loading succeeds: {}'.format(FENet_name, FENet_weight_path))
    except Exception as e:
        logging.info('{} weight-loading fails: {}'.format(FENet_name, e))
    try:
        SegNet_weight_path = '{}/{}.pth'.format(SegNet_dir, SegNet_name)
        SegNet_state_dict = torch.load(SegNet_weight_path, map_location='cuda:0')
        SegNet.load_state_dict(SegNet_state_dict)
        logging.info('{} weight-loading succeeds: {}'.format(SegNet_name, SegNet_weight_path))
    except Exception as e:
        logging.info('{} weight-loading fails: {}'.format(SegNet_name, e))
    try:
        ClsNet_weight_path = '{}/{}.pth'.format(ClsNet_dir, ClsNet_name)
        ClsNet_state_dict = torch.load(ClsNet_weight_path, map_location='cuda:0')
        ClsNet.load_state_dict(ClsNet_state_dict)
        logging.info('{} weight-loading succeeds: {}'.format(ClsNet_name, ClsNet_weight_path))
    except Exception as e:
        logging.info('{} weight-loading fails: {}'.format(ClsNet_name, e))

    logging.info('length of traindata: {}'.format(len(train_data_loader)))
    previous_score = validation(FENet, SegNet, ClsNet, args)
    logging.info('previous_score {0:.4f}'.format(previous_score))

    authentic_ratio = args['train_ratio'][0]
    fake_ratio = 1 - authentic_ratio
    logging.info('authentic_ratio: {}  fake_ratio: {}'.format(authentic_ratio, fake_ratio))
    weights = [1. / authentic_ratio, 1. / fake_ratio]
    weights = torch.tensor(weights).to(device)
    CE_loss = nn.CrossEntropyLoss(weight=weights).to(device)

    BCE_loss_full = nn.BCEWithLogitsLoss(reduction='none').to(device)

    # 断点续训
    initial_epoch = findLastCheckpoint(save_dir=SegNet_dir)
    if initial_epoch > 0:
        try:
            FENet_checkpoint = torch.load('{0}/{1}_{2}.pth'.format(FENet_dir, FENet_name, initial_epoch))
            FENet.load_state_dict(FENet_checkpoint['model'])
            logging.info("resuming FENet by loading epoch {}".format(initial_epoch))

            SegNet_checkpoint = torch.load('{0}/{1}_{2}.pth'.format(SegNet_dir, SegNet_name, initial_epoch))
            SegNet.load_state_dict(SegNet_checkpoint['model'])
            logging.info("resuming SegNet by loading epoch {}".format(initial_epoch))

            ClsNet_checkpoint = torch.load('{0}/{1}_{2}.pth'.format(ClsNet_dir, ClsNet_name, initial_epoch))
            ClsNet.load_state_dict(ClsNet_checkpoint['model'])
            optimizer.load_state_dict(ClsNet_checkpoint['optimizer'])
            logging.info("resuming ClsNet by loading epoch {}".format(initial_epoch))
        except Exception as e:
            logging.info('cannot load checkpoint on epoch {}: {}'.format(initial_epoch, e))
            initial_epoch = 0
            logging.info("resuming by loading epoch {}".format(initial_epoch))

    epoch_list = []
    loss_list = []
    score_list = []
    diversity_history = []
    routing_history = []

    for epoch in range(initial_epoch, args['num_epochs']):
        adjust_learning_rate(optimizer, epoch, args['lr_strategy'], args['lr_decay_step'])
        seg_total, seg_correct, seg_loss_sum = 0, 0, 0
        cls_total, cls_correct, cls_loss_sum = 0, 0, 0

        for batch_id, train_data in enumerate(train_data_loader):
            # image, [mask, mask2, mask3, mask4], cls
            image, masks, cls = train_data
            cls[cls != 0] = 1  # 只要不是origin, 不区分fake种类
            mask1, mask2, mask3, mask4 = masks

            mask1_balance = torch.ones_like(mask1)
            if (mask1 == 1).sum():
                mask1_balance[mask1 == 1] = 0.5 / ((mask1 == 1).sum().float() / mask1.numel())
                mask1_balance[mask1 == 0] = 0.5 / ((mask1 == 0).sum().float() / mask1.numel())
            else:
                logging.info('Mask1 balance is not working!')
            mask2_balance = torch.ones_like(mask2)
            if (mask2 == 1).sum():
                mask2_balance[mask2 == 1] = 0.5 / ((mask2 == 1).sum().float() / mask2.numel())
                mask2_balance[mask2 == 0] = 0.5 / ((mask2 == 0).sum().float() / mask2.numel())
            else:
                logging.info('Mask2 balance is not working!')
            mask3_balance = torch.ones_like(mask3)
            if (mask3 == 1).sum():
                mask3_balance[mask3 == 1] = 0.5 / ((mask3 == 1).sum().float() / mask3.numel())
                mask3_balance[mask3 == 0] = 0.5 / ((mask3 == 0).sum().float() / mask3.numel())
            else:
                logging.info('Mask3 balance is not working!')
            mask4_balance = torch.ones_like(mask4)
            if (mask4 == 1).sum():
                mask4_balance[mask4 == 1] = 0.5 / ((mask4 == 1).sum().float() / mask4.numel())
                mask4_balance[mask4 == 0] = 0.5 / ((mask4 == 0).sum().float() / mask4.numel())
            else:
                logging.info('Mask4 balance is not working!')

            image = image.to(device)
            mask1, mask2, mask3, mask4 = mask1.to(device), mask2.to(device), mask3.to(device), mask4.to(device)
            mask1_balance = mask1_balance.to(device)
            mask2_balance = mask2_balance.to(device)
            mask3_balance = mask3_balance.to(device)
            mask4_balance = mask4_balance.to(device)
            cls = cls.to(device)

            optimizer.zero_grad()

            # Feature extraction network
            FENet.train()
            feat = FENet(image)

            # Segmentation network
            SegNet.train()
            current_temp = max(0.5, 2.0 * (1 - epoch / 100))
            SegNet.module.getmask4.temp = current_temp
            SegNet.module.getmask3.temp = current_temp
            SegNet.module.getmask2.temp = current_temp
            SegNet.module.getmask1.temp = current_temp
            [pred_mask1, pred_mask2, pred_mask3, pred_mask4] = SegNet(feat)
            div4 = SegNet.module.getmask4.get_diversity()
            div3 = SegNet.module.getmask3.get_diversity()
            div2 = SegNet.module.getmask2.get_diversity()
            div1 = SegNet.module.getmask1.get_diversity()
            diversity_history.append([div4, div3, div2, div1])
            rout4 = SegNet.module.getmask4.get_routing_stats()
            rout3 = SegNet.module.getmask3.get_routing_stats()
            rout2 = SegNet.module.getmask2.get_routing_stats()
            rout1 = SegNet.module.getmask1.get_routing_stats()
            routing_history.append([rout4, rout3, rout2, rout1])
            if batch_id % 10 == 9:
                moe_log = []
                for k, div, rout, temp in zip(
                    ["getmask4", "getmask3", "getmask2", "getmask1"],
                    [div4, div3, div2, div1],
                    [rout4, rout3, rout2, rout1],
                    [SegNet.module.getmask4.temp, SegNet.module.getmask3.temp,
                     SegNet.module.getmask2.temp, SegNet.module.getmask1.temp]):
                    moe_log.append(f"{k}:temp={temp:.4f}, div={div:.4f}, route={np.round(rout, 3)}")
                moe_log_str = " | ".join(moe_log)

                logging.info(
                    '[Epoch {0}, Batch {1}] MoE: {2}'
                    .format(epoch + 1, batch_id + 1, moe_log_str))

            # Classification network
            ClsNet.train()
            pred_logit = ClsNet(feat)

            pred_mask1 = pred_mask1.squeeze(dim=1)
            pred_mask2 = pred_mask2.squeeze(dim=1)
            pred_mask3 = pred_mask3.squeeze(dim=1)
            pred_mask4 = pred_mask4.squeeze(dim=1)

            mask1_loss = torch.mean(BCE_loss_full(pred_mask1, mask1) * mask1_balance)
            mask2_loss = torch.mean(BCE_loss_full(pred_mask2, mask2) * mask2_balance)
            mask3_loss = torch.mean(BCE_loss_full(pred_mask3, mask3) * mask3_balance)
            mask4_loss = torch.mean(BCE_loss_full(pred_mask4, mask4) * mask4_balance)
            seg_loss = mask1_loss + mask2_loss + mask3_loss + mask4_loss

            cls_loss = CE_loss(pred_logit, cls)
            loss = seg_loss + cls_loss

            loss.backward()
            optimizer.step()

            # localization accuracy
            binary_mask1 = torch.zeros_like(pred_mask1)
            binary_mask1[pred_mask1 > 0.5] = 1
            binary_mask1[pred_mask1 <= 0.5] = 0

            seg_correct += (binary_mask1 == mask1).sum().item()
            seg_total += int(torch.ones_like(mask1).sum().item())

            _, binary_cls = torch.max(pred_logit, 1)
            cls_correct += (binary_cls == cls).sum().item()
            cls_total += int(torch.ones_like(cls).sum().item())

            seg_loss_sum += seg_loss.item()
            cls_loss_sum += cls_loss.item()

            if batch_id % 10 == 9:
                logging.info('[Epoch {0}, Batch {1}] Localization Accuracy: [{2}/{3}] {4:.2f}%, seg_loss: {5:.4f}; '
                             'Classification Accuracy: [{6}/{7}] {8:.2f}%, cls_loss: {9:.4f}'.format(
                                epoch + 1, batch_id + 1, seg_correct, seg_total,
                                seg_correct / seg_total * 100,
                                seg_loss_sum / 10, cls_correct, cls_total,
                                cls_correct / cls_total * 100,
                                cls_loss_sum / 10))
                seg_total, seg_correct, seg_loss_sum, cls_total, cls_correct, cls_loss_sum = 0, 0, 0, 0, 0, 0

        checkpoint = {
            'epoch': epoch + 1,
            'FENet': FENet.state_dict(),
            'SegNet': SegNet.state_dict(),
            'ClsNet': ClsNet.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, '{0}/{1}_{2}.pth'.format(FENet_dir, FENet_name, epoch + 1))
        torch.save(checkpoint, '{0}/{1}_{2}.pth'.format(SegNet_dir, SegNet_name, epoch + 1))
        torch.save(checkpoint, '{0}/{1}_{2}.pth'.format(ClsNet_dir, ClsNet_name, epoch + 1))

        current_score = validation(FENet, SegNet, ClsNet, args)
        logging.info('Epoch {0}: current_score: {1:.4f}'.format(epoch + 1, current_score))

        if current_score >= previous_score:
            torch.save(FENet.state_dict(), '{0}/{1}.pth'.format(FENet_dir, FENet_name))
            torch.save(SegNet.state_dict(), '{0}/{1}.pth'.format(SegNet_dir, SegNet_name))
            torch.save(ClsNet.state_dict(), '{0}/{1}.pth'.format(ClsNet_dir, ClsNet_name))
            previous_score = current_score

        epoch_list.append(epoch + 1)
        loss_list.append(loss.item())
        score_list.append(current_score)

    logging.info("Training finished.")
    div_arr = np.array(diversity_history)  # shape: [steps, 4]
    for i, name in enumerate(["getmask4", "getmask3", "getmask2", "getmask1"]):
        plt.plot(div_arr[:, i], label=name)
    plt.legend()
    plt.title("Expert Diversity (Variance) at Different Scales")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.grid()
    plt.show()

    routing_arr = np.array(routing_history)  # shape: [steps, 4, num_experts]
    for i, name in enumerate(["getmask4", "getmask3", "getmask2", "getmask1"]):
        plt.bar(np.arange(routing_arr.shape[2]), routing_arr[-1, i, :], alpha=0.7, label=name)
    plt.legend()
    plt.title("Routing Distribution (last step)")
    plt.xlabel("Expert ID")
    plt.ylabel("Mean Routing")
    plt.show()

def validation(FENet, SegNet, ClsNet, args):
    val_data_loader = DataLoader(ValData(args), batch_size=args['val_bs'], shuffle=False, num_workers=8)
    seg_correct, seg_total, cls_correct, cls_total = 0, 0, 0, 0

    for batch_id, val_data in enumerate(val_data_loader):
        image, mask, cls, name = val_data
        image = image.to(device)
        mask = mask.to(device)
        cls[cls != 0] = 1
        cls = cls.to(device)

        with torch.no_grad():
            FENet.eval()
            feat = FENet(image)
            SegNet.eval()
            pred_mask = SegNet(feat)[0]
            ClsNet.eval()
            pred_logit = ClsNet(feat)

        if pred_mask.shape != mask.shape:
            pred_mask = F.interpolate(pred_mask, size=(mask.size(1), mask.size(2)), mode='bilinear', align_corners=True)

        binary_mask1 = torch.zeros_like(pred_mask)
        binary_mask1[pred_mask > 0.5] = 1
        binary_mask1[pred_mask <= 0.5] = 0

        seg_correct += (binary_mask1 == mask).sum().item()
        seg_total += int(torch.ones_like(mask).sum().item())

        sm = nn.Softmax(dim=1)
        pred_logit = sm(pred_logit)
        _, binary_cls = torch.max(pred_logit, 1)
        cls_correct += (binary_cls == cls).sum().item()
        cls_total += int(torch.ones_like(cls).sum().item())

        if batch_id % 100 == 0:
            save_image(binary_mask1, name, 'mask')

    return (seg_correct / seg_total + cls_correct / cls_total) / 2

if __name__ == '__main__':
    args = get_pscc_args()
    train(args)
