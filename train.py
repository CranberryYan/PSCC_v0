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

# TODO: MoE不适用于多卡训练, 会有Bug
# device_ids = list(range(torch.cuda.device_count()))
device_ids = [0]
device = torch.device('cuda:0')

is_MoE = True
is_HiLo = True

def train(args):
  filename = f'total={args["train_num"]}_epochs={args["num_epochs"]}_batch={args["train_bs"]}_with_MoE={is_MoE}(CBAM_K=8)_with_HiLo={is_HiLo}.log'

  logging.basicConfig(filename=filename, level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')
  FENet_name = 'HRNet'
  FENet_cfg = get_hrnet_cfg()
  FENet = get_seg_model(FENet_cfg)

  SegNet_name = 'NLCDetection'
  SegNet = NLCDetection(args)

  ClsNet_name = 'DetectionHead'
  ClsNet = DetectionHead(args)

  train_data_loader = DataLoader(TrainData(args),
                                 batch_size=args['train_bs'],
                                 shuffle=True, num_workers=8)

  FENet = FENet.to(device)
  SegNet = SegNet.to(device)
  ClsNet = ClsNet.to(device)

  # 多卡并行
  FENet = nn.DataParallel(FENet, device_ids=device_ids)
  SegNet = nn.DataParallel(SegNet, device_ids=device_ids)
  ClsNet = nn.DataParallel(ClsNet, device_ids=device_ids)

  params = list(FENet.parameters()) + \
           list(SegNet.parameters()) + \
           list(ClsNet.parameters())
  optimizer = torch.optim.Adam(params, lr=args['learning_rate'])

  FENet_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
  if not os.path.exists(FENet_dir):
    os.makedirs(FENet_dir, exist_ok=True)
  SegNet_dir = './checkpoint/{}_checkpoint'.format(SegNet_name)
  if not os.path.exists(SegNet_dir):
    os.makedirs(SegNet_dir, exist_ok=True)
  ClsNet_dir = './checkpoint/{}_checkpoint'.format(ClsNet_name)
  if not os.path.exists(ClsNet_dir):
    os.makedirs(ClsNet_dir, exist_ok=True)

  try:
    FENet_weight_path = '{}/{}.pth'.format(FENet_dir, FENet_name)
    FENet_state_dict = torch.load(FENet_weight_path, map_location='cuda:0')
    FENet.load_state_dict(FENet_state_dict)
    logging.info(
      '{} weight-loading succeed: {}'.format(FENet_name, FENet_weight_path))
  except Exception as e:
    logging.info(
      '{} weight-loading fails: {}'.format(FENet_name, e))
  try:
    SegNet_weight_path = '{}/{}.pth'.format(SegNet_dir, SegNet_name)
    SegNet_state_dict = torch.load(SegNet_weight_path, map_location='cuda:0')
    SegNet.load_state_dict(SegNet_state_dict)
    logging.info(
      '{} weight-loading succeed: {}'.format(SegNet_name, SegNet_weight_path))
  except Exception as e:
    logging.info(
      '{} weight-loading fails: {}'.format(SegNet_name, e))
  try:
    ClsNet_weight_path = '{}/{}.pth'.format(ClsNet_dir, ClsNet_name)
    ClsNet_state_dict = torch.load(ClsNet_weight_path, map_location='cuda:0')
    ClsNet.load_state_dict(ClsNet_state_dict)
    logging.info(
      '{} weight-loading succeed: {}'.format(ClsNet_name, ClsNet_weight_path))
  except Exception as e:
    logging.info('{} weight-loading fails: {}'.format(ClsNet_name, e))

  logging.info('length of traindata: {}'.format(len(train_data_loader)))
  previous_score = validation(FENet, SegNet, ClsNet, args, -1)
  logging.info('previous_score {0:.4f}'.format(previous_score))

  authentic_ratio = args['train_ratio'][0]
  fake_ratio = 1 - authentic_ratio
  logging.info(
    'authentic_ratio: {}  fake_ratio: {}'.format(authentic_ratio, fake_ratio))
  weights = [1. / authentic_ratio, 1. / fake_ratio]
  weights = torch.tensor(weights).to(device)
  CE_loss = nn.CrossEntropyLoss(weight=weights).to(device)
  BCE_loss_full = nn.BCELoss(reduction='none').to(device)

  # 断点续训
  initial_epoch = findLastCheckpoint(save_dir=SegNet_dir)
  if initial_epoch > 0:
    try:
      FENet_checkpoint = torch.load(
        '{0}/{1}_{2}.pth'.format(FENet_dir, FENet_name, initial_epoch))
      FENet.load_state_dict(FENet_checkpoint['model'])
      logging.info(
        "resuming FENet by loading epoch {}".format(initial_epoch))

      SegNet_checkpoint = torch.load(
        '{0}/{1}_{2}.pth'.format(SegNet_dir, SegNet_name, initial_epoch))
      SegNet.load_state_dict(SegNet_checkpoint['model'])
      logging.info(
        "resuming SegNet by loading epoch {}".format(initial_epoch))

      ClsNet_checkpoint = torch.load(
        '{0}/{1}_{2}.pth'.format(ClsNet_dir, ClsNet_name, initial_epoch))
      ClsNet.load_state_dict(ClsNet_checkpoint['model'])
      optimizer.load_state_dict(ClsNet_checkpoint['optimizer'])
      logging.info(
        "resuming ClsNet by loading epoch {}".format(initial_epoch))
    except Exception as e:
      logging.info(
        'cannot load checkpoint on epoch {}: {}'.format(initial_epoch, e))
      initial_epoch = 0
      logging.info(
        "resuming by loading epoch {}".format(initial_epoch))

  f1_sum = 0
  batch_count = 0
  epoch_list = []
  loss_list = []
  score_list = []
  diversity_history = []
  routing_history = []

  previous_score = -1e9
  best_epoch = 0

  for epoch in range(initial_epoch, args['num_epochs']):
    adjust_learning_rate(optimizer,
                         epoch,
                         args['lr_strategy'], args['lr_decay_step'])
    seg_total, seg_correct, seg_loss_sum = 0, 0, 0
    cls_total, cls_correct, cls_loss_sum = 0, 0, 0

    for batch_id, train_data in enumerate(train_data_loader):
      # image, [mask, mask2, mask3, mask4], cls
      image, masks, cls = train_data
      cls[cls != 0] = 1 # 只要不是origin, 不区分fake种类
      mask1, mask2, mask3, mask4 = masks

      mask1_balance = torch.ones_like(mask1)
      if (mask1 == 1).sum():
        mask1_balance[mask1 == 1] = \
          0.5 / ((mask1 == 1).sum().float() / mask1.numel())
        mask1_balance[mask1 == 0] = \
          0.5 / ((mask1 == 0).sum().float() / mask1.numel())
      else:
        logging.info('Mask1 balance is not working!')

      mask2_balance = torch.ones_like(mask2)
      if (mask2 == 1).sum():
        mask2_balance[mask2 == 1] = \
          0.5 / ((mask2 == 1).sum().float() / mask2.numel())
        mask2_balance[mask2 == 0] = \
          0.5 / ((mask2 == 0).sum().float() / mask2.numel())
      else:
        logging.info('Mask2 balance is not working!')

      mask3_balance = torch.ones_like(mask3)
      if (mask3 == 1).sum():
        mask3_balance[mask3 == 1] = \
          0.5 / ((mask3 == 1).sum().float() / mask3.numel())
        mask3_balance[mask3 == 0] = \
          0.5 / ((mask3 == 0).sum().float() / mask3.numel())
      else:
        logging.info('Mask3 balance is not working!')

      mask4_balance = torch.ones_like(mask4)
      if (mask4 == 1).sum():
        mask4_balance[mask4 == 1] = \
          0.5 / ((mask4 == 1).sum().float() / mask4.numel())
        mask4_balance[mask4 == 0] = \
          0.5 / ((mask4 == 0).sum().float() / mask4.numel())
      else:
        logging.info('Mask4 balance is not working!')

      image = image.to(device)
      mask1, mask2, mask3, mask4 = (mask1.to(device),
                                    mask2.to(device),
                                    mask3.to(device),
                                    mask4.to(device))
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
      pred_masks, feats = SegNet(feat)
      pred_mask1, pred_mask2, pred_mask3, pred_mask4 = pred_masks
      feat1, feat2, feat3, feat4 = feats
      if is_MoE:
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
            [SegNet.module.getmask4.temp,
            SegNet.module.getmask3.temp,
            SegNet.module.getmask2.temp,
            SegNet.module.getmask1.temp]):
            moe_log.append(
              f"{k}:temp={temp:.4f}, div={div:.4f}, route={np.round(rout, 3)}")
          moe_log_str = " | ".join(moe_log)

          logging.info(
            "[Epoch {0}, Batch {1}] MoE: {2}".format(
              epoch + 1, batch_id + 1, moe_log_str))

      # Classification network
      ClsNet.train()
      pred_logit = ClsNet(feats)

      pred_mask1 = pred_mask1.squeeze(dim=1)
      pred_mask2 = pred_mask2.squeeze(dim=1)
      pred_mask3 = pred_mask3.squeeze(dim=1)
      pred_mask4 = pred_mask4.squeeze(dim=1)

      pred_mask1 = torch.clamp(pred_mask1, 0.0, 1.0)
      pred_mask2 = torch.clamp(pred_mask2, 0.0, 1.0)
      pred_mask3 = torch.clamp(pred_mask3, 0.0, 1.0)
      pred_mask4 = torch.clamp(pred_mask4, 0.0, 1.0)

      # 检查是否有NaN或inf
      if (torch.isnan(pred_mask1).any()
          or torch.isinf(pred_mask1).any()):
        print("Warning: pred_mask1 contains NaN or inf")
        continue

      mask1_loss = \
        torch.mean(BCE_loss_full(pred_mask1, mask1) * mask1_balance)
      mask2_loss = \
        torch.mean(BCE_loss_full(pred_mask2, mask2) * mask2_balance)
      mask3_loss = \
        torch.mean(BCE_loss_full(pred_mask3, mask3) * mask3_balance)
      mask4_loss = \
        torch.mean(BCE_loss_full(pred_mask4, mask4) * mask4_balance)
      seg_loss = mask1_loss + mask2_loss + mask3_loss + mask4_loss

      cls_loss = CE_loss(pred_logit, cls)
      loss = seg_loss + cls_loss

      loss.backward()
      optimizer.step()

      # localization accuracy
      binary_mask1 = (pred_mask1 > 0.4).cpu().numpy().flatten()
      mask1_flatten = mask1.cpu().numpy().flatten()
      batch_f1 = f1_score(mask1_flatten, binary_mask1, zero_division=1)
      f1_sum += batch_f1
      batch_count += 1

      _, binary_cls = torch.max(pred_logit, 1)
      cls_correct += (binary_cls == cls).sum().item()
      cls_total += int(torch.ones_like(cls).sum().item())

      seg_loss_sum += seg_loss.item()
      cls_loss_sum += cls_loss.item()

      if batch_id % 10 == 9:
        mean_f1 = f1_sum / batch_count
        logging.info(
          "[Epoch {0}, Batch {1}] F1 Score: {2:.4f}, seg_loss: {3:.4f}; "
          "Classification Accuracy: [{4}/{5}] {6:.2f}%, cls_loss: {7:.4f}"
          .format(epoch + 1,
                  batch_id + 1,
                  mean_f1,
                  seg_loss_sum / 10,
                  cls_correct,
                  cls_total,
                  cls_correct / cls_total * 100,
                  cls_loss_sum / 10))
        f1_sum, batch_count = 0, 0
        seg_loss_sum, cls_loss_sum, cls_correct, cls_total = 0, 0, 0, 0

    checkpoint = {"epoch": epoch + 1,
                  "FENet": FENet.state_dict(),
                  "SegNet": SegNet.state_dict(),
                  "ClsNet": ClsNet.state_dict(),
                  "optimizer": optimizer.state_dict()}

    if (epoch + 1) % 5 == 0:
      torch.save(checkpoint,
                  "{0}/{1}_{2}.pth".format(FENet_dir, FENet_name, epoch + 1))
      torch.save(checkpoint,
                  "{0}/{1}_{2}.pth".format(SegNet_dir, SegNet_name, epoch + 1))
      torch.save(checkpoint,
                  "{0}/{1}_{2}.pth".format(ClsNet_dir, ClsNet_name, epoch + 1))
      logging.info("Saved checkpoint at epoch {0}".format(epoch + 1))

    torch.save(checkpoint, "{0}/{1}_last.pth".format(FENet_dir, FENet_name))
    torch.save(checkpoint, "{0}/{1}_last.pth".format(SegNet_dir, SegNet_name))
    torch.save(checkpoint, "{0}/{1}_last.pth".format(ClsNet_dir, ClsNet_name))

    current_score = validation(FENet, SegNet, ClsNet, args, epoch + 1)
    logging.info(
      "Epoch {0}: current_score: {1:.4f}".format(epoch + 1, current_score))
    if current_score >= previous_score:
      previous_score = current_score
      best_epoch = epoch + 1

      torch.save(checkpoint, "{0}/{1}_best.pth".format(FENet_dir, FENet_name))
      torch.save(checkpoint, "{0}/{1}_best.pth".format(SegNet_dir, SegNet_name))
      torch.save(checkpoint, "{0}/{1}_best.pth".format(ClsNet_dir, ClsNet_name))
      logging.info(
        "*** New best model at epoch {0}, score={1:.4f} ***".format(
          best_epoch, previous_score))

    epoch_list.append(epoch + 1)
    loss_list.append(loss.item())
    score_list.append(current_score)

  logging.info(
    "Training finished. Best epoch: {0}, best score: {1:.4f}".format(
      best_epoch, previous_score))

  # # ====== 可视化多专家路由和多样性 ======
  # div_arr = np.array(diversity_history)  # shape: [steps, 4]
  # for i, name in enumerate(["getmask4", "getmask3", "getmask2", "getmask1"]):
  #     plt.plot(div_arr[:, i], label=name)
  # plt.legend()
  # plt.title("Expert Diversity (Variance) at Different Scales")
  # plt.xlabel("Iteration")
  # plt.ylabel("Variance")
  # plt.grid()
  # plt.show()

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

def validation(FENet, SegNet, ClsNet, args, epoch):
  val_data_loader = DataLoader(
      ValData(args),
      batch_size=args['val_bs'],
      shuffle=False,
      num_workers=8
  )
  all_mask_pred = []
  all_mask_gt = []
  cls_correct, cls_total = 0, 0

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
      pred_masks, feats = SegNet(feat)
      pred_mask1, pred_mask2, pred_mask3, pred_mask4 = pred_masks
      feat1, feat2, feat3, feat4 = feats

      ClsNet.eval()
      pred_logit = ClsNet(feats)

    # 保证 pred_mask1 和 mask 对齐
    if pred_mask1.shape != mask.shape:
      pred_mask1 = F.interpolate(
          pred_mask1,
          size=mask.shape[-2:],
          mode='bilinear',
          align_corners=True
      )

    # seg F1 统计
    binary_mask = (pred_mask1 > 0.5).cpu().numpy().astype(np.uint8)
    gt_mask = mask.cpu().numpy().astype(np.uint8)

    all_mask_pred.append(binary_mask.flatten())
    all_mask_gt.append(gt_mask.flatten())

    # 分类准确率统计
    sm = nn.Softmax(dim=1)
    pred_logit = sm(pred_logit)
    _, binary_cls = torch.max(pred_logit, 1)
    cls_correct += (binary_cls == cls).sum().item()
    cls_total += int(torch.ones_like(cls).sum().item())

    # ---------- 固定图像可视化部分 ----------
    # 这里的逻辑保持你原来的“每 100 个 batch 保存一次”
    # 因为 shuffle=False，所以每个 epoch 保存的是同一批图
    if batch_id % 100 == 0:
      # binary_mask: numpy, 形状 [B, 1, H, W] 或 [1, H, W]
      tensor_mask = torch.from_numpy(binary_mask).float()
      if tensor_mask.ndim == 2:
        tensor_mask = tensor_mask.unsqueeze(0)  # shape: (1, H, W)

      # 用 epoch 作为子目录名，比如 "epoch_001"
      epoch_folder = f"epoch_{epoch:03d}"

      # name 可能是 list 或单个字符串，统一加上 epoch 前缀
      if isinstance(name, (list, tuple)):
        # 例如原来 name = ["img1", "img2"]
        # 现在变成 ["epoch_001/img1", "epoch_001/img2"]
        name_with_epoch = [f"{epoch_folder}/{n}" for n in name]
      else:
        # 原来 name = "img1"，现在 = "epoch_001/img1"
        name_with_epoch = f"{epoch_folder}/{name}"

      # 仍然复用你原来的 save_image 接口：
      # 它内部如果是用 os.path.join(output_dir, subdir, f"{name}.png")
      # 那么传 "epoch_001/img1" 就会自动生成 mask/epoch_001/img1.png
      save_image(tensor_mask, name_with_epoch, 'mask')
    # ---------- 可视化部分结束 ----------

  all_mask_pred = np.concatenate(all_mask_pred)
  all_mask_gt = np.concatenate(all_mask_gt)
  seg_f1 = f1_score(all_mask_gt, all_mask_pred, zero_division=1)

  cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
  print(f"Validation: Seg F1={seg_f1:.4f}, Cls Acc={cls_acc:.4f}")
  return (seg_f1 + cls_acc) / 2

if __name__ == '__main__':
  args = get_pscc_args()
  train(args)
