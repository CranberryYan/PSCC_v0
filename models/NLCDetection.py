import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seg_hrnet_config import get_hrnet_cfg
from models.attention.CBAM import CBAM
from models.attention.SE import SE
from models.attention.SCCM import SCCM
from models.attention.BiFormer import BiLevelRoutingAttention


class NLCDetection(nn.Module):
  def __init__(self, args):
    super(NLCDetection, self).__init__()

    self.crop_size = args['crop_size']

    FENet_cfg = get_hrnet_cfg()

    num_channels = FENet_cfg['STAGE4']['NUM_CHANNELS']

    feat1_num, feat2_num, feat3_num, feat4_num = num_channels
    # print(num_channels)
    # self.getmask4 = SCCM(feat4_num, 1)
    # self.getmask3 = SCCM(feat3_num, 2)
    # self.getmask2 = SCCM(feat2_num, 2)
    # self.getmask1 = SCCM(feat1_num, 4)
    self.getmask4 = SE(feat4_num, 1)
    self.getmask3 = SE(feat3_num, 2)
    self.getmask2 = SE(feat2_num, 2)
    self.getmask1 = SE(feat1_num, 4)
    # self.getmask4 = CBAM(feat4_num, 1)
    # self.getmask3 = CBAM(feat3_num, 2)
    # self.getmask2 = CBAM(feat2_num, 2)
    # self.getmask1 = CBAM(feat1_num, 4)
    # self.getmask1 = BiLevelRoutingAttention(dim=feat1_num,
    #                                         n_win=7,
    #                                         num_heads=2,
    #                                         qk_dim=feat1_num,
    #                                         qk_scale=None,
    #                                         kv_per_win=4,
    #                                         kv_downsample_ratio=2,
    #                                         kv_downsample_mode='avgpool',
    #                                         topk=3,
    #                                         param_attention='qkvo',
    #                                         param_routing=False,
    #                                         diff_routing=False,
    #                                         soft_routing=False,
    #                                         side_dwconv=3,
    #                                         auto_pad=True)
    # self.getmask2 = BiLevelRoutingAttention(dim=feat2_num,
    #                                         n_win=7,
    #                                         num_heads=2,
    #                                         qk_dim=feat2_num,
    #                                         qk_scale=None,
    #                                         kv_per_win=4,
    #                                         kv_downsample_ratio=2,
    #                                         kv_downsample_mode='avgpool',
    #                                         topk=3,
    #                                         param_attention='qkvo',
    #                                         param_routing=False,
    #                                         diff_routing=False,
    #                                         soft_routing=False,
    #                                         side_dwconv=3,
    #                                         auto_pad=True)
    # self.getmask3 = BiLevelRoutingAttention(dim=feat3_num,
    #                                         n_win=7,
    #                                         num_heads=2,
    #                                         qk_dim=feat3_num,
    #                                         qk_scale=None,
    #                                         kv_per_win=4,
    #                                         kv_downsample_ratio=2,
    #                                         kv_downsample_mode='avgpool',
    #                                         topk=3,
    #                                         param_attention='qkvo',
    #                                         param_routing=False,
    #                                         diff_routing=False,
    #                                         soft_routing=False,
    #                                         side_dwconv=3,
    #                                         auto_pad=True)
    # self.getmask4 = BiLevelRoutingAttention(dim=feat4_num,
    #                                         n_win=7,
    #                                         num_heads=2,
    #                                         qk_dim=feat4_num,
    #                                         qk_scale=None,
    #                                         kv_per_win=4,
    #                                         kv_downsample_ratio=2,
    #                                         kv_downsample_mode='avgpool',
    #                                         topk=3,
    #                                         param_attention='qkvo',
    #                                         param_routing=False,
    #                                         diff_routing=False,
    #                                         soft_routing=False,
    #                                         side_dwconv=3,
    #                                         auto_pad=True)

  def forward(self, feat):
    """
    inputs :
        feat : a list contains features from s1, s2, s3, s4
    output:
        mask1: output mask ( B X 1 X H X W)
        pred_cls: output cls (B X 4)
    """
    s1, s2, s3, s4 = feat

    if s1.shape[2:] == self.crop_size:
      pass
    else:
      s1 = F.interpolate(
        s1, size=self.crop_size,
        mode='bilinear', align_corners=True)
      s2 = F.interpolate(
        s2,size=[i // 2 for i in self.crop_size],
        mode='bilinear', align_corners=True)
      s3 = F.interpolate(
        s3, size=[i // 4 for i in self.crop_size],
        mode='bilinear', align_corners=True)
      s4 = F.interpolate(
        s4, size=[i // 8 for i in self.crop_size],
        mode='bilinear', align_corners=True)

    mask4 = self.getmask4(s4)
    mask4U = F.interpolate(
      mask4, size=s3.size()[2:], mode='bilinear', align_corners=True)

    s3 = s3 * mask4U
    mask3 = self.getmask3(s3)
    mask3U = F.interpolate(
      mask3, size=s2.size()[2:], mode='bilinear', align_corners=True)

    s2 = s2 * mask3U
    mask2 = self.getmask2(s2)
    mask2U = F.interpolate(
      mask2, size=s1.size()[2:], mode='bilinear', align_corners=True)

    s1 = s1 * mask2U
    mask1  = self.getmask1(s1)

    return mask1, mask2, mask3, mask4
