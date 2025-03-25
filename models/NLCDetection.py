import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seg_hrnet_config import get_hrnet_cfg
from models.attention.CBAM import CBAM
from models.attention.SE import SE
from models.attention.SCCM import SCCM


class NLCDetection(nn.Module):
  def __init__(self, args):
    super(NLCDetection, self).__init__()

    self.crop_size = args['crop_size']

    FENet_cfg = get_hrnet_cfg()

    num_channels = FENet_cfg['STAGE4']['NUM_CHANNELS']

    feat1_num, feat2_num, feat3_num, feat4_num = num_channels

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

    mask4, z4 = self.getmask4(s4)
    mask4U = F.interpolate(
      mask4, size=s3.size()[2:], mode='bilinear', align_corners=True)

    s3 = s3 * mask4U
    mask3, z3 = self.getmask3(s3)
    mask3U = F.interpolate(
      mask3, size=s2.size()[2:], mode='bilinear', align_corners=True)

    s2 = s2 * mask3U
    mask2, z2 = self.getmask2(s2)
    mask2U = F.interpolate(
      mask2, size=s1.size()[2:], mode='bilinear', align_corners=True)

    s1 = s1 * mask2U
    mask1, z1 = self.getmask1(s1)

    return mask1, mask2, mask3, mask4
