import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seg_hrnet_config import get_hrnet_cfg
from models.attention.CBAM import CBAM
from models.attention.SE import SE
from models.attention.SCCM import SCCM
from models.attention.BiFormer import BiLevelRoutingAttention

DEBUG = False

class MoEAttention(nn.Module):
  """
  dim (int): 输入特征通道数 C
  num_experts (int): 专家子网数量 E
  hidden_dim (int, optional): 专家子网内部隐层维度
  return_routing (bool): 是否在 forward 时返回 routing 权重
  temp (float): softmax 温度系数，默认为 1.0
  """
  def __init__(self, dim, num_experts, hidden_dim,
               return_routing=False, temp=1.0, norm_type='bn'):
    super().__init__()
    self.num_experts = num_experts
    self.return_routing = return_routing
    self.temp = temp
    self.diversity = 0

    # Gating: 两层MLP
    self.gate = nn.Sequential(
        nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        nn.BatchNorm2d(dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim, num_experts, 1, bias=True))

    # Expert: 每个专家输出 1 通道
    norm_cls = nn.BatchNorm2d if norm_type == 'bn' \
      else lambda c: nn.GroupNorm(8, c)
    self.experts = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=False),
        norm_cls(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
      ) for _ in range(num_experts)])

    # 负载均衡
    self.register_buffer('routing_stats', torch.zeros(num_experts))

    self._init_weights()

  def _init_weights(self):
    # 合理初始化, 防止某一专家被独占
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x: torch.Tensor):
    """
    x: [B, C, H, W]
    mask: [B, 1, H, W]   routing: [B, E, H, W]
    """
    B, C, H, W = x.shape
    assert C == self.gate[0].in_channels, \
      f"Expected input channel={self.gate[0].in_channels}, got {C}"

    # 1. routing
    gate_logits = self.gate(x) / self.temp
    noise = torch.randn_like(gate_logits) * 0.01
    gate_logits = gate_logits + noise
    routing = F.softmax(gate_logits, dim=1)
    with torch.no_grad():
      self.routing_stats.copy_(routing.mean(dim=(0, 2, 3)).detach())

    # 2. experts_outs
    # experts_outs -> [B, E, 1, H, W]
    experts_outs = torch.stack(
      [expert(x).squeeze(1) for expert in self.experts], dim=1)

    # 3. mask: [B, 1, H, W]
    mask = (experts_outs * routing).sum(dim=1, keepdim=True)

    # 4. MoE status
    self.diversity = experts_outs.var(dim=1).mean().item()
    if DEBUG:
      print("temp:", self.temp)
      print("diversity calc in forward:", self.diversity)
      print("experts_outs mean/std:", \
        experts_outs.mean().item(), experts_outs.std().item())
      print("experts_outs[0, :, 0, 0]", experts_outs[0, :, 0, 0])

    if self.return_routing:
      return mask, routing
    else:
      return mask

  def get_diversity(self):
    return self.diversity

  def get_routing_stats(self):
    return self.routing_stats.cpu().numpy()

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
    # self.getmask4 = SE(feat4_num, 1)
    # self.getmask3 = SE(feat3_num, 2)
    # self.getmask2 = SE(feat2_num, 2)
    # self.getmask1 = SE(feat1_num, 4)
    # self.getmask4 = CBAM(feat4_num, 1)
    # self.getmask3 = CBAM(feat3_num, 2)
    # self.getmask2 = CBAM(feat2_num, 2)
    # self.getmask1 = CBAM(feat1_num, 4)
    self.getmask4 = MoEAttention(feat4_num, 3, 256)
    self.getmask3 = MoEAttention(feat3_num, 3, 256)
    self.getmask2 = MoEAttention(feat2_num, 3, 256)
    self.getmask1 = MoEAttention(feat1_num, 3, 256)
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
    mask1 = self.getmask1(s1)

    return mask1, mask2, mask3, mask4
