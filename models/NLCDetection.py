import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seg_hrnet_config import get_hrnet_cfg
from models.attention.SE import SE
from models.attention.SCCM import SCCM
from models.attention.CBAM import CBAM
from models.attention.BiFormer import BiLevelRoutingAttention

DEBUG = False

class GatingGLU(nn.Module):
  def __init__(self, in_channels, num_experts):
    super().__init__()
    self.conv_A = nn.Conv2d(in_channels, num_experts, 1)
    self.conv_B = nn.Conv2d(in_channels, num_experts, 1)
  def forward(self, x):
    return self.conv_A(x) * torch.sigmoid(self.conv_B(x))

class ExpertWithAttention(nn.Module):
  def __init__(self, in_channels, attn):
    super().__init__()
    self.attn = attn(in_channels)

  def forward(self, x):
    mask, res = self.attn(x)
    return mask, res

class MoEAttention(nn.Module):
  def __init__(self, dim, attn, num_experts, hidden_dim,
               return_routing=False, temp=1.0, norm_type='bn', topk=5):
    super().__init__()
    self.num_experts = num_experts
    self.return_routing = return_routing
    self.temp = max(1e-4, float(temp))
    self.topk = topk
    self.diversity = 0

    self.gate = GatingGLU(dim, num_experts)

    # Experts: 每个expert内部是一个attn
    self.experts = nn.ModuleList([
      ExpertWithAttention(dim, attn)
      for _ in range(num_experts)])

    # 负载均衡统计: 每个 expert 的平均 routing 概率
    self.register_buffer('routing_stats', torch.zeros(num_experts))

    self.init_weights()

  def init_weights(self):
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
    返回:
      mask: [B, 1, H, W]
      res : [B, C, H, W]
      (可选) routing: [B, E, H, W]
    """
    B, C, H, W = x.shape

    # 1. gate logits: [B, E, H, W]
    gate_logits = self.gate(x) / self.temp

    # 1.1 训练: dense softmax + 噪声
    if self.training:
      noise = torch.randn_like(gate_logits) * 0.01
      gate_logits = gate_logits + noise
      routing = F.softmax(gate_logits, dim=1)
    # 1.2 测试: top-k sparse routing
    else:
      topk = min(self.topk, self.num_experts)
      topk_val, topk_idx = torch.topk(gate_logits, topk, dim=1)  # [B, k, H, W]
      topk_probs = F.softmax(topk_val, dim=1)                    # [B, k, H, W]
      routing = torch.zeros_like(gate_logits)                    # [B, E, H, W]
      routing.scatter_(1, topk_idx, topk_probs)

    # 2. 统计每个 expert 的平均使用率: [E]
    with torch.no_grad():
      self.routing_stats.copy_(routing.mean(dim=(0, 2, 3)).detach())

    # 3. 调用所有 experts
    #   每个 expert(x) -> (mask_e, res_e)
    #   mask_e: [B, 1, H, W], res_e: [B, C, H, W]
    expert_outputs = [expert(x) for expert in self.experts]
    masks_list = [m for (m, r) in expert_outputs]   # list of [B, 1, H, W]
    res_list   = [r for (m, r) in expert_outputs]   # list of [B, C, H, W]

    # 3.1 masks: [B, E, H, W]
    masks = torch.stack([m.squeeze(1) for m in masks_list], dim=1)

    # 3.2 res: [B, E, C, H, W]
    res_stack = torch.stack(res_list, dim=1)

    # 4. Routing 加权融合
    # 4.1 mask: [B, 1, H, W]
    mask = (masks * routing).sum(dim=1, keepdim=True)
    mask = torch.clamp(mask, 0.0, 1.0)

    # 4.2 res: [B, C, H, W]
    # routing: [B, E, H, W] -> [B, E, 1, H, W] 用于广播到 C 维
    routing_expanded = routing.unsqueeze(2)              # [B, E, 1, H, W]
    res = (res_stack * routing_expanded).sum(dim=1)      # [B, C, H, W]

    # 5. MoE 多样性指标(基于 mask 输出)
    self.diversity = masks.var(dim=1).mean().item()
    if DEBUG:
      print("temp:", self.temp)
      print("diversity calc in forward:", self.diversity)
      print("masks mean/std:",
            masks.mean().item(), masks.std().item())
      print("masks[0, :, 0, 0]", masks[0, :, 0, 0])

    # 6. 返回
    if self.return_routing:
      return mask, res, routing
    else:
      return mask, res

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
    self.getmask4 = MoEAttention(feat4_num, CBAM, 8, self.crop_size)
    self.getmask3 = MoEAttention(feat3_num, CBAM, 8, self.crop_size)
    self.getmask2 = MoEAttention(feat2_num, CBAM, 8, self.crop_size)
    self.getmask1 = MoEAttention(feat1_num, CBAM, 8, self.crop_size)
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

    mask4, feat4 = self.getmask4(s4)
    mask4U = F.interpolate(
      mask4, size=s3.size()[2:], mode='bilinear', align_corners=True)

    s3 = s3 * mask4U
    mask3, feat3 = self.getmask3(s3)
    mask3U = F.interpolate(
      mask3, size=s2.size()[2:], mode='bilinear', align_corners=True)

    s2 = s2 * mask3U
    mask2, feat2 = self.getmask2(s2)
    mask2U = F.interpolate(
      mask2, size=s1.size()[2:], mode='bilinear', align_corners=True)

    s1 = s1 * mask2U
    mask1, feat1 = self.getmask1(s1)

    return [mask1, mask2, mask3, mask4], [feat1, feat2, feat3, feat4]
