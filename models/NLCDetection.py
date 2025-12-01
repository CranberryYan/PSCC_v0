import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seg_hrnet_config import get_hrnet_cfg
from models.attention.SE import SE
from models.attention.SCCM import SCCM
from models.attention.CBAM import CBAM
from models.attention.BiFormer import BiLevelRoutingAttention

DEBUG = False

class ConvMaskBlock(nn.Module):
    """
    无注意力的对照模块: 
      - 不使用 ChannelAttention / SpatialAttention
      - 仅通过若干卷积提取特征 result
      - 再用一个小的卷积 head 生成 mask
    接口与 CBAM 一致: forward(x) -> (mask, result)
    """
    def __init__(self, in_channels, mid_channels=None):
        super(ConvMaskBlock, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels

        # 一个简单的 conv block，模拟“特征增强”，但没有显式注意力
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # 和你原来的 getmask 类似的小 head，只是输入改为 result
        self.getmask = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # 不做 x * attention，仅做普通卷积特征变换
        result = self.body(x)
        mask = torch.sigmoid(self.getmask(result))  # [B,1,H,W]

        return mask, result


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

import torch
import torch.nn as nn
import torch.nn.functional as F

# 依赖：你工程里已实现
# from .xxx import GatingGLU, ExpertWithAttention


class MoEAttention(nn.Module):
    def __init__(
        self,
        dim,
        attn,
        num_experts,
        hidden_dim,
        return_routing: bool = False,
        temp: float = 1.0,
        norm_type: str = "bn",
        topk: int = 32,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.return_routing = return_routing
        self.temp = max(1e-4, float(temp))
        self.topk = int(topk)

        self.diversity = 0.0

        # gate: [B,C,H,W] -> [B,E,H,W]
        self.gate = GatingGLU(dim, self.num_experts)

        # experts: 每个 expert 内部是一个 attention block
        self.experts = nn.ModuleList([ExpertWithAttention(dim, attn) for _ in range(self.num_experts)])

        # 负载均衡统计: 每个 expert 的平均 routing 概率
        self.register_buffer("routing_stats", torch.zeros(self.num_experts))

        # ===== debug 缓存(不改变主网络返回值也能拿到专家输出)=====
        self.debug_mode = False
        self._last_debug = None

        self.init_weights()

    # ---------------- debug API ----------------
    def enable_debug(self, flag: bool = True):
        """开启后，每次 forward 都会缓存 expert_masks/expert_res/routing 等到 self._last_debug"""
        self.debug_mode = bool(flag)

    def get_last_debug(self):
        """返回最近一次 forward 缓存的 debug dict(可能为 None)"""
        return self._last_debug

    # ---------------- init ----------------
    def init_weights(self):
        # 合理初始化, 防止某一专家被独占
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ---------------- routing ----------------
    def _build_routing(self, x: torch.Tensor):
        """
        返回:
          routing: [B,E,H,W]
          gate_logits: [B,E,H,W]
        """
        gate_logits = self.gate(x) / self.temp  # [B,E,H,W]

        if self.training:
            noise = torch.randn_like(gate_logits) * 0.01
            gate_logits = gate_logits + noise
            routing = F.softmax(gate_logits, dim=1)  # dense
        else:
            k = min(self.topk, self.num_experts)
            topk_val, topk_idx = torch.topk(gate_logits, k, dim=1)  # [B,k,H,W]
            topk_probs = F.softmax(topk_val, dim=1)                  # [B,k,H,W]
            routing = torch.zeros_like(gate_logits)                  # [B,E,H,W]
            routing.scatter_(1, topk_idx, topk_probs)

        return routing, gate_logits

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor, return_expert_maps: bool = False):
        """
        x: [B,C,H,W]
        默认返回:
          mask: [B,1,H,W]
          res : [B,C,H,W]
          (可选) routing: [B,E,H,W] 由 self.return_routing 控制

        若 return_expert_maps=True，则额外返回 debug dict:
          expert_masks: [B,E,H,W]
          expert_res  : [B,E,C,H,W]
          routing     : [B,E,H,W]
          gate_logits : [B,E,H,W]
          diversity   : float
        """
        B, C, H, W = x.shape

        # 1) routing
        routing, gate_logits = self._build_routing(x)

        # 2) 统计每个 expert 的平均使用率: [E]
        with torch.no_grad():
            self.routing_stats.copy_(routing.mean(dim=(0, 2, 3)).detach())

        # 3) experts 前向：每个 expert(x) -> (mask_e, res_e)
        expert_outputs = [expert(x) for expert in self.experts]
        masks_list = [m for (m, r) in expert_outputs]  # each: [B,1,H,W]
        res_list = [r for (m, r) in expert_outputs]    # each: [B,C,H,W]

        expert_masks = torch.stack([m.squeeze(1) for m in masks_list], dim=1)  # [B,E,H,W]
        expert_res = torch.stack(res_list, dim=1)                               # [B,E,C,H,W]

        # 4) routing 加权融合
        mask = (expert_masks * routing).sum(dim=1, keepdim=True)  # [B,1,H,W]
        mask = torch.clamp(mask, 0.0, 1.0)

        routing_expanded = routing.unsqueeze(2)                    # [B,E,1,H,W]
        res = (expert_res * routing_expanded).sum(dim=1)           # [B,C,H,W]

        # 5) 多样性(基于 expert_masks)
        self.diversity = expert_masks.var(dim=1).mean().item()

        # 6) debug 缓存：不改变原有返回也能取到
        if self.debug_mode or return_expert_maps:
            debug = {
                "expert_masks": expert_masks.detach(),   # [B,E,H,W]
                "expert_res": expert_res.detach(),       # [B,E,C,H,W]
                "routing": routing.detach(),             # [B,E,H,W]
                "gate_logits": gate_logits.detach(),     # [B,E,H,W]
                "diversity": float(self.diversity),
            }
            self._last_debug = debug
        else:
            debug = None

        # 7) 返回
        if return_expert_maps:
            if self.return_routing:
                return mask, res, routing, debug
            else:
                return mask, res, debug

        if self.return_routing:
            return mask, res, routing
        else:
            return mask, res

    def get_diversity(self):
        return self.diversity

    def get_routing_stats(self):
        return self.routing_stats.detach().cpu().numpy()


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
    # self.getmask4 = ConvMaskBlock(feat4_num, 1)
    # self.getmask3 = ConvMaskBlock(feat3_num, 2)
    # self.getmask2 = ConvMaskBlock(feat2_num, 2)
    # self.getmask1 = ConvMaskBlock(feat1_num, 4)
    self.getmask4 = MoEAttention(feat4_num, CBAM, 2, self.crop_size)
    self.getmask3 = MoEAttention(feat3_num, CBAM, 2, self.crop_size)
    self.getmask2 = MoEAttention(feat2_num, CBAM, 2, self.crop_size)
    self.getmask1 = MoEAttention(feat1_num, CBAM, 2, self.crop_size)
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
    inputs:
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
