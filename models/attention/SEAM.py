import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAM(nn.Module):
  def __init__(self, in_channels, num_classes=2, reduction_ratio=16):
    super(SEAM, self).__init__()

    self.in_channels = in_channels
    self.num_classes = num_classes
    self.reduction_ratio = reduction_ratio

    # channel-wise attention part
    self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
    self.relu = nn.ReLU(inplace=True)
    self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
    self.sigmoid = nn.Sigmoid()

    # spatial attention part
    self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
    self.softmax = nn.Softmax(dim=-1)

    # Final convolution for class-specific CAM generation
    self.fc8 = nn.Conv2d(in_channels, self.num_classes, 1, bias=False)

  def forward(self, x):
    N, C, H, W = x.size()

    # Channel-wise Attention
    avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
    max_out, _ = torch.max(x, dim=2, keepdim=True)
    max_out, _ = torch.max(max_out, dim=3, keepdim=True)
    channel_att = self.fc2(self.relu(self.fc1(avg_out))) + self.fc2(self.relu(self.fc1(max_out)))
    channel_att = self.sigmoid(channel_att)
    x = x * channel_att

    # CAM generation from the feature map
    cam = self.fc8(x)
    n, c, h, w = cam.size()

    with torch.no_grad():
      cam_d = F.relu(cam.detach())
      cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
      cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
      cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
      cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
      cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0

    # Spatial Attention
    spatial_att = self.conv1(x)
    spatial_att = self.softmax(spatial_att.view(x.size(0), -1))
    spatial_att = spatial_att.view(x.size(0), 1, x.size(2), x.size(3))
    x = x * spatial_att

    # Return the adjusted CAM
    return cam_d_norm, x

def test_SEAM():
  B, C, H, W = 4, 64, 32, 32
  input_tensor = torch.randn(B, C, H, W)

  seam = SEAM(in_channels=C, reduction_ratio=16)

  output_mask, x = seam(input_tensor)

  print("Input shape:", input_tensor.shape)
  print("Output shape:", output_mask.shape)
  print("x shape:", x.shape)


if __name__ == "__main__":
  test_SEAM()