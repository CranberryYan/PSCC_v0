import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn

class ChannelAttention(nn.Module):
  def __init__(self, in_channels, ratio=16):
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

    self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
    max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
    out = avg_out + max_out
    return self.sigmoid(out)


class SpatialAttention(nn.Module):
  def __init__(self, kernel_size=7):
    super(SpatialAttention, self).__init__()

    assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
    padding = 3 if kernel_size == 7 else 1

    self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    x = torch.cat([avg_out, max_out], dim=1)
    x = self.conv1(x)
    return self.sigmoid(x)

class CBAM(nn.Module):
  def __init__(self, in_channels, ratio=16, kernel_size=7):
    super(CBAM, self).__init__()
    self.ca = ChannelAttention(in_channels, ratio)
    self.sa = SpatialAttention(kernel_size)

    self.getmask = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=1,
                  kernel_size=3, stride=1, padding=1))

  def forward(self, x):
    out = x * self.ca(x)
    result = out * self.sa(out)
    # get mask
    mask = torch.sigmoid(self.getmask(result.clone()))
    
    return mask, result


if __name__ == '__main__':
  print('testing ChannelAttention'.center(100,'-'))
  torch.manual_seed(seed=20200910)
  CA = ChannelAttention(32)
  data_in = torch.randn(8,32,300,300)
  data_out = CA(data_in)
  print(data_in.shape)  # torch.Size([8, 32, 300, 300])
  print(data_out.shape)  # torch.Size([8, 32, 1, 1])



if __name__ == '__main__':
  print('testing SpatialAttention'.center(100,'-'))
  torch.manual_seed(seed=20200910)
  SA = SpatialAttention(7)
  data_in = torch.randn(8,32,300,300)
  data_out = SA(data_in)
  print(data_in.shape)  # torch.Size([8, 32, 300, 300])
  print(data_out.shape)  # torch.Size([8, 1, 300, 300])


if __name__ == '__main__':
  print('testing CBAM'.center(100,'-'))
  torch.manual_seed(seed=20200910)
  cbam = CBAM(32, 16, 7)
  data_in = torch.randn(8,32,300,300)
  mask, result = cbam(data_in)
  print(data_in.shape)  # torch.Size([8, 32, 300, 300])
  print(mask.shape)  # torch.Size([8, 1, 300, 300])
  print(result.shape)  # torch.Size([8, 32, 300, 300])
