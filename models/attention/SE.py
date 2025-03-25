import torch
from torch import nn

class SE(nn.Module):
  def __init__(self, channel, reduction=16):
    super(SE, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channel // reduction, channel, bias=False),
      nn.Sigmoid()
    )
    self.getmask = nn.Sequential(
      nn.Conv2d(in_channels=channel, out_channels=16,
                kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=16, out_channels=1,
                kernel_size=3, stride=1, padding=1))

  def forward(self, x):
    b, c, _, _ = x.size()
    result = self.avg_pool(x).view(b, c)
    result = self.fc(result).view(b, c, 1, 1)
    result = x * result.expand_as(x)
    # get mask
    mask = torch.sigmoid(self.getmask(result.clone()))
    
    return mask, result


if __name__ == '__main__':
  torch.manual_seed(seed=20200910)
  data_in = torch.randn(8,32,300,300)
  se = SE(32)
  mask, result = se(data_in)
  print(data_in.shape)  # torch.Size([8, 32, 300, 300])
  print(mask.shape)  # torch.Size([8, 32, 300, 300])
  print(result.shape)  # torch.Size([8, 32, 300, 300])
