import torch
import torch.nn as nn
import torch.nn.functional as F

class SCCM(nn.Module):
  def __init__(self, in_channels, reduce_scale):
    super(SCCM, self).__init__()

    self.r = reduce_scale

    # input channel number
    self.ic = in_channels * self.r * self.r

    # middle channel number
    self.mc = self.ic

    self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                        kernel_size=1, stride=1, padding=0)

    self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                            kernel_size=1, stride=1, padding=0)
    self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                          kernel_size=1, stride=1, padding=0)

    self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=1, stride=1, padding=0)

    self.W_c = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=1, stride=1, padding=0)

    self.gamma_s = nn.Parameter(torch.ones(1))

    self.gamma_c = nn.Parameter(torch.ones(1))

    self.getmask = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=1,
                  kernel_size=3, stride=1, padding=1))

  def forward(self, x):
    """
    inputs :
      x : input feature maps( B X C X H X W)
    value :
      f: B X (HxW) X (HxW)
      ic: intermediate channels
      z: feature maps( B X C X H X W)
    output:
      mask: feature maps( B X 1 X H X W)
    """

    b, c, h, w = x.shape

    x1 = x.reshape(b, self.ic, h // self.r, w // self.r)

    # g x
    g_x = self.g(x1).view(b, self.ic, -1)
    g_x = g_x.permute(0, 2, 1)

    # theta
    theta_x = self.theta(x1).view(b, self.mc, -1)

    theta_x_s = theta_x.permute(0, 2, 1)
    theta_x_c = theta_x

    # phi x
    phi_x = self.phi(x1).view(b, self.mc, -1)

    phi_x_s = phi_x
    phi_x_c = phi_x.permute(0, 2, 1)

    # non-local attention
    f_s = torch.matmul(theta_x_s, phi_x_s)
    f_s_div = F.softmax(f_s, dim=-1)

    f_c = torch.matmul(theta_x_c, phi_x_c)
    f_c_div = F.softmax(f_c, dim=-1)

    # get y_s
    y_s = torch.matmul(f_s_div, g_x)
    y_s = y_s.permute(0, 2, 1).contiguous()
    y_s = y_s.view(b, c, h, w)

    # get y_c
    y_c = torch.matmul(g_x, f_c_div)
    y_c = y_c.view(b, c, h, w)

    # get z
    z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)

    # get mask
    mask = torch.sigmoid(self.getmask(z.clone()))

    return mask, z

def test_SCCM():
  B, C, H, W = 4, 64, 32, 32
  input_tensor = torch.randn(B, C, H, W)

  sccm = SCCM(in_channels=C, reduce_scale=16)

  output_mask, z = sccm(input_tensor)

  print("Input shape:", input_tensor.shape)
  print("Output shape:", output_mask.shape)


if __name__ == "__main__":
  test_SCCM()