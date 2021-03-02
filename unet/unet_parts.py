""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, self_attention=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.self_attention = self_attention

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))
        
        self.second_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, grey=None):
        x1 = self.first_conv(x)
        x1 = x1*grey if self.self_attention else x1
        x2 = self.second_conv(x1)
        return x2

class TripleResidual(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.first_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.first_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.first_conv_5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(mid_channels*3, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
  
    def forward(self, x, res_1):
        x1= self.first_conv_1(x)
        x3= self.first_conv_3(x)
        x5= self.first_conv_5(x)
        y = torch.cat([x1,x3,x5], dim=1)
        y = self.second_conv(y)
        return y+res_1

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, self_attention=False):
        super().__init__()
        self.maxpool= nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, self_attention=self_attention)
        
    def forward(self, x, grey=None):
        x = self.maxpool(x)
        return self.conv(x, grey)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, self_attention=False):
        super().__init__()
        self.self_attention = self_attention
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.deconv = TripleResidual(in_channels, out_channels)
            self.conv = TripleResidual(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.deconv = DoubleConv(in_channels, out_channels, in_channels)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2, grey=None):
        x1 = self.up(x1)
        """
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        """
        x2 = x2*grey if self.self_attention else x2
        x1 = self.deconv(x1,x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x,x2)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)