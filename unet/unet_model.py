""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, self_attention=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.self_attention = self_attention
        
        if self.self_attention:
            n_channels+=1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 1
        self.down4 = Down(512, 1024 // factor, self_attention=self.self_attention)
        self.up1 = Up(1024, 512 // factor, bilinear, self_attention=self.self_attention)
        self.up2 = Up(512, 256 // factor, bilinear, self_attention=self.self_attention)
        self.up3 = Up(256, 128 // factor, bilinear, self_attention=self.self_attention)
        self.up4 = Up(128, 64, bilinear, self_attention=self.self_attention)
        self.outc = OutConv(64, n_classes)

        #attention
        self.downsample_1 = nn.MaxPool2d(2)
        self.downsample_2 = nn.MaxPool2d(2)
        self.downsample_3 = nn.MaxPool2d(2)
        self.downsample_4 = nn.MaxPool2d(2)

    def forward(self, x, gray=None):
        if self.self_attention:
            x = torch.cat((x, gray), 1)
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4, gray_5) if self.self_attention else self.down4(x4)
        x = self.up1(x5, x4, gray_4) if self.self_attention else self.up1(x5, x4)
        x = self.up2(x, x3, gray_3) if self.self_attention else self.up2(x, x3)
        x = self.up3(x, x2, gray_2) if self.self_attention else self.up3(x, x2)
        x = self.up4(x, x1, gray) if self.self_attention else self.up4(x, x1)
        logits = self.outc(x)
        return logits