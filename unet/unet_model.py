""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn

from .unet_parts import *


######################################################################################
# Padding Tensors
######################################################################################

def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


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

        flag = 0
        if x.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            x = avg(x)
            if self.self_attention:
                gray = avg(gray)
            flag = 1
            # pass
        x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x)

        if self.self_attention:
            gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)

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
        output = self.outc(x)

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        
        if self.self_attention:
            gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            if self.self_attention:
                gray = F.upsample(gray, scale_factor=2, mode='bilinear')

        return output