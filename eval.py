from pathlib import Path
from math import exp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from losses import dice_coeff
from predict import target_to_image

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def SSIM(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    print(img1.shape)
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def PSNR(img1, img2):
    mse = torch.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    target_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot_MSE = 0
    tot_SSIM = 0
    tot_PSNR = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', position=0, leave=True) as pbar:
        for batch in loader:
            imgs, true_targets, grays = batch['image'], batch['target'], batch['gray']
            imgs = imgs.to(device=device, dtype=torch.float32)
            grays = grays.to(device=device, dtype=torch.float32)
            true_targets = true_targets.to(device=device, dtype=target_type)

            with torch.no_grad():
                target_pred = net(imgs, grays)

            loss = nn.MSELoss()
            tot_MSE += loss(target_pred, true_targets).item()
            tot_SSIM += SSIM(target_pred, true_targets).item()
            tot_PSNR += PSNR(target_pred, true_targets)
            """
            if net.n_classes > 1:
                tot += nn.cross_entropy(target_pred, true_targets).item()
            else:
                pred = torch.sigmoid(target_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_targets).item()
            """
            pbar.update()

    net.train()
    return tot_MSE / n_val, tot_SSIM / n_val, tot_PSNR / n_val

def eval_gen_net(net, loader, device, out_dir):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    target_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot_MSE = 0
    tot_SSIM = 0
    tot_PSNR = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', position=0, leave=True) as pbar:
        for batch in loader:
            imgs, true_targets, grays, indices = batch['image'], batch['target'], batch['gray'], batch['idx']
            imgs = imgs.to(device=device, dtype=torch.float32)
            grays = grays.to(device=device, dtype=torch.float32)
            true_targets = true_targets.to(device=device, dtype=target_type)

            with torch.no_grad():
                target_pred = net(imgs, grays)

            loss = nn.MSELoss()
            tot_MSE += loss(target_pred, true_targets).item()
            tot_SSIM += SSIM(target_pred, true_targets).item()
            tot_PSNR += PSNR(target_pred, true_targets)

            for index, single_target_pred in enumerate(target_pred):
                image = target_to_image(single_target_pred.squeeze().cpu().numpy())
                image.save(Path(out_dir,indices[index]+"_generated.png"))
            """
            if net.n_classes > 1:
                tot += nn.cross_entropy(target_pred, true_targets).item()
            else:
                pred = torch.sigmoid(target_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_targets).item()
            """
            pbar.update()

    net.train()
    return tot_MSE / n_val, tot_SSIM / n_val, tot_PSNR / n_val
