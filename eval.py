from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from lossses import dice_coeff
from predict import target_to_image

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    target_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', position=0, leave=True) as pbar:
        for batch in loader:
            imgs, true_targets = batch['image'], batch['target']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_targets = true_targets.to(device=device, dtype=target_type)

            with torch.no_grad():
                target_pred = net(imgs)

            loss = nn.MSELoss()
            tot += loss(target_pred, true_targets).item()
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
    return tot / n_val

def eval_gen_net(net, loader, device, out_dir):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    target_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', position=0, leave=True) as pbar:
        for batch in loader:
            imgs, true_targets, indices = batch['image'], batch['target'], batch['idx']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_targets = true_targets.to(device=device, dtype=target_type)

            with torch.no_grad():
                target_pred = net(imgs)

            loss = nn.MSELoss()
            tot += loss(target_pred, true_targets).item()

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
    return tot / n_val
