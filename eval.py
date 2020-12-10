import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    target_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_targets = batch['image'], batch['target']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_targets = true_targets.to(device=device, dtype=target_type)

            with torch.no_grad():
                target_pred = net(imgs)

            tot += nn.L1Loss(target_pred, true_targets).item()
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
