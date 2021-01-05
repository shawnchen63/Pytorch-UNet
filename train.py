"""
original code used for image segmentation
re-adapted for low light image enhancement
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from losses import PerceptualLoss
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

def train_net(net,
              device,
              dir_img,
              dir_target,
              dir_checkpoint,
              dir_log,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.2,
              save_cp=True,
              img_scale=1.0,
              augmented=True):

    dataset = BasicDataset(dir_img, dir_target, img_scale, augmented=augmented)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    indices = list(range(len(dataset)))

    train_idx, valid_idx = indices[n_val:], indices[:n_val]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(dir_log, comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    """
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    """
    criterion = torch.nn.MSELoss()
    content_loss = PerceptualLoss()
    content_loss.initialize(nn.MSELoss())

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_targets = batch['target']
                grays = batch['gray']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                target_type = torch.float32
                true_targets = true_targets.to(device=device, dtype=target_type)

                targets_pred = net(imgs,grays)
                loss = criterion(targets_pred, true_targets) + 0.2*content_loss.get_loss(targets_pred, true_targets)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (2 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_scores = eval_net(net, val_loader, device)
                    scheduler.step(val_scores[0])
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info(f'Validation MSE:{val_scores[0]:.4f} SSIM:{val_scores[1]:.4f} PSNR:{val_scores[2]:.4f}')
                    """
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                    """
                    writer.add_images('images', imgs, global_step)
                    writer.add_images('pred', targets_pred, global_step)
                    writer.add_images('ground_truths', true_targets, global_step)


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-i', '--dir_img', dest='dir_img', type=str, default='data/imgs/',
                        help='Input directory')
    parser.add_argument('-t', '--dir_target', dest='dir_target', type=str, default='data/targets/',
                        help='Target directory')
    parser.add_argument('-c', '--dir_checkpoint', dest='dir_checkpoint', type=str, default='checkpoints/',
                        help='Checkpoints directory')
    parser.add_argument('-g', '--dir_log', dest='dir_log', type=str, default='runs/',
                        help='Log directory for tensorboard')
    parser.add_argument('-a', '--use_augment', action='store_true', help='use augmented data')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=3, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  dir_img = args.dir_img,
                  dir_target = args.dir_target,
                  dir_checkpoint = args.dir_checkpoint,
                  dir_log = args.dir_log,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  augmented=args.use_augment)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
