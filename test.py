import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from numpy import moveaxis
from torchvision import transforms

from eval import eval_gen_net
from predict import predict_img
from unet import UNet
from utils.data_vis import plot_img_and_target
from utils.dataset import BasicDataset

def test_net(net,
            device,
            dir_img,
            dir_target,
            dir_out,
            img_scale = 1.0,
            val_percent = 0.1,
            batch_size = 16):
    
    dataset = BasicDataset(dir_img, dir_target, img_scale, augmented=False)
    n_val = int(len(dataset) * val_percent)
    indices = list(range(len(dataset)))

    test_idx = indices[:n_val]
    test_sampler = SubsetRandomSampler(test_idx)

    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=8, pin_memory=True, drop_last=True)
    with torch.no_grad():
        test_score = eval_gen_net(net, val_loader, device, dir_out)
    print(f"Test Score: {test_score}")



def get_args():
    parser = argparse.ArgumentParser(description='Predict targets from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, metavar='FILE',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-i', '--dir_img', dest='dir_img', type=str, default='data/imgs/',
                        help='Input directory')
    parser.add_argument('-t', '--dir_target', dest='dir_target', type=str, default='data/targets/',
                        help='Target directory')
    parser.add_argument('-c', '--dir_out', dest='dir_out', type=str, default='results/',
                        help='Generated directory')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    net = UNet(n_channels=3, n_classes=3)
    logging.basicConfig(level = logging.INFO)
    logging.info("Loading model {}".format(args.load))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    
    test_net(net=net,
              dir_img = args.dir_img,
              dir_target = args.dir_target,
              dir_out = args.dir_out,
              batch_size=args.batchsize,
              device=device,
              img_scale=args.scale,
              val_percent=args.val / 100)

    print("Generation completed.")