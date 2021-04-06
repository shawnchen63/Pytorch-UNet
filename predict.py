import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from numpy import moveaxis
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_target
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = BasicDataset.preprocess(full_img, scale_factor, train=False)
    
    r,g,b = img[0]+1, img[1]+1, img[2]+1
    gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
    
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    gray = torch.from_numpy(gray).type(torch.FloatTensor)
    gray = torch.unsqueeze(gray, 0)
    gray = torch.unsqueeze(gray, 0)
    gray = gray.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = net(img, gray)
        output = output.squeeze()
        """
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_target = probs.squeeze().cpu().numpy()
        """

    return output.cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict targets from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output targets",
                        default=False)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1.0)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def target_to_image(target):
    target = moveaxis(target, 0, 2)
    return Image.fromarray((target * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=3, bilinear=True, self_attention=True)
    logging.basicConfig(level = logging.INFO)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nGenerating image {} ...".format(fn))

        img = Image.open(fn).convert('RGB')
        target = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = target_to_image(target)
            result = result.resize((img.size[0],img.size[1]))
            try:
                Path(out_fn).parents[0].mkdir(parents=True, exist_ok=True)
            except:
                print("directory exists")
            result.save(out_files[i])

            logging.info("target saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_target(img, target)
