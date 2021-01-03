from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from utils.image_folder import make_dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, targets_dir, scale=1, target_suffix='Target'):
        self.imgs_dir = imgs_dir
        self.targets_dir = targets_dir
        self.scale = scale
        self.target_suffix = target_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        #self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
        #            if not file.startswith('.')]

        self.imgs_paths = make_dataset(self.imgs_dir)
        #self.target_paths = sorted(make_dataset(self.targets_dir))
        logging.info(f'Creating dataset with {len(self.imgs_paths)} input images')

    def __len__(self):
        return len(self.imgs_paths)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        img_file = self.imgs_paths[i]
        idx = img_file.split('/')[-1].split('_')[0] + "_"
        target_file = glob(self.targets_dir + idx + self.target_suffix + '.*')

        assert len(target_file) == 1, \
            f'Either no target or multiple targets found for the ID {idx}: {target_file}'
        target = Image.open(target_file[0])
        img = Image.open(img_file)

        assert img.size == target.size, \
            f'Image and target {idx} should be the same size, but are {img.size} and {target.size}'

        img = self.preprocess(img, self.scale)
        target = self.preprocess(target, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'target': torch.from_numpy(target).type(torch.FloatTensor),
            'idx': idx.split("_")[0]
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, targets_dir, scale=1):
        super().__init__(imgs_dir, targets_dir, scale, target_suffix='_target')
