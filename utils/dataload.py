import glob
import sys

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

from mytransforms import mytransforms
import numpy as np
import torchvision
import random
import torch
import rasterio
import tifffile

class dataload_valid(Dataset):
    def __init__(self, path='train', H=600, W=480, pow_n=3, aug=True, mode='img'):
        self.H = H
        self.W = W
        self.pow_n = pow_n
        self.aug = aug
        self.mode = mode

        if mode == 'img':
            self.path = path
            self.data_num = 1
        elif mode == 'dir':
            self.path = glob.glob(path + '/*.png')
            self.data_num = len(self.path)

        self.mask_trans = transforms.Compose([transforms.Resize((self.H, self.W)),
                                              transforms.Grayscale(1),
                                              transforms.ToTensor()])
        self.norm = mytransforms.Compose([transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.mode == 'img':
            input = Image.open(self.path)
        if self.mode == 'dir':
            input = Image.open(self.path[idx])
        input = self.mask_trans(input)
        input = self.norm(input)

        return input


class dataload_train(Dataset):
    def __init__(self,  path='', H=600, W=480, aug=True, phase='train'):

        self.path_mtx = path
        self.phase = phase
        self.mask_num = int(len(self.path_mtx[0]))
        self.data_num = len(self.path_mtx)

        self.aug=aug
        self.H = H
        self.W = W

        self.mask_trans = transforms.Compose([
                                              # transforms.Resize((224, 224)),
                                              # mytransforms.Affine(0, translate=[0, 0], scale=1, fillcolor=0),
                                              transforms.ToTensor()])

        self.norm = mytransforms.Compose([transforms.Normalize((0.5,), (0.5,))])
        self.col_trans = transforms.Compose([transforms.ColorJitter(brightness=random.random())])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.aug:
            self.mask_trans.transforms[0].degrees = random.randrange(-25, 25)
            self.mask_trans.transforms[0].translate = [random.uniform(0, 0.05), random.uniform(0, 0.05)]
            self.mask_trans.transforms[0].scale = random.uniform(0.9, 1.1)

        if self.phase == 'train':
            _input = tifffile.imread(self.path_mtx[idx, 0])
            _input = np.float32(_input) / 65535
            mask = tifffile.imread(self.path_mtx[idx, 1])
            mask = np.float32(mask)

            # _input = self.tif_transform(_input, _type='image')
            # _mask = self.tif_transform(mask, _type='mask')

        else:
            _input = tifffile.imread(self.path_mtx[idx, 0])
            _input = np.float32(_input) / 65535
            _input = self.mask_trans(_input)
            return _input

        _input, mask = self.mask_trans(_input), self.mask_trans(mask)
        _input, mask = self.norm(_input), mask

        return [_input, mask]

    def get_img_762bands(self, path):
        img = rasterio.open(path).read((7, 6, 2)).transpose((1, 2, 0))
        img = np.float32(img) / 65535

        return img

    def tif_transform(self, image, _type):
        mask = torch.empty(10, self.H, self.W, dtype=torch.float)  # 150 * H * W

        for i in range(10):
            print(image.shape)
            mask[i] = self.mask_trans(image[:, :, i])
            if self.aug and _type != 'mask':
                mask[i] = self.col_trans(mask[i])
            mask[i] = self.norm(mask[i])

        return mask
