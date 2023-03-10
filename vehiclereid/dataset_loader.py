from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from PIL import Image
import os.path as osp

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        # print(self.dataset)
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class DoubleImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, strong=None):
        self.dataset = dataset
        self.transform = transform
        self.strong = strong

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        # print(self.dataset)
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        strong = self.strong(img)
        if self.transform is not None:
            img = self.transform(img)
        
        img = torch.stack([img, strong], dim=0)

        return img, pid, camid, img_path