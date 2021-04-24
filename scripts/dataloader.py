import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from skimage import io
from skimage.color import rgb2gray


class GANDataset(Dataset):
    """
    Dataset for human faces and cartoon faces, the channels are flipped already.
    """

    def __init__(self, args):
        self.x = args.x
        self.x_array = self._get_file_list(args.x)
        len1 = len(self.x_array)
        self.latentDims = args.LatentDims
        if not self.latentDims:
            self.y = args.y
            self.y_array = self._get_file_list(args.y)
            len2 = len(self.y_array)

            if len1 > len2:
                self.trueLen = len2
                self.x_array = self.x_array[:len2]
            else:
                self.trueLen = len1
                self.y_array = self.y_array[:len1]

            assert(len(self.x_array) == len(self.y_array))
        else:
            self.trueLen = len1

        self.gray = args.gray

    def _get_file_list(self, root_dir):
        x = []
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for name in files:
                if name.endswith(".jpg") or name.endswith(".png"):
                    x.append(os.path.join(root, name))
        return x

    def __len__(self):
        return self.trueLen

    def load(self, root_dir, idstr):
        # img_name = os.path.join(root_dir, idstr)
        img_name = idstr
        image = io.imread(img_name).astype("float32")
        if self.gray:
            image = rgb2gray(image)
        image = (image / 127.5) - 1
        if len(image.shape) == 3:
            image = np.moveaxis(image, -1, 0)
        elif len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image).float()

    def __getitem__(self, idx):
        x_img = self.load(self.x, self.x_array[idx])
        if not self.latentDims:
            y_img = self.load(self.y, self.y_array[idx])
            return y_img, x_img
        return x_img


def get_data_loader(args):
    data_set = GANDataset(args)
    dataloader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    return dataloader
