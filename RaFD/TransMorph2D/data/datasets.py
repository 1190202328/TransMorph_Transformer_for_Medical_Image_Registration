import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data_utils import pkload


class FIREDataset(Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms
        self.init()

    def init(self):
        full_path_pair = []
        added_names = set()
        for path in os.listdir(self.data_dir):
            prefix = path.split('_')[0]
            if prefix in added_names:
                continue
            added_names.add(prefix)
            full_path_pair.append([f'{self.data_dir}/{prefix}_1.jpg', f'{self.data_dir}/{prefix}_2.jpg'])
        self.paths = full_path_pair

    def __getitem__(self, index):
        path1, path2 = self.paths[index]
        image_1 = Image.open(path1)
        image_2 = Image.open(path2)
        x = image_1.convert('RGB')
        y = image_2.convert('RGB')
        x_grey = image_1.convert('1')
        y_grey = image_2.convert('1')
        x = self.transforms(x)
        y = self.transforms(y)
        x_grey = self.transforms(x_grey)
        y_grey = self.transforms(y_grey)
        return x, y, x_grey, y_grey

    def __len__(self):
        return len(self.paths)


class RaFDDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_gray, y_gray = pkload(path)
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        x_gray, y_gray = self.transforms([x_gray, y_gray])
        # plt.figure()
        # plt.imshow(x_gray[0], cmap='gray')
        # plt.show()
        x = np.ascontiguousarray(x_gray)
        y = np.ascontiguousarray(y_gray)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class RaFDInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_gray, y_gray = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_gray, y_gray = x_gray[None, ...], y_gray[None, ...]
        x_gray = np.ascontiguousarray(x_gray.astype(np.float32))
        y_gray = np.ascontiguousarray(y_gray.astype(np.float32))
        x = np.ascontiguousarray(x.astype(np.float32))
        y = np.ascontiguousarray(y.astype(np.float32))
        x_gray, y_gray = torch.from_numpy(x_gray), torch.from_numpy(y_gray)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, x_gray, y_gray

    def __len__(self):
        return len(self.paths)
