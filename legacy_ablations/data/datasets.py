import os, glob
import torch, sys
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        # 加载nii.gz格式的image和label图像
        image_nii = nib.load(path)
        fixed_dir = 'D:/1DATA/LPBA40/fixed.nii.gz'
        fixedimage = nib.load(fixed_dir)
        # 将image和label转换为numpy数组
        x = image_nii.get_fdata()
        y = fixedimage.get_fdata()
        # 执行变换之后，将numpy数据转换为Tensor
        x, y = self.transforms([x[None, ...], y[None, ...]])

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        # 加载nii.gz格式的image和label图像
        image_nii = nib.load(path)
        fixed_dir = 'D:/1DATA/LPBA40/fixed.nii.gz'
        fixedimage = nib.load(fixed_dir)

        fixedlable_dir = 'D:/1DATA/LPBA40/label/S01.delineation.structure.label.nii.gz'
        fixedlableimage = nib.load(fixedlable_dir)

        x_seg_dir = 'D:/1DATA/LPBA40/label/'
        x_seg_path = os.path.join(x_seg_dir, os.path.basename(path))
        x_seg_nii = nib.load(x_seg_path)

        # 将image和label转换为numpy数组
        x = image_nii.get_fdata()
        y = fixedimage.get_fdata()
        y_seg = fixedlableimage.get_fdata()
        x_seg = x_seg_nii.get_fdata()

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)