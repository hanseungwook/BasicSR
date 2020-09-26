import mmcv
import numpy as np
from torch.utils import data as data
import torch
import h5py

from basicsr.data.transforms import augment, paired_random_crop, totensor
from basicsr.models.archs import arch_util


class SingleHDF5Dataset(data.Dataset):
    """HDF5 single image dataset.

    Read 64x64 downsampled (using DUF) images


    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq (path to NPZ file).
    """

    def __init__(self, opt):
        super(SingleHDF5Dataset, self).__init__()
        self.opt = opt

        self.lq_folder = opt['dataroot_lq']
        self.lq_dataset = h5py.File(self.lq_Folder, 'r').get('x')

    def __getitem__(self, index):
        return {'lq': self.lq_dataset[index], 'lq_path': self.lq_folder.split('.')[0] + '_{}.png'.format(index)}

    def __len__(self):
        return self.lq_dataset.shape[0]
