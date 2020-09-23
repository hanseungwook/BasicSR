import mmcv
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, totensor
from basicsr.models.archs import arch_util


class SingleNumpyDataset(data.Dataset):
    """Numpy single image (WT) dataset.

    Read 64x64 WT images (need to apply one more WT operation in order to split it into 4 quadrants as the models expect)


    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq (path to NPZ file).
    """

    def __init__(self, opt):
        super(SingleNumpyDataset, self).__init__()
        self.opt = opt

        self.lq_folder = opt['dataroot_lq']
        self.lq_dataset = np.load(self.lq_folder)['x']
        
        self.filters = arch_util.create_filters()

    def __getitem__(self, index):
        # Apply 1 more WT
        img = self.lq_dataset[index]
        img_wt = arch_util.wt(torch.from_numpy(img).unsqueeze(0), self.filters, levels=1).squeeze()

        return {'lq': img_wt, 'lq_path': self.lq_folder.split('.')[0] + '_{}.png'.format(index)}

    def __len__(self):
        return self.lq_dataset.shape[0]
