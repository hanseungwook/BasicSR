import mmcv
import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, totensor
from basicsr.data.util import single_paths_from_lmdb
from basicsr.utils import FileClient


class SingleImageWTDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, WT).

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(SingleImageWTDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.lq_folder = opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = single_paths_from_lmdb(
                [self.lq_folder], ['lq'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.lq_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = [
                osp.join(self.lq_folder, v)
                for v in mmcv.scandir(self.lq_folder)
            ]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = np.copy(np.frombuffer(img_bytes, dtype='float32')).reshape(img_gt_h//scale, img_gt_w//scale, -1)

        # No augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
        #                                         gt_path)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
        #                              self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = totensor(img_lq, bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'lq_path': lq_path,
        }

    def __len__(self):
        return len(self.paths)
