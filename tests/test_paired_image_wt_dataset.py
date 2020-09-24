import math
import mmcv
import torchvision.utils
import numpy as np

from basicsr.data import create_dataloader, create_dataset
from basicsr.models.archs import arch_util


def main(mode='folder'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'ImageNetWT'
    opt['type'] = 'PairedImageWTDataset'
    # if mode == 'folder':
    #     opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
    #     opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    #     opt['filename_tmpl'] = '{}'
    #     opt['io_backend'] = dict(type='disk')
    # elif mode == 'meta_info_file':
    #     opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
    #     opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    #     opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'  # noqa:E501
    #     opt['filename_tmpl'] = '{}'
    #     opt['io_backend'] = dict(type='disk')
    if mode == 'lmdb':
        opt['dataroot_gt'] = '/disk_c/han/data/ImageNet_lmdb/ImageNet_train_HR.lmdb'
        opt['dataroot_lq'] = '/disk_c/han/data/ImageNet_lmdb/ImageNet_train_INTER_WT.lmdb'
        opt['io_backend'] = dict(type='lmdb')

    opt['gt_size'] = 256

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 2
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1

    mmcv.mkdir_or_exist('tmp')

    dataset = create_dataset(opt)
    data_loader = create_dataloader(
        dataset, opt, num_gpu=1, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    device = 'cuda:0'
    inv_filters = arch_util.create_inv_filters().to(device)

    print('start...')
    min_val = float('inf')
    max_val = float('-inf')
    for i, data in enumerate(data_loader):
        print(i)

        lq = data['lq'].to(device)
        lq_iwt = arch_util.iwt(lq, inv_filters, levels=1)
        min_val = min(min_val, lq_iwt.min())
        max_val = max(max_val, lq_iwt.max())
    
    print('Min val: {}\nMax val: {}'.format(min_val, max_val))
    shift = torch.ceil(torch.abs(min_val))
    scale = shift + torch.ceil(max_val)
    print('Shift: {}\nScale: {}'.format(shift, scale))

    np.savez('/disk_c/han/data/biggan/pretrained512_norm_values.npz', **{'min' : min_val.cpu(), 'max' : max_val.cpu(), 'shift': shift.cpu(), 'scale': scale.cpu()})

        

if __name__ == '__main__':
    main()
