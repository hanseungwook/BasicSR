import cv2
import mmcv
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp

from basicsr.utils.util import ProgressBar
from basicsr.data.util import duf_downsample
from torchvision.transforms import ToTensor


def main():
    """A multi-thread tool to crop ImageNet images to 256x256.

    It is used for ImageNet dataset.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.

        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR
            DIV2K_train_LR_bicubic/X2
            DIV2K_train_LR_bicubic/X3
            DIV2K_train_LR_bicubic/X4
        After process, each sub_folder should have the same number of
        subimages.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    # HR train images
    opt['input_folder'] = '/disk_c/han/data/ImageNet_256x256/train/'
    opt['save_folder'] = '/disk_c/han/data/ImageNet_64x64/train/'
    opt['scale'] = 4
    extract_subimages(opt)

    # HR val images
    opt['input_folder'] = '/disk_c/han/data/ImageNet_256x256/val/'
    opt['save_folder'] = '/disk_c/han/data/ImageNet_64x64/val/'
    opt['scale'] = 4
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Overwriting.')

    img_list = list(mmcv.scandir(input_folder, recursive=True))
    img_list = [osp.join(input_folder, v) for v in img_list]

    pbar = ProgressBar(len(img_list))
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: pbar.update(arg))
    pool.close()
    pool.join()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    scale = opt['scale']
    img_name, extension = osp.splitext(osp.basename(path))
    extension = '.png'

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    img = ToTensor()(img)
    downsampled_img = duf_downsample(img.unsqueeze(0), scale=scale).squeeze(0)
    downsampled_img *= 255.0
    downsampled_img = downsampled_img.numpy()

    cv2.imwrite(
        osp.join(opt['save_folder'],
                    f'{img_name}{extension}'), downsampled_img,
        [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
