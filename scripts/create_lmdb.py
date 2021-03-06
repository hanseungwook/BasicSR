import mmcv
import torch
from os import path as osp

from basicsr.utils.lmdb import make_lmdb_from_imgs, make_lr_lmdb_from_imgs, make_wt_lmdb_from_imgs, make_inter_wt_lmdb_from_imgs


def create_lmdb_for_div2k():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = 'datasets/DIV2K/DIV2K_train_HR_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx2 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X2_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx3 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X3_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx4 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_imagenet():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR train images (256x)
    # folder_path = '/disk_c/han/data/ImageNet_256x256/train/'
    # lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_train_HR.lmdb'
    # img_path_list, keys = prepare_keys_imagenet(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, n_thread=16)

    # # HR val images (256x)
    # folder_path = '/disk_c/han/data/ImageNet_256x256/val/'
    # lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_val_HR.lmdb'
    # img_path_list, keys = prepare_keys_imagenet(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, n_thread=16)

    # LR train images (64x)
    # folder_path = '/disk_c/han/data/ImageNet_64x64/train/'
    # lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_train_LR.lmdb'
    # img_path_list, keys = prepare_keys_imagenet(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, n_thread=16)

    # # LR val images (64x)
    # folder_path = '/disk_c/han/data/ImageNet_64x64/val/'
    # lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_val_LR.lmdb'
    # img_path_list, keys = prepare_keys_imagenet(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, n_thread=16)

    # LR val images (128x)
    folder_path = '/disk_c/han/data/ImageNet_128x128/val/'
    lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_val_128.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, n_thread=16)

def create_lmdb_for_imagenet_wt():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR train images
    folder_path = '/disk_c/han/data/ImageNet_256x256/train/'
    lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_train_WT.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_wt_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # HR val images
    folder_path = '/disk_c/han/data/ImageNet_256x256/val/'
    lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_val_WT.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_wt_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_biggan_samples():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # Ground truth samples (in jpg format not png)
    # folder_path = '/disk_c/han/data/Pretrained_BigGAN_256x256/'
    # lmdb_path = '/disk_c/han/data/Pretrained_BigGAN_lmdb/Pretrained256_HR.lmdb'
    # img_path_list, keys = prepare_keys_imagenet_jpg(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, n_thread=16)

    # Samples WT'ed (in jpg format not png)
    folder_path = '/disk_c/han/data/Pretrained_BigGAN_256x256/'
    lmdb_path = '/disk_c/han/data/Pretrained_BigGAN_lmdb/Pretrained256_WT.lmdb'
    img_path_list, keys = prepare_keys_imagenet_jpg(folder_path)
    make_wt_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_lmdb_for_imagenet_inter_wt():
    """Create lmdb files for ImageNet dataset. Applying interpolation from
    256 => 128 and then applying 1 WT from 128 => 64.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR train images
    folder_path = '/disk_c/han/data/ImageNet_256x256/train/'
    lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_train_INTER_WT.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_inter_wt_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # HR val images
    folder_path = '/disk_c/han/data/ImageNet_256x256/val/'
    lmdb_path = '/disk_c/han/data/ImageNet_lmdb/ImageNet_val_INTER_WT.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_inter_wt_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_lsun_church_wt():
    """Create lmdb files for LSUN Church dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR train images
    folder_path = '/disk_c/han/data/lsun_church_256x256/train/'
    lmdb_path = '/disk_c/han/data/lsun_church_lmdb/lsun_church_train_HR.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, n_thread=16)

    # HR val images
    folder_path = '/disk_c/han/data/lsun_church_256x256/val/'
    lmdb_path = '/disk_c/han/data/lsun_church_lmdb/lsun_church_val_HR.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, n_thread=16)

    # WT train images
    folder_path = '/disk_c/han/data/lsun_church_256x256/train/'
    lmdb_path = '/disk_c/han/data/lsun_church_lmdb/lsun_church_train_WT.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_wt_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # WT val images
    folder_path = '/disk_c/han/data/lsun_church_256x256/val/'
    lmdb_path = '/disk_c/han/data/lsun_church_lmdb/lsun_church_val_WT.lmdb'
    img_path_list, keys = prepare_keys_imagenet(folder_path)
    make_wt_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def prepare_keys_imagenet(folder_path):
    """Prepare image path list and keys for ImageNet dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(folder_path, suffix=('', 'png'))))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def prepare_keys_imagenet_jpg(folder_path):
    """Prepare image path list and keys for ImageNet dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(folder_path, suffix=('', 'jpg'))))
    keys = [img_path.split('.jpg')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def create_lmdb_for_reds():
    """Create lmdb files for REDS dataset.

    Usage:
        Before run this script, please run `merge_reds_train_val.py`.
        We take two folders for example:
            train_sharp
            train_sharp_bicubic
        Remember to modify opt configurations according to your settings.
    """
    # train_sharp
    folder_path = 'datasets/REDS/train_sharp'
    lmdb_path = 'datasets/REDS/train_sharp_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb_from_imgs(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_sharp_bicubic
    folder_path = 'datasets/REDS/train_sharp_bicubic'
    lmdb_path = 'datasets/REDS/train_sharp_bicubic_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb_from_imgs(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_reds(folder_path):
    """Prepare image path list and keys for REDS dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def create_lmdb_for_vimeo90k():
    """Create lmdb files for Vimeo90K dataset.

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # GT
    folder_path = 'datasets/vimeo90k/vimeo_septuplet/sequences'
    lmdb_path = 'datasets/vimeo90k/vimeo90k_train_GT_only4th.lmdb'
    train_list_path = 'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path,
                                                'gt')
    make_lmdb_from_imgs(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # LQ
    folder_path = 'datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
    lmdb_path = 'datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
    train_list_path = 'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path,
                                                'lq')
    make_lmdb_from_imgs(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_vimeo90k(folder_path, train_list_path, mode):
    """Prepare image path list and keys for Vimeo90K dataset.

    Args:
        folder_path (str): Folder path.
        train_list_path (str): Path to the official train list.
        mode (str): One of 'gt' or 'lq'.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    with open(train_list_path, 'r') as fin:
        train_list = [line.strip() for line in fin]

    img_path_list = []
    keys = []
    for line in train_list:
        folder, sub_folder = line.split('/')
        img_path_list.extend(
            [osp.join(folder, sub_folder, f'im{j + 1}.png') for j in range(7)])
        keys.extend([f'{folder}/{sub_folder}/im{j + 1}' for j in range(7)])

    if mode == 'gt':
        print('Only keep the 4th frame for the gt mode.')
        img_path_list = [v for v in img_path_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('/im4')]

    return img_path_list, keys


if __name__ == '__main__':
    create_lmdb_for_imagenet()
    # create_lmdb_for_imagenet_wt()
    # create_lmdb_for_biggan_samples()
    # create_lmdb_for_imagenet_inter_wt()
    # create_lmdb_for_lsun_church_wt()
    # create_lmdb_for_imagenet_lr()
    # create_lmdb_for_div2k()
    # create_lmdb_for_reds()
    # create_lmdb_for_vimeo90k()
