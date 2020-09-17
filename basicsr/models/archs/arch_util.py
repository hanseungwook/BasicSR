import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.models.ops.dcn import (ModulatedDeformConvPack,
                                    modulated_deform_conv)
from basicsr.utils import get_root_logger
import pywt


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=True)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)

# Creating filters for WT
def create_filters(wt_fn='bior2.2'):
    w = pywt.Wavelet(wt_fn)

    dec_hi = torch.Tensor(w.dec_hi[::-1])
    dec_lo = torch.Tensor(w.dec_lo[::-1])

    filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                           dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                           dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                           dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

    return filters

def create_inv_filters(wt_fn='bior2.2'):
    w = pywt.Wavelet(wt_fn)

    rec_hi = torch.Tensor(w.rec_hi)
    rec_lo = torch.Tensor(w.rec_lo)
    
    inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

    return inv_filters


# Zeroing out the first patch's portion of the mask
def zero_mask(mask, num_iwt, cur_iwt):
    padded = torch.zeros(mask.shape, device=mask.device)
    h = mask.shape[2]

    inner_patch_h0 = h // (np.power(2, num_iwt-cur_iwt+1))
    inner_patch_w0 = h // (np.power(2, num_iwt-cur_iwt+1))

    if len(mask.shape) == 3:
        padded[:, inner_patch_h0:, :] = mask[:, inner_patch_h0:, :]
        padded[:, :inner_patch_h0, inner_patch_w0:] = mask[:, :inner_patch_h0, inner_patch_w0:]
    elif len(mask.shape) == 4:
        padded[:, :, inner_patch_h0:, :] = mask[:, :, inner_patch_h0:, :]
        padded[:, :, :inner_patch_h0, inner_patch_w0:] = mask[:, :, :inner_patch_h0, inner_patch_w0:]
    
    return padded


# Create padding on patch so that this patch is formed into a square image with other patches as 0
# 3 x 128 x 128 => 3 x target_dim x target_dim
def zero_pad(img, target_dim, device='cpu'):
    batch_size = img.shape[0]
    num_channels = img.shape[1]
    padded_img = torch.zeros((batch_size, num_channels, target_dim, target_dim), device=device)
    padded_img[:, :, :img.shape[2], :img.shape[3]] = img.to(device)
    
    return padded_img


def wt(vimg, filters, levels=1):
    bs = vimg.shape[0]
    h = vimg.size(2)
    w = vimg.size(3)
    vimg = vimg.reshape(-1, 1, h, w)
    padded = torch.nn.functional.pad(vimg,(2,2,2,2))
    res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]),stride=2)
    if levels>1:
        res[:,:1] = wt(res[:,:1], filters, levels-1)
        res[:,:1,32:,:] = res[:,:1,32:,:]*1.
        res[:,:1,:,32:] = res[:,:1,:,32:]*1.
        res[:,1:] = res[:,1:]*1.
    res = res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)

    return res.reshape(bs, -1, h, w)


def iwt(vres, inv_filters, levels=1):
    bs = vres.shape[0]
    h = vres.size(2)
    w = vres.size(3)
    vres = vres.reshape(-1, 1, h, w)
    res = vres.contiguous().view(-1, h//2, 2, w//2).transpose(1, 2).contiguous().view(-1, 4, h//2, w//2).clone()
    if levels > 1:
        res[:,:1] = iwt(res[:,:1], inv_filters, levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, Variable(inv_filters[:,None]),stride=2)
    res = res[:,:,2:-2,2:-2] #removing padding

    return res.reshape(bs, -1, h, w)
    

# Returns IWT of img with only TL patch (low frequency) zero-ed out
def wt_hf(vimg, filters, inv_filters, levels=1):
    # Apply WT
    wt_img = wt(vimg, filters, levels)

    # Zero out TL patch
    wt_img_hf = zero_mask(wt_img, levels, 1)

    # Apply IWT
    iwt_img_hf = iwt(wt_img_hf, inv_filters, levels)

    return iwt_img_hf

def get_3masks(img, mask_dim):
    tr = img[:, :, :mask_dim, mask_dim:]
    bl = img[:, :, mask_dim:, :mask_dim]
    br = img[:, :, mask_dim:, mask_dim:]
    
    return tr.squeeze(), bl.squeeze(), br.squeeze()


# Gets 4 masks in order of top-left, top-right, bottom-left, and bottom-right quadrants
def get_4masks(img, mask_dim):
    tl = img[:, :, :mask_dim, :mask_dim]
    tr = img[:, :, :mask_dim, mask_dim:]
    bl = img[:, :, mask_dim:, :mask_dim]
    br = img[:, :, mask_dim:, mask_dim:]
    
    return tl, tr, bl, br

# Splits 3 masks collated channel-wise
def split_masks_from_channels(data):
    nc_mask = data.shape[1] // 3
    
    return data[:, :nc_mask, :, :], data[:, nc_mask:2*nc_mask, :, :], data[:, 2*nc_mask:, :, :]

def collate_patches_to_img(tl, tr, bl, br, device='cpu'):
    bs = tl.shape[0]
    c = tl.shape[1]
    h = tl.shape[2]
    w = tl.shape[3]
    
    frame = torch.empty((bs, c, 2*h, 2*w), device=device)
    frame[:, :, :h, :w] = tl.to(device)
    frame[:, :, :h, w:] = tr.to(device)
    frame[:, :, h:, :w] = bl.to(device)
    frame[:, :, h:, w:] = br.to(device)
    
    return frame

# Assumes four patches concatenated channel-wise and converts into image
def collate_channels_to_img(img_channels, device='cpu'):
    bs = img_channels.shape[0]
    c = img_channels.shape[1] // 4
    h = img_channels.shape[2]
    w = img_channels.shape[3]
    
    img = collate_patches_to_img(img_channels[:,:c], img_channels[:,c:2*c], img_channels[:, 2*c:3*c], img_channels[:, 3*c:], device)
    
    return img

# Assumes 16 patches concatenated channel-wise and converts into image
def collate_16_channels_to_img(img_channels, device='cpu'):
    bs = img_channels.shape[0]
    c = img_channels.shape[1] // 4
    h = img_channels.shape[2]
    w = img_channels.shape[3]
    
    tl = collate_channels_to_img(img_channels[:, :c], device)
    tr = collate_channels_to_img(img_channels[:, c:2*c], device)
    bl = collate_channels_to_img(img_channels[:, 2*c:3*c], device)
    br = collate_channels_to_img(img_channels[:, 3*c:], device)
    
    img = collate_patches_to_img(tl, tr, bl, br, device)
    
    return img


def apply_iwt_quads_128(img_quad, inv_filters):
    h = img_quad.shape[2] // 2
    w = img_quad.shape[3] // 2
    
    img_quad[:, :, :h, w:] = iwt(img_quad[:, :, :h, w:], inv_filters, levels=1)
    img_quad[:, :, h:, :w] = iwt(img_quad[:, :, h:, :w], inv_filters, levels=1)
    img_quad[:, :, h:, w:] = iwt(img_quad[:, :, h:, w:], inv_filters, levels=1)
    
    img_quad = iwt(img_quad, inv_filters, levels=2)
    
    return img_quad

# Function for arranging two levels of outputs (128, 256) into an image
# Outputs two reconstructions -- one with only 128 level and another with both 128 and 256 level masks
# If mask option is True, then add a fully IWT'ed reconstructed mask
def mask_outputs_to_img(Y_64, recon_mask_128_all, recon_mask_256_all, inv_filters, mask=False): 
    device = recon_mask_128_all.device
    
    # Split 128 and 256 level outputs into quadrants
    recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br = split_masks_from_channels(recon_mask_128_all)
    recon_mask_256_tr, recon_mask_256_bl, recon_mask_256_br = split_masks_from_channels(recon_mask_256_all)

    # Collate all masks constructed by first 128 level
    recon_mask_128_tr_img = collate_channels_to_img(recon_mask_128_tr, device)
    recon_mask_128_bl_img = collate_channels_to_img(recon_mask_128_bl, device)   
    recon_mask_128_br_img = collate_channels_to_img(recon_mask_128_br, device)
    
    recon_mask_128_tr_img = iwt(recon_mask_128_tr_img, inv_filters, levels=1)
    recon_mask_128_bl_img = iwt(recon_mask_128_bl_img, inv_filters, levels=1)
    recon_mask_128_br_img = iwt(recon_mask_128_br_img, inv_filters, levels=1) 
    
    recon_mask_128_iwt = collate_patches_to_img(Y_64, recon_mask_128_tr_img, recon_mask_128_bl_img, recon_mask_128_br_img)

    # Collate all masks concatenated by channel to an image (slice up and put into a square)
    recon_mask_256_tr_img = collate_16_channels_to_img(recon_mask_256_tr, device)
    recon_mask_256_bl_img = collate_16_channels_to_img(recon_mask_256_bl, device)   
    recon_mask_256_br_img = collate_16_channels_to_img(recon_mask_256_br, device)

    recon_mask_256 = collate_patches_to_img(torch.zeros(recon_mask_256_tr_img.shape, device=device), recon_mask_256_tr_img, recon_mask_256_bl_img, recon_mask_256_br_img)
    
    recon_mask_256_tr_img = apply_iwt_quads_128(recon_mask_256_tr_img, inv_filters)
    recon_mask_256_bl_img = apply_iwt_quads_128(recon_mask_256_bl_img, inv_filters)
    recon_mask_256_br_img = apply_iwt_quads_128(recon_mask_256_br_img, inv_filters)
    
    recon_mask_256_iwt = collate_patches_to_img(torch.zeros(recon_mask_256_tr_img.shape, device=device), recon_mask_256_tr_img, recon_mask_256_bl_img, recon_mask_256_br_img)
    
    recon_mask_padded = zero_pad(recon_mask_256_iwt, 256, device)
    recon_mask_padded[:, :, :128, :128] = recon_mask_128_iwt
    recon_img = iwt(recon_mask_padded, inv_filters, levels=3)

    recon_mask_128_padded = zero_pad(recon_mask_128_iwt, 256, device)
    recon_img_128 = iwt(recon_mask_128_padded, inv_filters, levels=3)

    if mask:
        recon_mask_128_iwt = collate_patches_to_img(torch.zeros(recon_mask_128_tr_img.shape, device=device), recon_mask_128_tr_img, recon_mask_128_bl_img, recon_mask_128_br_img)
        recon_mask_256_iwt = collate_patches_to_img(torch.zeros(recon_mask_256_tr_img.shape, device=device), recon_mask_256_tr_img, recon_mask_256_bl_img, recon_mask_256_br_img)
        recon_mask_padded = zero_pad(recon_mask_256_iwt, 256, device)
        recon_mask_padded[:, :, :128, :128] = recon_mask_128_iwt
        recon_mask = iwt(recon_mask_padded, inv_filters, levels=3)

        return recon_img_128, recon_img, recon_mask

    return recon_img_128, recon_img