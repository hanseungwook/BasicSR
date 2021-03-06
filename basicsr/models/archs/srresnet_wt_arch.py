import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs import arch_util as arch_util
import pywt


class MSRResNet_WT_Pixel(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4.
            Default: 4.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_block=16,
                 upscale=4):
        super(MSRResNet_WT_Pixel, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat,
                                     num_feat * self.upscale * self.upscale, 3,
                                     1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights(
            [self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            arch_util.default_init_weights(self.upconv2, 0.1)

        # WT filter
        inv_filters = arch_util.create_inv_filters()
        self.register_buffer('inv_filters', inv_filters)

        # Normalization buffers
        self.register_buffer(
            'shift',
            torch.Tensor([3.0]))
        self.register_buffer(
            'scale',
                torch.Tensor([10.0]))

    def forward(self, x):
        # IWT x once to get LFC
        x = arch_util.iwt(x, self.inv_filters, 1)

        # Normalize to (0, 1) range
        x = arch_util.normalize_wt(x, self.shift, self.scale)
        assert (x.min() >= 0.0 and x.max() <= 1.0)
        
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))

        # IWT'ed version of x with zero padding for dimensions
        base = arch_util.iwt(arch_util.zero_pad(x, x.shape[3]*self.upscale, x.device), self.inv_filters, 2)
        out += base
        return out
