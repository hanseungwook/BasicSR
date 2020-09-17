from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs import arch_util as arch_util
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

class OutConv2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(OutConv2, self).__init__()
        if not mid_channels:
            mid_channels = in_channels // 2

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet_NTail_128_Mod(nn.Module):
    def __init__(self, n_channels, n_classes, n_tails=3, bilinear=True):
        super(UNet_NTail_128_Mod, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()

        self.inc = DoubleConv(n_channels, 512)
        self.down1 = Down(512, 512)
        self.down2 = Down(512, 512)
        factor = 2 if bilinear else 1
        self.down3 = Down(512, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(1024, 512, bilinear)
        self.outc_modules = nn.ModuleList()
        for i in range(n_tails):
            self.outc_modules.append(OutConv2(512, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_out = torch.empty(0, device=x.device)

        for layer in self.outc_modules:
            cur_x_out = layer(x)
            x_out = torch.cat((x_out, cur_x_out), dim=1)
        
        return x_out

class UNet_Hierarchical(nn.Module):
    def __init__(self):
        super(UNet_Hierarchical, self).__init__()
        self.unet_128 = UNet_NTail_128_Mod(n_channels=12, n_classes=3, n_tails=12, bilinear=True)
        self.unet_256 = UNet_NTail_128_Mod(n_channels=48, n_classes=3, n_tails=48, bilinear=True)

        inv_filters = arch_util.create_inv_filters()
        self.register_buffer('inv_filters', inv_filters)

    def forward(self, x):
        '''
        Args:
            x (tensor): Input to the model, assuming B x C x H x W (wavelet transformation applied to image thrice)

        Returns:
            out (tensor): Output of model, B x C x (4H) x (4W) (high frequency mask IWT'ed to image space)
        '''
        # Split input into 4 quadrants and concatenate channel-wise
        real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = arch_util.get_4masks(x, 32)
        Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

        # Run through model 128
        recon_mask_128_all = self.unet_128(Y_64_patches)
        recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br = arch_util.split_masks_from_channels(recon_mask_128_all)

        Y_128_patches = torch.cat((Y_64_patches, recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br), dim=1)

        # Run through 128 mask network and get reconstructed image
        recon_mask_256_all = self.unet_256(Y_128_patches)

        _, recon_img = arch_util.mask_outputs_to_img(Y_64, recon_mask_128_all, recon_mask_256_all, self.inv_filters, mask=False)

        return recon_img


