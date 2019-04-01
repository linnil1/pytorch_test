import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pytorch_ssim


class CountLoss(nn.Module):
    def __init__(self, ssim=.001, window_size=11):
        super().__init__()
        self.ssim = ssim
        self.loss_fn = nn.MSELoss()
        self.loss_ssim_fn = pytorch_ssim.SSIM(window_size=window_size)

    def forward(self, dmap, gtmap):
        loss = self.loss_fn(dmap, gtmap)
        loss_ssim = 1 - self.loss_ssim_fn(dmap, gtmap)
        return loss + self.ssim * loss_ssim


# UNET
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.norm = nn.InstanceNorm2d(n_channels)
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = nn.Dropout(0.2)(x2)
        x3 = self.down2(x2)
        x3 = nn.Dropout(0.3)(x3)
        x4 = self.down3(x3)
        x4 = nn.Dropout(0.4)(x4)
        x5 = self.down4(x4)
        x5 = nn.Dropout(0.5)(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.act(x)
        return x

def block(n_in, n_out, batch=True):
    layer = [nn.Conv2d(n_in, n_out, kernel_size=4, stride=2, padding=1)]
    if batch:
        layer += [nn.InstanceNorm2d(n_out)]
        # layer += nn.BatchNorm2d()
    layer += [nn.LeakyReLU(0.2, inplace=True)]
    return layer


class Discriminator(nn.Module):
    def __init__(self, n_channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            *block(n_channels * 2, 64, batch=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


if __name__ == '__main__':
    a = torch.zeros(2, 1, 70, 70)
    d = Discriminator(1)
    print(d(a, a).size())

