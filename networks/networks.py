import torch.nn as nn
from .revresnet import resnet18
from torch import cat


class ImageEncoder(nn.Module):
    """
    Used for 2.5D maps to 3D voxels
    """

    def __init__(self, input_nc, encode_dims=200):
        super().__init__()
        resnet_m = resnet18(pretrained=True)
        resnet_m.conv1 = nn.Conv2d(
            input_nc, 64, 7, stride=2, padding=3, bias=False
        )
        resnet_m.avgpool = nn.AdaptiveAvgPool2d(1)
        resnet_m.fc = nn.Linear(512, encode_dims)
        self.main = nn.Sequential(resnet_m)

    def forward(self, x):
        return self.main(x)


class VoxelDecoder(nn.Module):
    """
    Used for 2.5D maps to 3D voxels
    """

    def __init__(self, n_dims=200, nf=512):
        super().__init__()
        self.main = nn.Sequential(
            # volconv1
            deconv3d_add3(n_dims, nf, True),
            batchnorm3d(nf),
            relu(),
            # volconv2
            deconv3d_2x(nf, nf // 2, True),
            batchnorm3d(nf // 2),
            relu(),
            # volconv3
            nn.Sequential(),  # NOTE: no-op for backward compatibility; consider removing
            nn.Sequential(),  # NOTE
            deconv3d_2x(nf // 2, nf // 4, True),
            batchnorm3d(nf // 4),
            relu(),
            # volconv4
            deconv3d_2x(nf // 4, nf // 8, True),
            batchnorm3d(nf // 8),
            relu(),
            # volconv5
            deconv3d_2x(nf // 8, nf // 16, True),
            batchnorm3d(nf // 16),
            relu(),
            # volconv6
            deconv3d_2x(nf // 16, 1, True)
        )

    def forward(self, x):
        x_vox = x.view(x.size(0), -1, 1, 1, 1)
        return self.main(x_vox)


class VoxelGenerator(nn.Module):
    def __init__(self, nz=200, nf=64, bias=False, res=128):
        super().__init__()
        layers = [
            # nzx1x1x1
            deconv3d_add3(nz, nf * 8, bias),
            batchnorm3d(nf * 8),
            relu(),
            # (nf*8)x4x4x4
            deconv3d_2x(nf * 8, nf * 4, bias),
            batchnorm3d(nf * 4),
            relu(),
            # (nf*4)x8x8x8
            deconv3d_2x(nf * 4, nf * 2, bias),
            batchnorm3d(nf * 2),
            relu(),
            # (nf*2)x16x16x16
            deconv3d_2x(nf * 2, nf, bias),
            batchnorm3d(nf),
            relu(),
            # nfx32x32x32
        ]
        if res == 64:
            layers.append(deconv3d_2x(nf, 1, bias))
            # 1x64x64x64
        elif res == 128:
            layers += [
                deconv3d_2x(nf, nf, bias),
                batchnorm3d(nf),
                relu(),
                # nfx64x64x64
                deconv3d_2x(nf, 1, bias),
                # 1x128x128x128
            ]
        else:
            raise NotImplementedError(res)
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class VoxelDiscriminator(nn.Module):
    def __init__(self, nf=64, bias=False, res=128):
        super().__init__()
        layers = [
            # 1x64x64x64
            conv3d_half(1, nf, bias),
            relu_leaky(),
            # nfx32x32x32
            conv3d_half(nf, nf * 2, bias),
            # batchnorm3d(nf * 2),
            relu_leaky(),
            # (nf*2)x16x16x16
            conv3d_half(nf * 2, nf * 4, bias),
            # batchnorm3d(nf * 4),
            relu_leaky(),
            # (nf*4)x8x8x8
            conv3d_half(nf * 4, nf * 8, bias),
            # batchnorm3d(nf * 8),
            relu_leaky(),
            # (nf*8)x4x4
            conv3d_minus3(nf * 8, 1, bias),
            # 1x1x1
        ]
        if res == 64:
            pass
        elif res == 128:
            extra_layers = [
                conv3d_half(nf, nf, bias),
                relu_leaky(),
            ]
            layers = layers[:2] + extra_layers + layers[2:]
        else:
            raise NotImplementedError(res)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        y = self.main(x)
        return y.view(-1, 1).squeeze(1)


class Unet_3D(nn.Module):
    def __init__(self, nf=20, in_channel=2, no_linear=False):
        super(Unet_3D, self).__init__()
        self.nf = nf
        self.enc1 = Conv3d_block(in_channel, nf, 8, 2, 3)  # =>64
        self.enc2 = Conv3d_block(nf, 2 * nf, 4, 2, 1)  # =>32
        self.enc3 = Conv3d_block(2 * nf, 4 * nf, 4, 2, 1)  # =>16
        self.enc4 = Conv3d_block(4 * nf, 8 * nf, 4, 2, 1)  # =>8
        self.enc5 = Conv3d_block(8 * nf, 16 * nf, 4, 2, 1)  # =>4
        self.enc6 = Conv3d_block(16 * nf, 32 * nf, 4, 1, 0)  # =>1
        self.full_conv_block = nn.Sequential(
            nn.Linear(32 * nf, 32 * nf),
            nn.LeakyReLU(),
        )
        self.dec1 = Deconv3d_skip(32 * 2 * nf, 16 * nf, 4, 1, 0, 0)  # =>4
        self.dec2 = Deconv3d_skip(16 * 2 * nf, 8 * nf, 4, 2, 1, 0)  # =>8
        self.dec3 = Deconv3d_skip(8 * 2 * nf, 4 * nf, 4, 2, 1, 0)  # =>16
        self.dec4 = Deconv3d_skip(4 * 2 * nf, 2 * nf, 4, 2, 1, 0)  # =>32
        self.dec5 = Deconv3d_skip(4 * nf, nf, 8, 2, 3, 0)  # =>64
        self.dec6 = Deconv3d_skip(
            2 * nf, 1, 4, 2, 1, 0, is_activate=False)  # =>128
        self.no_linear = no_linear

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        # print(enc6.size())
        if not self.no_linear:
            flatten = enc6.view(enc6.size()[0], self.nf * 32)
            bottleneck = self.full_conv_block(flatten)
            bottleneck = bottleneck.view(enc6.size()[0], self.nf * 32, 1, 1, 1)
            dec1 = self.dec1(bottleneck, enc6)
        else:
            dec1 = self.dec1(enc6, enc6)
        dec2 = self.dec2(dec1, enc5)
        dec3 = self.dec3(dec2, enc4)
        dec4 = self.dec4(dec3, enc3)
        dec5 = self.dec5(dec4, enc2)
        out = self.dec6(dec5, enc1)
        return out


class Conv3d_block(nn.Module):
    def __init__(self, ncin, ncout, kernel_size, stride, pad, dropout=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ncin, ncout, kernel_size, stride, pad),
            nn.BatchNorm3d(ncout),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class Deconv3d_skip(nn.Module):
    def __init__(self, ncin, ncout, kernel_size, stride, pad, extra=0, is_activate=True):
        super(Deconv3d_skip, self).__init__()
        if is_activate:
            self.net = nn.Sequential(
                nn.ConvTranspose3d(ncin, ncout, kernel_size,
                                   stride, pad, extra),
                nn.BatchNorm3d(ncout),
                nn.LeakyReLU()
            )
        else:
            self.net = nn.ConvTranspose3d(
                ncin, ncout, kernel_size, stride, pad, extra)

    def forward(self, x, skip_in):
        y = cat((x, skip_in), dim=1)
        return self.net(y)


class ViewAsLinear(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


def relu():
    return nn.ReLU(inplace=True)


def relu_leaky():
    return nn.LeakyReLU(0.2, inplace=True)


def maxpool():
    return nn.MaxPool2d(3, stride=2, padding=0)


def dropout():
    return nn.Dropout(p=0.5, inplace=False)


def conv3d_half(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(
        n_ch_in, n_ch_out, 4, stride=2, padding=1, dilation=1, groups=1, bias=bias
    )


def deconv3d_2x(n_ch_in, n_ch_out, bias):
    return nn.ConvTranspose3d(
        n_ch_in, n_ch_out, 4, stride=2, padding=1, dilation=1, groups=1, bias=bias
    )


def conv3d_minus3(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(
        n_ch_in, n_ch_out, 4, stride=1, padding=0, dilation=1, groups=1, bias=bias
    )


def deconv3d_add3(n_ch_in, n_ch_out, bias):
    return nn.ConvTranspose3d(
        n_ch_in, n_ch_out, 4, stride=1, padding=0, dilation=1, groups=1, bias=bias
    )


def batchnorm1d(n_feat):
    return nn.BatchNorm1d(n_feat, eps=1e-5, momentum=0.1, affine=True)


def batchnorm(n_feat):
    return nn.BatchNorm2d(n_feat, eps=1e-5, momentum=0.1, affine=True)


def batchnorm3d(n_feat):
    return nn.BatchNorm3d(n_feat, eps=1e-5, momentum=0.1, affine=True)


def fc(n_in, n_out):
    return nn.Linear(n_in, n_out, bias=True)
