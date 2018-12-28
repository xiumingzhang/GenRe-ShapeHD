"""
This is an implementation of a U-Net using ResNet-18 blocks
"""
import torch
from torch import nn
from torchvision.models import resnet18


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        output_padding=output_padding
    )


class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(RevBasicBlock, self).__init__()
        self.deconv1 = deconv3x3(inplanes, planes, stride=1)
        # Note that in ResNet, the stride is on the second layer
        # Here we put it on the first layer as the mirrored block
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = deconv3x3(planes, planes, stride=stride,
                                 output_padding=1 if stride > 1 else 0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out


class RevBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(RevBottleneck, self).__init__()
        bottleneck_planes = int(inplanes / 4)
        self.deconv1 = nn.ConvTranspose2d(
            inplanes,
            bottleneck_planes,
            kernel_size=1,
            bias=False,
            stride=1
        ) # conv and deconv are the same when kernel size is 1
        self.bn1 = nn.BatchNorm2d(bottleneck_planes)
        self.deconv2 = nn.ConvTranspose2d(
            bottleneck_planes,
            bottleneck_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_planes)
        self.deconv3 = nn.ConvTranspose2d(
            bottleneck_planes,
            planes,
            kernel_size=1,
            bias=False,
            stride=stride,
            output_padding=1 if stride > 0 else 0
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.deconv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out


class RevResNet(nn.Module):
    def __init__(self, block, layers, planes, inplanes=None, out_planes=5):
        """
        planes: # output channels for each block
        inplanes: # input channels for the input at each layer
            If missing, it will be inferred.
        """
        if inplanes is None:
            inplanes = [512]
        self.inplanes = inplanes[0]
        super(RevResNet, self).__init__()
        inplanes_after_blocks = inplanes[4] if len(inplanes) > 4 else planes[3]
        self.deconv1 = nn.ConvTranspose2d(
            inplanes_after_blocks,
            planes[3],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            planes[3],
            out_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(planes[3])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes[0], layers[0], stride=2)
        if len(inplanes) > 1:
            self.inplanes = inplanes[1]
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        if len(inplanes) > 2:
            self.inplanes = inplanes[2]
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        if len(inplanes) > 3:
            self.inplanes = inplanes[3]
        self.layer4 = self._make_layer(block, planes[3], layers[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    output_padding=1 if stride > 1 else 0
                ),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x


def revresnet18(**kwargs):
    model = RevResNet(
        RevBasicBlock,
        [2, 2, 2, 2],
        [512, 256, 128, 64],
        **kwargs
    )
    return model


def revuresnet18(**kwargs):
    """
    Reverse ResNet-18 compatible with the U-Net setting
    """
    model = RevResNet(
        RevBasicBlock,
        [2, 2, 2, 2],
        [256, 128, 64, 64],
        inplanes=[512, 512, 256, 128, 128],
        **kwargs
    )
    return model


def _num_parameters(net):
    return sum([
        x.numel() for x in list(net.parameters())
    ])


def main():
    net = resnet18()
    revnet = revresnet18()
    net.avgpool = nn.AvgPool2d(kernel_size=8)
    for name, mod in net.named_children():
        mod.__name = name
        mod.register_forward_hook(
            lambda mod, input, output: print(mod.__name, output.shape)
        )
    for name, mod in revnet.named_children():
        mod.__name = name
        mod.register_forward_hook(
            lambda mod, input, output: print(mod.__name, output.shape)
        )
    # print(net)
    print('resnet', _num_parameters(net))
    net(torch.zeros(2, 3, 256, 256))
    print('')
    print('revresnet', _num_parameters(revnet))
    # print(revnet)
    revnet(torch.zeros(2, 512, 8, 8))
    print('')
    revunet = RevResNet(RevBasicBlock, [2, 2, 2, 2], [512, 512, 256, 128])
    print('revunet', _num_parameters(revunet))


if __name__ == '__main__':
    main()
