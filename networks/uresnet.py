import torch
from torch import nn
from networks.revresnet import revuresnet18, resnet18


class Net(nn.Module):
    """
    Used for RGB to 2.5D maps
    """

    def __init__(self, out_planes, layer_names, input_planes=3):
        super().__init__()

        # Encoder
        module_list = list()
        resnet = resnet18(pretrained=True)
        in_conv = nn.Conv2d(input_planes, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        module_list.append(
            nn.Sequential(
                resnet.conv1 if input_planes == 3 else in_conv,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        )
        module_list.append(resnet.layer1)
        module_list.append(resnet.layer2)
        module_list.append(resnet.layer3)
        module_list.append(resnet.layer4)
        self.encoder = nn.ModuleList(module_list)
        self.encoder_out = None

        # Decoder
        self.decoders = {}
        for out_plane, layer_name in zip(out_planes, layer_names):
            module_list = list()
            revresnet = revuresnet18(out_planes=out_plane)
            module_list.append(revresnet.layer1)
            module_list.append(revresnet.layer2)
            module_list.append(revresnet.layer3)
            module_list.append(revresnet.layer4)
            module_list.append(
                nn.Sequential(
                    revresnet.deconv1,
                    revresnet.bn1,
                    revresnet.relu,
                    revresnet.deconv2
                )
            )
            module_list = nn.ModuleList(module_list)
            setattr(self, 'decoder_' + layer_name, module_list)
            self.decoders[layer_name] = module_list

    def forward(self, im):
        # Encode
        feat = im
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
        self.encoder_out = feat_maps[-1]
        # Decode
        outputs = {}
        for layer_name, decoder in self.decoders.items():
            x = feat_maps[-1]
            for idx, f in enumerate(decoder):
                x = f(x)
                if idx < len(decoder) - 1:
                    feat_map = feat_maps[-(idx + 2)]
                    assert feat_map.shape[2:4] == x.shape[2:4]
                    x = torch.cat((x, feat_map), dim=1)
            outputs[layer_name] = x
        return outputs


class Net_inpaint(nn.Module):
    """
    Used for RGB to 2.5D maps
    """

    def __init__(self, out_planes, layer_names, input_planes=3):
        super().__init__()

        # Encoder
        module_list = list()
        resnet = resnet18(pretrained=True)
        in_conv = nn.Conv2d(input_planes, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        module_list.append(
            nn.Sequential(
                resnet.conv1 if input_planes == 3 else in_conv,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        )
        module_list.append(resnet.layer1)
        module_list.append(resnet.layer2)
        module_list.append(resnet.layer3)
        module_list.append(resnet.layer4)
        self.encoder = nn.ModuleList(module_list)
        self.encoder_out = None
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=8, stride=2, padding=3, bias=False, output_padding=0)
        # Decoder
        self.decoders = {}
        for out_plane, layer_name in zip(out_planes, layer_names):
            module_list = list()
            revresnet = revuresnet18(out_planes=out_plane)
            module_list.append(revresnet.layer1)
            module_list.append(revresnet.layer2)
            module_list.append(revresnet.layer3)
            module_list.append(revresnet.layer4)
            module_list.append(
                nn.Sequential(
                    revresnet.deconv1,
                    revresnet.bn1,
                    revresnet.relu,
                    self.deconv2
                )
            )
            module_list = nn.ModuleList(module_list)
            setattr(self, 'decoder_' + layer_name, module_list)
            self.decoders[layer_name] = module_list

    def forward(self, im):
        # Encode
        feat = im
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
        self.encoder_out = feat_maps[-1]
        # Decode
        outputs = {}
        for layer_name, decoder in self.decoders.items():
            x = feat_maps[-1]
            for idx, f in enumerate(decoder):
                x = f(x)
                if idx < len(decoder) - 1:
                    feat_map = feat_maps[-(idx + 2)]
                    assert feat_map.shape[2:4] == x.shape[2:4]
                    x = torch.cat((x, feat_map), dim=1)
            outputs[layer_name] = x
        return outputs
