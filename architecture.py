import torchvision.models as models
import torch.nn as nn
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

resnet = torchvision.models.resnet.resnet50(weights=True)


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        # x = F.softmax(x, dim=1)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x


class UnetVGG16(nn.Module):
    def __init__(self, num_classes):
        super(UnetVGG16, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)

        self.encoder1 = nn.Sequential(*self.vgg16.features[:5])
        self.encoder2 = nn.Sequential(*self.vgg16.features[5:10])
        self.encoder3 = nn.Sequential(*self.vgg16.features[10:17])
        self.encoder4 = nn.Sequential(*self.vgg16.features[17:24])
        self.encoder5 = nn.Sequential(*self.vgg16.features[24:])

        self.conv1 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.decoder5 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(512, 256, 3, 1, 1)
        self.decoder4 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(256, 128, 3, 1, 1)
        self.decoder3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.decoder2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv5 = nn.Conv2d(28, 64, 3, 1, 1)
        self.decoder1 = nn.ConvTranspose2d(
            64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.final = nn.Conv2d(28, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder part
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        # Decoder part
        x6 = self.decoder5(x5)
        x6 = torch.cat((x6, x4), dim=1)
        x6 = self.conv1(x6)
        x7 = self.decoder4(x6)
        x7 = torch.cat((x7, x3), dim=1)
        x7 = self.conv2(x7)
        x8 = self.decoder3(x7)
        x8 = torch.cat((x8, x2), dim=1)
        x8 = self.conv3(x8)
        x9 = self.decoder2(x8)
        x9 = torch.cat((x9, x1), dim=1)
        x9 = self.conv4(x9)
        x10 = self.decoder1(x9)
        # x12 = torch.cat((x10, x), dim=1)
        # x12 = self.conv5(x12)

        # Final part
        # x11 = self.final(10)
        return x10
