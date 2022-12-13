from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image
from tqdm import tqdm
import time
import glob
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

HEIGHT = 256
WIDTH = 256
BATCH_SIZE = 5
IMAGE_PATH = './msc-ai-2022/train_images/train_images/'
MASK_PATH = './msc-ai-2022/train_masks/train_masks/'
BEST_SAVE_PATH = './models/binary_entropy.pt'
SAVE_CHECKPOINT = './models/checkpoint.pt'


# PyTorch
ALPHA = 0.5
BETA = 0.5
GAMMA = 1


class config:
    BASE_PATH = "./msc-ai-2022/"
    TRAIN_IMG_PATH = BASE_PATH + "train_images/train_images"
    TRAIN_MASK_PATH = BASE_PATH + "train_masks/train_masks"
    TEST_IMG_PATH = BASE_PATH + "test_images/test_images"
    SAVE_PATH = "./models/resnet50.pt"

# Create a custom Dataset class


class KaggleDataset(Dataset):
    def __init__(self, images_paths: str, mask_paths: list, train: bool):
        self.image_paths = images_paths
        # self.mask_paths = [os.path.join(mask_folder, path) for path in os.listdir(mask_folder)]
        self.mask_paths = mask_paths
        self.train = train

        self.transform_img = transforms.Compose([
            transforms.Resize([HEIGHT, WIDTH]),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(
                [HEIGHT, WIDTH], interpolation=transforms.InterpolationMode.NEAREST),  # Important
            transforms.ToTensor()
        ])

    def __getitem__(self, index):

        # Select a specific image's path
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Load the image
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # if self.train: # Random crop over here
        #     i, j, h, w = transforms.RandomCrop.get_params(
        #         img, output_size=(HEIGHT, WIDTH))
        #     img = TF.crop(img, i, j, h, w)
        #     mask = TF.crop(mask, i, j, h, w)

        # Apply transformations
        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        # Scale the mask from 0-1 range to 0-255 range
        mask = mask * 255
        # mask = mask.squeeze(0)
        # mask = torch.squeeze(mask, 0)

        maps = torch.zeros((25, 256, 256))
        for i in range(25):
            indices = torch.where(mask == i)
            current_map = torch.zeros_like(mask)
            current_map[indices] = 1.0
            maps[i] = current_map

        return img, maps.type(torch.FloatTensor)

    def __len__(self):
        return len(self.image_paths)


def plot_img_masks(img, masks):
    counter = 1
    plt.figure(figsize=(15, 15))
    plt.subplot(5, 6, counter)
    plt.axis("off")
    plt.title("image")
    plt.imshow(img.permute(1, 2, 0))
    counter += 1
    for i in range(masks.shape[0]):
        plt.subplot(5, 6, counter)
        plt.axis("off")
        plt.title(idx_class[i])
        plt.imshow(masks[i])
        counter += 1
    plt.show()


resnet = torchvision.models.resnet.resnet50(pretrained=True)


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
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky


def train(model, epochs, optimizer, criterion):

    for epoch in range(epochs):

        train_losses = []
        train_accuracy = []
        val_losses = []
        val_accuracy = []
        best_val = -float("inf")

        ###### Train model ######
        torch.cuda.empty_cache()
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # Extract images and masks
            img_batch, mask_batch = batch  # img [B,3,H,W], mask[B,H,W]
            img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)

            # Optimize network
            optimizer.zero_grad()
            output = model(img_batch)  # output: [B, 25, H, W]
            output = F.sigmoid(output)
            loss = criterion(output, mask_batch)
            loss.backward()
            optimizer.step()

            # Save batch results
            train_losses.append(loss.item())
            preds = torch.where(output > 0.5, 1, 0)
            acc = torch.sum(preds == mask_batch).item(
            ) / (mask_batch.shape[0] * mask_batch.shape[1] * mask_batch.shape[2] * mask_batch.shape[3])
            # we divide by (batch_size * height * width) to get average accuracy per pixel
            train_accuracy.append(acc)

        ###### Validate model ######
        model.eval()
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            # Extract data, labels
            img_batch, mask_batch = batch  # img [B,3,H,W], mask[B,H,W]
            img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)

            # Validate model
            output = model(img_batch)
            output = F.sigmoid(output)
            loss = criterion(output, mask_batch)

            # Save batch results
            val_losses.append(loss.item())
            preds = torch.where(output > 0.5, 1, 0)
            acc = torch.sum(preds == mask_batch).item(
            ) / (mask_batch.shape[0] * mask_batch.shape[1] * mask_batch.shape[2] * mask_batch.shape[3])
            val_accuracy.append(acc)

        ##### Print epoch results ######
        print(
            f'TRAIN       Epoch: {epoch} | Epoch metrics | loss: {np.mean(train_losses):.4f}, accuracy: {np.mean(train_accuracy):.3f}')
        print(
            f'VALIDATION  Epoch: {epoch} | Epoch metrics | loss: {np.mean(val_losses):.4f}, accuracy: {np.mean(val_accuracy):.3f}')
        print('-' * 70)
        print('one example:')
        preds = F.sigmoid(model(batch[0].to(device)))[0]
        preds = torch.where(preds > 0.5, 1, 0)
        if np.mean(val_accuracy) >= best_val:
            torch.save(model.state_dict(), BEST_SAVE_PATH)
            best_val = np.mean(val_accuracy)
        else:
            torch.save(model.state_dict(), SAVE_CHECKPOINT)
        # plot_img_masks(batch[0][0], batch[1][0])
        # plot_img_masks(batch[0][0], preds.cpu().detach().numpy())
        print('-' * 70)


if __name__ == "__main__":
    # Very simple train/test split
    train_ratio = 0.8
    train_set_last_idx = int(
        len(glob.glob(config.TRAIN_IMG_PATH + "/*")) * train_ratio)

    train_img_paths = sorted(
        glob.glob(config.TRAIN_IMG_PATH + "/*"))[:train_set_last_idx]
    train_mask_paths = sorted(
        glob.glob(config.TRAIN_MASK_PATH + "/*"))[:train_set_last_idx]
    val_img_paths = sorted(
        glob.glob(config.TRAIN_IMG_PATH + "/*"))[train_set_last_idx:]
    val_mask_paths = sorted(
        glob.glob(config.TRAIN_MASK_PATH + "/*"))[train_set_last_idx:]

    # Create datasets
    train_dataset = KaggleDataset(
        train_img_paths, train_mask_paths, train=True)
    val_dataset = KaggleDataset(val_img_paths, val_mask_paths, train=False)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    idx_class = {
        0: "Background",
        1: "Property Roof",
        2: "Secondary Structure",
        3: "Swimming Pool",
        4: "Vehicle",
        5: "Grass",
        6: "Trees / Shrubs",
        7: "Solar Panels",
        8: "Chimney",
        9: "Street Light",
        10: "Window",
        11: "Satellite Antenna",
        12: "Garbage Bins",
        13: "Trampoline",
        14: "Road/Highway",
        15: "Under Construction / In Progress Status",
        16: "Power Lines & Cables",
        17: "Water Tank / Oil Tank",
        18: "Parking Area - Commercial",
        19: "Sports Complex / Arena",
        20: "Industrial Site",
        21: "Dense Vegetation / Forest",
        22: "Water Body",
        23: "Flooded",
        24: "Boat",
    }

    # start = time.time()
    # for batch in tqdm(train_dataloader):
    #     img_batch, img_mask = batch

    # for batch in tqdm(val_dataloader):
    #     img_batch, img_mask = batch

    # end = time.time()
    # print(f'Seconds needed to load one train + val epoch: {end - start :.3f}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else device)
    print('Using device:', device)
    epochs = 15
    lr = 0.0009
    model = UNetWithResnet50Encoder(n_classes=len(idx_class.keys())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalTverskyLoss()

    train(model, epochs, optimizer, criterion)

    torch.save(model.state_dict(), config.SAVE_PATH)
