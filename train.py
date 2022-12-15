from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from torch import nn
import torch.nn.functional as F

from dataset import KaggleDataset
from architecture import UNetWithResnet50Encoder


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


def plot_img_masks(imgs, masks, preds, idx_class):
    batch_size = imgs.shape[0]
    n_cols = 3
    plt.figure(figsize=(15, 15))
    masks = torch.argmax(masks, dim=1)
    counter = 1
    for i in range(batch_size):
        plt.subplot(5, n_cols, counter)
        counter += 1
        plt.title("Image")
        plt.axis("off")
        plt.imshow(imgs[i].permute(1, 2, 0))

        plt.subplot(5, n_cols, counter)
        counter += 1
        plt.title("Mask")
        plt.axis("off")
        plt.imshow(masks[i])

        plt.subplot(5, n_cols, counter)
        counter += 1
        plt.title("Pred")
        plt.axis("off")
        plt.imshow(preds[i])

    plt.show()


def train(model, epochs, optimizer, criterion, train_dataloader, val_dataloader, device, idx_class):

    best_val = -float("inf")
    for epoch in range(epochs):

        train_losses = []
        train_accuracy = []
        val_losses = []
        val_accuracy = []

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
            output = F.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)  # output: [B,H,W]
            loss = criterion(output, mask_batch)
            loss.backward()
            optimizer.step()

            # Save batch results
            train_losses.append(loss.item())
            preds = torch.argmax(output, dim=1)
            formatted_mask = torch.argmax(mask_batch, dim=1)
            acc = torch.sum(preds == formatted_mask).item(
            ) / (formatted_mask.shape[0] * formatted_mask.shape[1] * formatted_mask.shape[2])
            # we divide by (batch_size * height * width * channels) to get average accuracy per pixel
            train_accuracy.append(acc)

        ###### Validate model ######
        model.eval()
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            # Extract data, labels
            img_batch, mask_batch = batch  # img [B,3,H,W], mask[B,H,W]
            img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)

            # Validate model
            output = model(img_batch)
            output = F.softmax(output, dim=1)
            loss = criterion(output, mask_batch)

            # Save batch results
            val_losses.append(loss.item())
            preds = torch.argmax(output, dim=1)
            formatted_mask = torch.argmax(mask_batch, dim=1)
            acc = torch.sum(preds == formatted_mask).item(
            ) / (formatted_mask.shape[0] * formatted_mask.shape[1] * formatted_mask.shape[2])
            val_accuracy.append(acc)

        ##### Print epoch results ######
        print(
            f'TRAIN       Epoch: {epoch} | Epoch metrics | loss: {np.mean(train_losses):.4f}, accuracy: {np.mean(train_accuracy):.3f}')
        print(
            f'VALIDATION  Epoch: {epoch} | Epoch metrics | loss: {np.mean(val_losses):.4f}, accuracy: {np.mean(val_accuracy):.3f}')
        print('-' * 70)
        print('one example:')
        preds = F.softmax(model(batch[0].to(device)), dim=1)
        preds = torch.argmax(preds, dim=1)
        # preds = torch.where(preds > 0.5, 1, 0)
        if np.mean(val_accuracy) >= best_val:
            torch.save(model.state_dict(), BEST_SAVE_PATH)
            best_val = np.mean(val_accuracy)
        else:
            torch.save(model.state_dict(), SAVE_CHECKPOINT)
        plot_img_masks(batch[0], batch[1],
                       preds.cpu().detach().numpy(), idx_class)
        # plot_img_masks(batch[0][0], preds.cpu().detach().numpy(), idx_class)
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
