from __future__ import print_function, division
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from torch import nn
import os
import torch.nn.functional as F
from loss import DiceLoss, FocalTverskyLoss

from sklearn.metrics import f1_score
# from segmentation.models import all_models

from dataset import KaggleDataset
from architecture import UNetWithResnet50Encoder, UnetVGG16

from matplotlib.colors import ListedColormap

colors = ListedColormap([
    "#000000",
    "#FFFFFF",
    "#808080",
    "#0000FF",
    "#FF0000",
    "#00FF00",
    "#023020",
    "#ADD8E6",
    "#964B00",
    "#FFFF00",
    "#f5f5f5",
    "#A9A9A9",
    "#A020F0",
    "#8B8000",
    "#c4c4c4",
    "#8B8000",
    "#FFC0CB",
    "#00008B",
    "#FFCCCB",
    "#FFD700",
    "#AA6C39",
    "#023020",
    "#9cd3db",
    "#C4A484",
    "#90EE90"

])


# Ignore warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class config:
    BASE_PATH = "./msc-ai-2022/"
    TRAIN_IMG_PATH = BASE_PATH + "train_images/train_images"
    TRAIN_MASK_PATH = BASE_PATH + "train_masks/train_masks"
    TEST_IMG_PATH = BASE_PATH + "test_images/test_images"
    SAVE_PATH = "./models/scheduled_reduced_dice_vgg_model.pt"
    HEIGHT = 512
    WIDTH = 256
    BATCH_SIZE = 4
    IMAGE_PATH = './msc-ai-2022/train_images/train_images/'
    MASK_PATH = './msc-ai-2022/train_masks/train_masks/'
    BEST_SAVE_PATH = './models/scheduled_reduced_dice_vgg_best.pt'
    SAVE_CHECKPOINT = './models/scheduled_reduced_dice_vgg_checkpoint.pt'

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
        plt.imshow(masks[i], cmap=colors, vmin=0, vmax=25)

        plt.subplot(5, n_cols, counter)
        counter += 1
        plt.title("Pred")
        plt.axis("off")
        plt.imshow(preds[i], cmap=colors, vmin=0, vmax=25)

    plt.show()


def get_factor_lr(epoch: int):
    if epoch <= 5:
        return 1
    if epoch <= 10:
        return 0.9
    if epoch <= 15:
        return 0.7
    if epoch <= 25:
        return 0.3
    if epoch <= 95:
        return 0.15
    return 0.08


def train(model, epochs, optimizer, criterion, train_dataloader, val_dataloader, device, scheduler):

    passed_epochs = []
    best_val = -float("inf")
    total_train_loss = []
    total_val_loss = []
    total_train_acc = []
    total_val_acc = []
    for epoch in range(epochs):

        train_losses = []
        train_accuracy = []
        val_losses = []
        val_accuracy = []

        ###### Train model ######
        torch.cuda.empty_cache()
        model.train()
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
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
            preds = torch.argmax(output, dim=1).cpu()
            formatted_mask = torch.argmax(batch[1], dim=1)
            acc = f1_score(formatted_mask.view(-1),
                           preds.view(-1).detach().numpy(), average="micro")
            # acc = torch.sum(preds == formatted_mask).item(
            # ) / (formatted_mask.shape[0] * formatted_mask.shape[1] * formatted_mask.shape[2])
            # we divide by (batch_size * height * width * channels) to get average accuracy per pixel
            train_accuracy.append(acc)

        ###### Validate model ######
        model.eval()
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            # Extract data, labels
            img_batch, mask_batch = batch  # img [B,3,H,W], mask[B,H,W]
            img_batch, mask_batch = img_batch.to(device), mask_batch.to(device)

            output = model(img_batch)
            output = F.softmax(output, dim=1)
            loss = criterion(output, mask_batch)

            # Save batch results
            val_losses.append(loss.item())
            preds = torch.argmax(output, dim=1).cpu()
            formatted_mask = torch.argmax(batch[1], dim=1)
            acc = f1_score(formatted_mask.view(-1),
                           preds.view(-1).detach().numpy(), average="micro")
            # acc = torch.sum(preds == formatted_mask).item(
            # ) / (formatted_mask.shape[0] * formatted_mask.shape[1] * formatted_mask.shape[2])
            val_accuracy.append(acc)

        ##### Print epoch results ######
        print(
            f'TRAIN       Epoch: {epoch} | Epoch metrics | loss: {np.mean(train_losses):.4f}, accuracy: {np.mean(train_accuracy):.3f}')
        print(
            f'VALIDATION  Epoch: {epoch} | Epoch metrics | loss: {np.mean(val_losses):.4f}, accuracy: {np.mean(val_accuracy):.3f}')
        print(f"Learning rate for this epoch is: {scheduler.get_last_lr()[0]}")
        print('-' * 70)
        print('one example:')
        scheduler.step()
        batch = next(val_dataloader._get_iterator())
        preds = F.softmax(model(batch[0].to(device)), dim=1)
        preds = torch.argmax(preds, dim=1)
        # preds = torch.where(preds > 0.5, 1, 0)
        if np.mean(val_accuracy) >= best_val:
            torch.save(model.state_dict(), config.BEST_SAVE_PATH)
            best_val = np.mean(val_accuracy)
        else:
            torch.save(model.state_dict(), config.SAVE_CHECKPOINT)
        total_val_acc.append(np.mean(val_accuracy).item())
        total_val_loss.append(np.mean(val_losses).item())
        total_train_acc.append(np.mean(train_accuracy).item())
        total_train_loss.append(np.mean(train_losses).item())
        fig, axes = plt.subplots(figsize=(15, 10), ncols=2)
        passed_epochs.append(epoch)
        axes[0].plot(passed_epochs, total_val_loss, label="val_loss")
        axes[0].plot(passed_epochs, total_train_loss, label="train_loss")
        axes[0].set_title("Losses")
        axes[0].legend()
        axes[1].plot(passed_epochs, total_train_acc, label="train_acc")
        axes[1].plot(passed_epochs, total_val_acc, label="val_acc")
        axes[1].set_title("F1-score")
        axes[1].legend()
        plt.savefig('plot.png')
        # plot_img_masks(batch[0], batch[1],
        #                preds.cpu().detach().numpy(), idx_class)
        # plot_img_masks(batch[0][0], preds.cpu().detach().numpy(), idx_class)
        print('-' * 70)


if __name__ == "__main__":
    # Very simple train/test split
    train_ratio = 0.8
    # train_set_last_idx = int(
    #     len(glob.glob(config.TRAIN_IMG_PATH + "/*")) * train_ratio)

    img_names = [name.split(".")[0]
                 for name in os.listdir(config.TRAIN_IMG_PATH)]
    n_train = int(train_ratio * len(img_names))
    train_names = random.sample(img_names, n_train)
    val_names = [name for name in img_names if name not in train_names]

    train_img_paths = [
        os.path.join(config.TRAIN_IMG_PATH, name + ".jpg")
        for name in train_names
    ]
    train_mask_paths = [
        os.path.join(config.TRAIN_MASK_PATH, name + ".png")
        for name in train_names
    ]
    val_img_paths = [
        os.path.join(config.TRAIN_IMG_PATH, name + ".jpg")
        for name in val_names
    ]
    val_mask_paths = [
        os.path.join(config.TRAIN_MASK_PATH, name + ".png")
        for name in val_names
    ]

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
    epochs = 200
    lr = 0.007
    # model = UNetWithResnet50Encoder(n_classes=len(idx_class.keys())).to(device)
    # model_name = "fcn8_vgg16"
    n_classes = len(idx_class.keys())
    model = UnetVGG16(25).to(device)
    # model = all_models.model_from_name[model_name](n_classes, BATCH_SIZE,
    #                                                pretrained=True,
    #                                                fixed_feature=False).to(device)
    # PATH = "./models/vgg_best.pt"
    # model.load_state_dict(torch.load(PATH))
    # print("model loaded!")
    if True:  # fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.SGD(params_to_update, lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_factor_lr)

    else:
        optimizer = torch.optim.Adadelta(model.parameters())
    criterion = DiceLoss()
    # criterion = FocalTverskyLoss()

    train(model, epochs, optimizer, criterion,
          train_dataloader, val_dataloader, device, scheduler)

    torch.save(model.state_dict(), config.SAVE_PATH)
