import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random

HEIGHT = 256
WIDTH = 256


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

        if self.train:  # Random crop over here
            p = random.random()
            if p > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                angle = transforms.RandomRotation.get_params([-5, 5])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
                # i, j, h, w = transforms.RandomCrop.get_params(
                #     img, output_size=(HEIGHT, WIDTH))
                # img = TF.crop(img, i, j, h, w)
                # mask = TF.crop(mask, i, j, h, w)

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
