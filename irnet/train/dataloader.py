import torch
import imageio as io
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

def transforms(array):
    if len(array.shape) == 2:
        array = array[..., np.newaxis]
        img = array.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
    elif len(array.shape) == 3:
        img = array.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
    return img

def color_transforms(array):
    img = array.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

class HDRDataset(Dataset):
    def __init__(self, input_dir='/home/jay96/Desktop/256-eevee-images/train/luminance', gt_dir='/home/jay96/Desktop/256-eevee-images/train/albedo', transforms=transforms, channels=1):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transforms
        self.channels = channels
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])

        # Load hdr images using imageio
        if self.channels == 3:
            input_image = io.imread(input_image_path)
        elif self.channels == 1:
            input_image = io.imread(input_image_path)[:, :, 0]

        gt_image = io.imread(gt_image_path)[:, :, 0]

        # Apply any transformations if needed
        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image

