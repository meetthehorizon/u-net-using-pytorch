import os
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LungDataset(Dataset):
    def __init__(self, path="data/train", transform=None, num=None):
        """Lung segmentation dataset.

        Parameters
        ----------
        image_path : str
                Path to images directory
        mask_path : str
                Path to masks directory
        transform : callable
                Transformation to apply to image and mask
        """
        image_dir = path + "/image/"
        mask_dir = path + "/mask/"
        self.image_paths = [
            image_dir + f for f in os.listdir(image_dir) if f.endswith(".png")
        ]
        self.mask_paths = [
            mask_dir + f for f in os.listdir(mask_dir) if f.endswith(".jpeg")
        ]

        if num is not None:
            self.image_paths = self.image_paths[:num]
            self.mask_paths = self.mask_paths[:num]

        self.transform = transform

    def __getitem__(self, index):
        """Load image and mask at index."""
        # print(self.image_paths)
        if index >= len(self) or index < 0:
            raise IndexError("Index out of range")

        image, mask = Image.open(self.image_paths[index]).convert("L"), Image.open(
            self.mask_paths[index]
        ).convert("L")

        if self.transform:
            image, mask = self.transform(image), self.transform(mask)

        return image, mask

    def __len__(self):
        """Return length of dataset."""
        return len(self.image_paths)


if __name__ == "__main__":
    print("passed")
