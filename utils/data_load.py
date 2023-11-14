import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class DroneDataset(Dataset):
    def __init__(self, root='../data/train/', transform=None):
        self.root = root
        self.transform = transform
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images/"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks/"))))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)
    
if __name__ == '__main__':
    print('passed')