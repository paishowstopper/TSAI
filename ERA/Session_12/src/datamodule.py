import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
        self.classes = dataset.classes

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = np.array(image)
        transformed = self.transforms(image=image)
        image = transformed['image']

        return image, target

    def __len__(self):
        return len(self.dataset)


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../data", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
      
    def setup(self, stage=None):
        self.train_transforms = A.Compose([
            A.PadIfNeeded(min_height=32 + 4, min_width=32 + 4, p=1),
            A.RandomCrop(32, 32),
            A.HorizontalFlip(),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(0.49139968, 0.48215827, 0.44653124), p=1),  # Apply coarse dropout
            A.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),  # Normalize the image
            ToTensorV2() # Convert image to a PyTorch tensor
        ])
        self.test_transforms = A.Compose([
            A.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
            ToTensorV2()
        ])

        # Assign train/val datasets 
        if stage == "fit" or stage is None:
            self.train_data = AlbumentationsDataset(datasets.CIFAR10(self.data_dir, train=True),  self.train_transforms)
            self.val_data = AlbumentationsDataset(datasets.CIFAR10(self.data_dir, train=False), self.test_transforms)

        # Assign test dataset
        if stage == "test" or stage is None:
            self.test_data = AlbumentationsDataset(datasets.CIFAR10(self.data_dir, train=False), self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()
