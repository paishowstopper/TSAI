import os
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


NUM_WORKERS = os.cpu_count()

class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
        self.classes = dataset.classes

    def __getitem__(self, index):
        image, target = self.dataset[index]

        image = np.array(image)

        # Apply Albumentations transforms
        transformed = self.transforms(image=image)
        image = transformed['image']

        return image, target

    def __len__(self):
        return len(self.dataset)

def create_dataloaders(train_dir, test_dir, train_transforms, test_transforms, batch_size, num_workers=NUM_WORKERS):

  # Download the dataset
  train_data = datasets.CIFAR10(train_dir, train=True, download=True)
  test_data = datasets.CIFAR10(test_dir, train=False, download=True)

  # Wrap the datasets with AlbumentationsDataset
  train_data = AlbumentationsDataset(train_data, train_transforms)
  test_data = AlbumentationsDataset(test_data, test_transforms)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
