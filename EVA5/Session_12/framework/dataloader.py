import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
from tinyimagenet import TinyImageNet, TINLoad, download_images, class_names

class Loader():
    
    def __init__(self, train_transform, test_transform, batch_size=128, num_workers=4):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform

    def CIFAR10Load(self):

        self.dataloader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.training_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.train_transform)
        self.trainloader = DataLoader(self.training_set, shuffle=True, **self.dataloader_args)
        
        self.test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.test_transform)
        self.testloader = DataLoader(self.test_set, shuffle=True, **self.dataloader_args)
        
        return self.trainloader, self.testloader, self.classes

    
    def TinyImageNetLoad(self):
        
        download_images()
        self.dataloader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.classes = class_names()
        print(self.classes)

        dataset = TinyImageNet(self.classes)
        train_split = len(dataset) * 70 // 100
        test_split = len(dataset) - train_split
        train_data, test_data = random_split(dataset, [train_split, test_split])

        self.training_set = TINLoad(train_data, self.train_transform)
        self.trainloader = DataLoader(self.training_set, **self.dataloader_args)
        
        self.test_set = TINLoad(test_data, self.test_transform)
        self.testloader = DataLoader(self.test_set, **self.dataloader_args)

        return self.trainloader, self.testloader, self.classes