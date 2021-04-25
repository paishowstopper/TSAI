import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader

class Loader():
    
    def __init__(self, transform=None, root='./data', batch_size=128, num_workers=2, train=True, download=True, shuffle=True):

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.download = download
        self.shuffle = shuffle
        self.transform = transform

    def CIFAR10Load(self):

        self.dataloader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.training_set = datasets.CIFAR10(root=self.root, train=self.train, download=self.download, transform=self.transform)
        self.trainloader = DataLoader(self.training_set, shuffle=self.shuffle, **self.dataloader_args)
        
        self.test_set = datasets.CIFAR10(root=self.root, train=(not self.train), download=self.download, transform=self.transform)
        self.testloader = DataLoader(self.test_set, shuffle=(not self.shuffle), **self.dataloader_args)
        
        return self.trainloader, self.testloader, self.classes