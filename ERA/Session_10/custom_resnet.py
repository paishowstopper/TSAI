import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomResNet(nn.Module):
    def __init__(self, dropout_value=0.01):
        super(CustomResNet, self).__init__()
        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.resblock1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.maxpoollayer = nn.Sequential(
            nn.MaxPool2d(kernel_size = 4, stride = 4)
            )
        self.fclayer = nn.Sequential(
            nn.Linear(512,10)
            )
        
    def forward(self, x):
        x = self.preplayer(x)
        x = self.layer1(x)
        r1 = self.resblock1(x)
        x = x + r1
        x = self.layer2(x)
        x = self.layer3(x)
        r2 = self.resblock2(x)
        x = x + r2
        x = self.maxpoollayer(x)
        x = x.view((x.shape[0],-1))
        x = self.fclayer(x)
        
        return x
