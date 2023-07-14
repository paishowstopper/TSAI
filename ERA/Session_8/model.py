import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# S8 Models
class S8Model1(nn.Module):
    def __init__(self, norm='bn'):

        dropout_value = 0.02
        GROUP_SIZE = 2

        super(S8Model1, self).__init__()

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, 16)

        self.C2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, 16)

        self.c3t = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n3t = nn.BatchNorm2d(8)
        elif norm == 'gn':
            self.n3t = nn.GroupNorm(GROUP_SIZE, 8)
        elif norm == 'ln':
            self.n3t = nn.GroupNorm(1, 8)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.C3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, 16)


        self.C4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, 16)

        self.C5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, 16)

        self.c6 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n6t = nn.BatchNorm2d(8)
        elif norm == 'gn':
            self.n6t = nn.GroupNorm(GROUP_SIZE, 8)
        elif norm == 'ln':
            self.n6t = nn.GroupNorm(1, 8)

        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.C7 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, 16)

        self.C8 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value))

        if norm == 'bn':
            self.n7 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n7 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n7 = nn.GroupNorm(1, 16)

        self.C9 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))
        self.C10 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        x = self.n1(self.C1(x))
        x = self.n2(self.C2(x))
        x = self.pool1(self.n3t(self.c3t(x)))
        skip1 = self.n3(self.C3(x))
        x = self.n4(self.C4(skip1))
        x = self.n5(self.C5(x) + skip1)
        x = self.n6t(self.pool2(self.c6(x)))
        skip2 = self.n6(self.C7(x))
        x = self.n7(self.C8(skip2))
        x = self.C9(x) + skip2
        x = self.gap(x)
        x = self.C10(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x




# S7 Models

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1, stride=1, bias=False))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 11, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(11, 11, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(11, 14, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(14, 14, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, stride=1, bias=False),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, stride=1, bias=False))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 11, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(11, 11, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(11, 14, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(14, 14, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, stride=1, bias=False))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x

class Model4(nn.Module):
    def __init__(self):
        dropout_value = 0.01
        super(Model4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 11, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU(),
            nn.Dropout(dropout_value))
        self.layer2 = nn.Sequential(
            nn.Conv2d(11, 11, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(11),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(11, 14, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(dropout_value))
        self.layer4 = nn.Sequential(
            nn.Conv2d(14, 14, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value))
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, stride=1, bias=False))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.gap(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)

        return x




# S6 Models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
                        nn.BatchNorm2d(8),
                        nn.ReLU(inplace=True)
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
                        nn.BatchNorm2d(8),
                        nn.ReLU(inplace=True)
                        )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
                        nn.BatchNorm2d(8),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True)
                        )
        self.conv5 = nn.Sequential(
                        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                        nn.BatchNorm2d(16) ,
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.conv6 = nn.Sequential(
                        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True)
                        )
        self.conv7 = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                        )
        self.conv8 = nn.Conv2d(32,10,1)
        self.av = nn.AvgPool2d(3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.av(x)
        x = x.view((x.shape[0],-1))
        x = F.log_softmax(x, dim=1)
        return x
