import torch
import torch.nn as nn
import torch.nn.functional as F

class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()

        #x1 = Input
        self.tran1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        #x2 = Conv(x1)
        # Convolution Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        #x3 = Conv(x1 + x2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ) 

        #x4 = MaxPooling(x1 + x2 + x3)
        self.Pool1 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2),stride=2)
            nn.MaxPool2d(2, 2)
        ) 

        self.tran2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        #x5 = Conv(x4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) 

        #x6 = Conv(x4 + x5)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) 

        # x7 = Conv(x4 + x5 + x6)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) 

        # x8 = MaxPooling(x5 + x6 + x7)
        self.Pool2 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2),stride=2),
            nn.MaxPool2d(2, 2)
        ) 

        self.tran3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # x9 = Conv(x8)
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) 

        # x10 = Conv (x8 + x9)
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) 

        # x11 = Conv (x8 + x9 + x10)
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) 

        # x12 = GAP(x11)
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        )
         
        # x13 = FC(x12)
        self.fc = nn.Sequential(
            nn.Linear(128, 10, bias=False)
        ) 

    def forward(self, x):

        x1 = self.tran1(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x1 + x2)
        y4 = self.Pool1(x1 + x2 + x3)
        x4 = self.tran2(y4)
        x5 = self.conv3(y4)
        x6 = self.conv4(x4 + x5)
        x7 = self.conv5(x4 + x5 + x6)
        y8 = self.Pool2(x5 + x6 + x7)
        x8 = self.tran3(y8)
        x9 = self.conv6(y8)
        x10 =self.conv7 (x8 + x9)
        x11 =self.conv8 (x8 + x9 + x10)
        x12 =self.gap(x11)
        x12 = x12.view(-1, 128)
        x13 =self.fc(x12)
        return F.log_softmax(x13, dim=1)