import torch.nn as nn
import torch.nn.functional as F

#Depthwise Separable and Dilated convolution
class AdvancedConvolutionNet(nn.Module):
    def __init__(self):
        super(AdvancedConvolutionNet, self).__init__()

        #Referred the below link for depthwise separable convolution implementation
        #https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/6

        #Depthwise Separable Convolution
        self.conv1 = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), padding=1, dilation=1, bias=False, groups=3),
          nn.Conv2d(in_channels=3, out_channels=26, kernel_size=(1,1)),
          nn.ReLU(),
          nn.BatchNorm2d(26),
          nn.Conv2d(in_channels=26, out_channels=26, kernel_size=(3,3), padding=1, dilation=1, bias=False, groups=26),
          nn.Conv2d(in_channels=26, out_channels=52, kernel_size=(1,1)),
          nn.ReLU(),
          nn.BatchNorm2d(52)
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        #Depthwise Separable Convolution
        self.conv2 = nn.Sequential(
          nn.Conv2d(in_channels=52, out_channels=52, kernel_size=(3,3), padding=1, dilation=1, bias=False, groups=52),
          nn.Conv2d(in_channels=52, out_channels=104, kernel_size=(1,1)),
          nn.ReLU(),
          nn.BatchNorm2d(104),
          nn.Conv2d(in_channels=104, out_channels=104, kernel_size=(3,3), padding=1, dilation=1, bias=False, groups=104),
          nn.Conv2d(in_channels=104, out_channels=208, kernel_size=(1,1)),
          nn.ReLU(),
          nn.BatchNorm2d(208)
        )
        
        self.pool2 = nn.MaxPool2d(2, 2)

        #Dilated Convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=208, out_channels=416, kernel_size=(3,3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(416)
        ) 

        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=416, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),            
        ) 
        
        #GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return x