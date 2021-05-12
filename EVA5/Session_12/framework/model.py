import torch.nn as nn
import torch.nn.functional as F
import torch

#ResNet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#ResNet18 Modified
class ResNetModified(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetModified, self).__init__()

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.r1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Layer 2
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Layer 3
        # X2 = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.x2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.r2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.mp = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, 10, bias=False)

    def forward(self, x):

        #1
        x0 = self.prep(x)

        #2
        x1 = self.x1(x0)
        r1 = self.r1(x1)
        l1 = x1 + r1

        #3
        l2 = self.l2(l1)

        #4
        x2 = self.x2(l2)
        r2 = self.r2(x2)
        l3 = x2 + r2

        #5
        mp = self.mp(l3)
        flatten = mp.squeeze()

        #6
        fc = self.fc(flatten)

        #7 - Softmax
        y = F.softmax(fc, dim=1)
        return y

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