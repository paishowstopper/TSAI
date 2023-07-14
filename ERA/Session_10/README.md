# Residual CNN Model

This folder contains an implementation of a Custom ResNet model

## Folder Structure
```
└── README.md
└── src/
    └── data_setup.py
    └── utils.py
    └── engine.py
    └── custom_resnet.py
└── results/
    └── CustomResNet.pth
    └── incorrect_images.png
    └── loss_accuracy_plot.png
    └── lr_finder_plot.png
└── train.py
└── S9.ipynb
```

## How to Run the code
Clone the repo and run
Change your current directory to S9
```
python train.py
```

## OneCycle LR

```
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/40bbedb3-33ed-491b-a0d4-0015d115590f)

## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 256, 16, 16]         294,912
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-19            [-1, 512, 4, 4]               0
      BatchNorm2d-20            [-1, 512, 4, 4]           1,024
             ReLU-21            [-1, 512, 4, 4]               0
           Conv2d-22            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
             ReLU-27            [-1, 512, 4, 4]               0
        MaxPool2d-28            [-1, 512, 1, 1]               0
           Linear-29                   [-1, 10]           5,130
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.44
Params size (MB): 25.07
Estimated Total Size (MB): 31.53
----------------------------------------------------------------
```

## Incorrect Classified Images

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/519d7bbb-b96f-4a9d-a9c6-6d1e2103fc3c)


## Loss Accuracy Plot

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/ebdff881-ebaf-40bd-bb1c-edcaa22c2431)



## Key Achievements
* 92.53% Training Accuracy within 24 epochs using OneCycle LR
* Implemented Model with Residual Connections
* Used Image Augmentations to remove overfitting

