# Grad-CAM

Class Activation Maps (CAMs) are visualization methods used for explaining deep learning models. In this method, the model predicted class scores are traced back to the last convolution layer to highlight discriminative regions of interest in the image that are class-specific and not even generic to other computer vision or image processing algorithms. Gradient CAM or popularly called as Grad-CAMs combines the effect of guided backpropagation and CAM to highlight class discriminative regions of interest without highlighting the granular pixel importance. But Grad-CAM can be applied to any CNN architectures, unlike CAM, which can be applied to architectures that perform global average pooling over output feature maps coming from the convolution layer, just prior to the prediction layer. To get a more detailed understanding on the Grad-CAM process, you can have a look at this research paper Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Ramprasaath et. al —

## Grad-CAM Architecture:

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/d56a8ae5-14c7-4961-bc0c-79ee321638ad)


## Folder Structure
```
└── README.md
└── src/
    └── models
        └── resnet.py
    └── data_setup.py
    └── utils.py
    └── engine.py
└── results/
    └── ResNet18.pth
    └── incorrect_images.png
    └── incorrect_images_with_gradcam.png
    └── loss_accuracy_plot.png
    └── lr_finder_plot.png
└── train.py
└── S11.ipynb
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

![download](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/ba198128-9c3e-4a6a-aef3-ec01b340e32c)


## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```

## Incorrect Classified Images

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/49243d79-5f03-4780-93a1-0e10bd811d31)

## Incorrect Classified Images With GradCAM

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/2f900377-26d5-4515-a456-9b91203390f1)


## Loss Accuracy Plot

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/b1d2edb5-a0f8-4111-99c6-31845ef10453)


## Key Achievements
Implemented GradCAM to understand what the model looks at in predicting misclassified images
