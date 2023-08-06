# End-to-End Model Training with PyTorch Lightning

This repository provides a complete guide and code for training a custom ResNet model using PyTorch Lightning on the CIFAR-10 dataset. PyTorch Lightning simplifies the training process by providing a high-level interface and best practices for organizing your code.

## Folder Structure
```
└── results/
    └── CustomResNet.pth.zip
└── src/
    └── datamodule.py
    └── model.py
    └── utils.py
└── README.md
└── S12.ipynb
└── train.py
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

![adam](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/416c8983-189f-410b-a85c-c50b073580d4)


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

## Misclassified Images

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/b378b09c-97c0-48a6-8d7a-3e0c0d7b3b13)


## Grad CAM output for Misclassified Images

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/441d9972-29e1-43c3-a790-0234517e48bb)

## Loss/Accuracy Plot

![loss](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/d8742e57-4f8f-4166-a05f-4a2e8111bc43)

![acc](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/6e9a0fb6-6af0-4914-bf34-59a029de4385)

## Epoch

![epoch](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/de52461c-0e0c-48de-a575-20b3c78a2896)


## Key Achievements
* Created the end-to-end training pipeline using Pytorch Lightning
* Created a interactive Gradio interface for Model analysis and demo
* Link to Gradio app: https://huggingface.co/spaces/selvaraj-sembulingam/era-pl-resnet-demo



