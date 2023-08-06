# Grad-CAM

Class Activation Maps (CAMs) are visualization methods used for explaining deep learning models. In this method, the model predicted class scores are traced back to the last convolution layer to highlight discriminative regions of interest in the image that are class-specific and not even generic to other computer vision or image processing algorithms. Gradient CAM or popularly called as Grad-CAMs combines the effect of guided backpropagation and CAM to highlight class discriminative regions of interest without highlighting the granular pixel importance. But Grad-CAM can be applied to any CNN architectures, unlike CAM, which can be applied to architectures that perform global average pooling over output feature maps coming from the convolution layer, just prior to the prediction layer.

## Training and Testing Logs:

Epoch 0, Learning Rate: 0.000479
Train: Loss=1.5773 Batch_id=97 Accuracy=37.31: 100% 98/98 [00:41<00:00,  2.37it/s]
Test set: Loss: 0.0040, Accuracy: 3710/10000 (37.10%)

Epoch 1, Learning Rate: 0.00998259509202454
Train: Loss=1.2418 Batch_id=97 Accuracy=48.47: 100% 98/98 [00:40<00:00,  2.43it/s]
Test set: Loss: 0.0029, Accuracy: 5025/10000 (50.25%)

Epoch 2, Learning Rate: 0.01948619018404908
Train: Loss=0.9970 Batch_id=97 Accuracy=57.90: 100% 98/98 [00:41<00:00,  2.37it/s]
Test set: Loss: 0.0022, Accuracy: 6133/10000 (61.33%)

Epoch 3, Learning Rate: 0.028989785276073616
Train: Loss=0.8569 Batch_id=97 Accuracy=64.12: 100% 98/98 [00:41<00:00,  2.34it/s]
Test set: Loss: 0.0020, Accuracy: 6763/10000 (67.63%)

Epoch 4, Learning Rate: 0.03849338036809816
Train: Loss=0.9033 Batch_id=97 Accuracy=67.73: 100% 98/98 [00:42<00:00,  2.33it/s]
Test set: Loss: 0.0027, Accuracy: 6119/10000 (61.19%)

Epoch 5, Learning Rate: 0.0478674182244898
Train: Loss=0.7751 Batch_id=97 Accuracy=71.63: 100% 98/98 [00:41<00:00,  2.35it/s]
Test set: Loss: 0.0039, Accuracy: 5240/10000 (52.40%)

Epoch 6, Learning Rate: 0.044674404224489796
Train: Loss=0.5745 Batch_id=97 Accuracy=74.76: 100% 98/98 [00:41<00:00,  2.34it/s]
Test set: Loss: 0.0016, Accuracy: 7312/10000 (73.12%)

Epoch 7, Learning Rate: 0.041481390224489795
Train: Loss=0.5533 Batch_id=97 Accuracy=77.28: 100% 98/98 [00:41<00:00,  2.34it/s]
Test set: Loss: 0.0013, Accuracy: 7727/10000 (77.27%)

Epoch 8, Learning Rate: 0.038288376224489794
Train: Loss=0.6315 Batch_id=97 Accuracy=79.05: 100% 98/98 [00:41<00:00,  2.34it/s]
Test set: Loss: 0.0012, Accuracy: 8025/10000 (80.25%)

Epoch 9, Learning Rate: 0.03509536222448979
Train: Loss=0.5282 Batch_id=97 Accuracy=80.61: 100% 98/98 [00:41<00:00,  2.35it/s]
Test set: Loss: 0.0010, Accuracy: 8281/10000 (82.81%)

Epoch 10, Learning Rate: 0.03190234822448979
Train: Loss=0.4483 Batch_id=97 Accuracy=82.33: 100% 98/98 [00:42<00:00,  2.32it/s]
Test set: Loss: 0.0009, Accuracy: 8453/10000 (84.53%)

Epoch 11, Learning Rate: 0.028709334224489794
Train: Loss=0.5446 Batch_id=97 Accuracy=83.77: 100% 98/98 [00:41<00:00,  2.35it/s]
Test set: Loss: 0.0010, Accuracy: 8388/10000 (83.88%)

Epoch 12, Learning Rate: 0.025516320224489793
Train: Loss=0.4750 Batch_id=97 Accuracy=84.84: 100% 98/98 [00:41<00:00,  2.35it/s]
Test set: Loss: 0.0011, Accuracy: 8327/10000 (83.27%)

Epoch 13, Learning Rate: 0.022323306224489792
Train: Loss=0.3535 Batch_id=97 Accuracy=86.49: 100% 98/98 [00:42<00:00,  2.32it/s]
Test set: Loss: 0.0008, Accuracy: 8732/10000 (87.32%)

Epoch 14, Learning Rate: 0.01913029222448979
Train: Loss=0.2907 Batch_id=97 Accuracy=87.46: 100% 98/98 [00:41<00:00,  2.35it/s]
Test set: Loss: 0.0008, Accuracy: 8740/10000 (87.40%)

Epoch 15, Learning Rate: 0.015937278224489794
Train: Loss=0.3166 Batch_id=97 Accuracy=88.76: 100% 98/98 [00:41<00:00,  2.35it/s]
Test set: Loss: 0.0007, Accuracy: 8905/10000 (89.05%)

Epoch 16, Learning Rate: 0.012744264224489793
Train: Loss=0.2650 Batch_id=97 Accuracy=89.81: 100% 98/98 [00:41<00:00,  2.35it/s]
Test set: Loss: 0.0006, Accuracy: 8981/10000 (89.81%)

Epoch 17, Learning Rate: 0.009551250224489792
Train: Loss=0.2253 Batch_id=97 Accuracy=91.32: 100% 98/98 [00:42<00:00,  2.32it/s]
Test set: Loss: 0.0006, Accuracy: 8974/10000 (89.74%)

Epoch 18, Learning Rate: 0.006358236224489798
Train: Loss=0.2441 Batch_id=97 Accuracy=92.69: 100% 98/98 [00:41<00:00,  2.35it/s]
Test set: Loss: 0.0006, Accuracy: 9017/10000 (90.17%)

Epoch 19, Learning Rate: 0.003165222224489797
Train: Loss=0.1654 Batch_id=97 Accuracy=93.74: 100% 98/98 [00:42<00:00,  2.33it/s]
Test set: Loss: 0.0006, Accuracy: 9089/10000 (90.89%)

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

![image](https://github.com/paishowstopper/TSAI/assets/26896746/21748270-7d84-4fac-9817-403e352107b0)


## Incorrect Classified Images With GradCAM

![image](https://github.com/paishowstopper/TSAI/assets/26896746/1edd6541-06a0-414b-b100-f10a5efa6e06)
