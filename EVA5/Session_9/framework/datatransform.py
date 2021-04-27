import torchvision.transforms as tf
import numpy as np
from albumentations.pytorch import transforms as tp
from albumentations.augmentations import transforms as ta
from albumentations.core import composition as tc

class DataTransformation:
    def __init__(self):
        pass

    def CIFAR10Transform(self):
        return tf.Compose([tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def ResNet18TrainTransform(self):
        return tf.Compose([tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip(), tf.ToTensor(), tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def ResNet18TestTransform(self):
        return tf.Compose([tf.ToTensor(), tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
    # https://albumentations.ai/docs/api_reference/augmentations/transforms/
    # https://albumentations.ai/docs/api_reference/pytorch/transforms/

    #Normalize is set as ToTensor parameter - dict(mean, std)
    def AlbumentationTrainTransform(self):
        atf = tc.Compose([ta.HorizontalFlip(),
                            # ta.Blur(),
                            # ta.ChannelShuffle(),
                            # ta.InvertImg(),
                            # ta.Rotate(),
                            ta.RandomCrop(height=30, width=30, p=5.0),
                            # ta.Cutout(1, 8, 8, [0.4914, 0.4822, 0.4465]),
                            tp.ToTensor(dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
                            ])
        return lambda img: atf(image = np.array(img))["image"]

    def AlbumentationTestTransform(self):
        atf = tc.Compose([tp.ToTensor(dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))])
        return lambda img: atf(image = np.array(img))["image"]
        
        
