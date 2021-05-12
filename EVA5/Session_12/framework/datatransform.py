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

    def AlbumentationTrainTransform(self):
        tf = tc.Compose([ta.PadIfNeeded(8, 8, always_apply=True),
                        ta.RandomCrop(height=64, width=64, always_apply=True),
                        ta.Cutout(num_holes = 1, max_h_size=32, max_w_size=32, always_apply=True),
                        ta.HorizontalFlip(),
                        tp.ToTensor(dict (mean=(0.4802, 0.4481, 0.3975), std=(0.2302, 0.2265, 0.2262)))
                        ])
        return lambda img: tf(image = np.array(img))["image"]

    def AlbumentationTestTransform(self):
        tf = tc.Compose([tp.ToTensor(dict (mean=(0.4802, 0.4481, 0.3975), std=(0.2302, 0.2265, 0.2262)))
                        ])
        return lambda img: tf(image = np.array(img))["image"]