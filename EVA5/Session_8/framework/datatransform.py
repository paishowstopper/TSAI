import torchvision.transforms as tf

class DataTransformation:
    def __init__(self):
        pass

    def CIFAR10Transform(self):
        return tf.Compose([tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def ResNet18TrainTransform(self):
        return tf.Compose([tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip(), tf.ToTensor(), tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def ResNet18TestTransform(self):
        return tf.Compose([tf.ToTensor(), tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])