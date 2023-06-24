import torchvision.transforms as tf

class DataTransformation:
    def __init__(self):
        pass

    def CIFAR10Transform(self):
        return tf.Compose([tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])