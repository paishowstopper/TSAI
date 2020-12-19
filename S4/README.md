This is a model to train MNIST dataset and achieved an accuracy of 99.46% on the 18th epoch with 11608 parameters.

1. Convolution -> Input - 1, Output, 16, Kernel - 3
2. ReLU
3. Batch Normalization
4. Convolution -> Input - 16, Output, 32, Kernel - 3
5. ReLU
6. Batch Normalization
7. Convolution -> Input - 32, Output, 10, Kernel - 1
8. ReLU
9. Batch Normalization
10. Max Pooling -> RF - 10x10
11. Convolution -> Input - 10, Output, 16, Kernel - 3
12. ReLU
13. Batch Normalization
14. Convolution -> Input - 16, Output, 16, Kernel - 3
15. ReLU
16. Batch Normalization
17. Convolution -> Input - 16, Output, 16, Kernel - 3
18. ReLU
19. Batch Normalization
20. Average Pooling
21. Convolution -> Input - 16, Output, 10, Kernel - 1
22. log_softmax output


Stride = 1 (Default)
Padding = 0 (Default)
Dropout - 0
LR = 0.01
Momentum = 0.9
Batch Size = 32
Epochs = 19
