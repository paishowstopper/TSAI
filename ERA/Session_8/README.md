# Image Classification with CNN: Normalization Techniques Comparison

This repository contains an implementation of a Convolutional Neural Network (CNN) for image classification, showcasing the comparison of three normalization techniques: `Batch Normalization`, `Layer Normalization`, and `Group Normalization`. The goal is to analyze their effects on the performance of the CNN model when applied to image datasets.

## Dataset
CIFAR10

## Receptive Field Calculation of the Model
|             | r_in | n_in | j_in | s | r_out | n_out | j_out |  | kernal_size | padding |
|-------------|------|------|------|---|-------|-------|-------|--|-------------|---------|
| Conv        | 1    | 32   | 1    | 1 | 3     | 32    | 1     |  | 3           | 1       |
| Conv        | 3    | 32   | 1    | 1 | 5     | 32    | 1     |  | 3           | 1       |
| Conv        | 5    | 32   | 1    | 1 | 5     | 32    | 1     |  | 1           | 0       |
| Max Pooling | 5    | 32   | 1    | 2 | 6     | 16    | 2     |  | 2           | 0       |
| Conv        | 6    | 16   | 2    | 1 | 10    | 16    | 2     |  | 3           | 1       |
| Conv        | 10   | 16   | 2    | 1 | 14    | 16    | 2     |  | 3           | 1       |
| Conv        | 14   | 16   | 2    | 1 | 18    | 16    | 2     |  | 3           | 1       |
| Conv        | 18   | 16   | 2    | 1 | 18    | 16    | 2     |  | 1           | 0       |
| Max Pooling | 18   | 16   | 2    | 2 | 20    | 8     | 4     |  | 2           | 0       |
| Conv        | 20   | 8    | 4    | 1 | 28    | 8     | 4     |  | 3           | 1       |
| Conv        | 28   | 8    | 4    | 1 | 36    | 8     | 4     |  | 3           | 1       |
| Conv        | 36   | 8    | 4    | 1 | 36    | 8     | 4     |  | 1           | 0       |
| GAP         | 36   | 8    | 4    | 1 | 64    | 1     | 4     |  | 8           | 0       |
| Conv        | 64   | 1    | 4    | 1 | 64    | 1     | 4     |  | 1           | 0       |

# Batch Normalization
### Results:
* Best Train Accuracy: 74.04
* Best Test Accuracy: 71.58

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/b44ecd71-ed23-46b3-b52c-8771e0e736c7)


# Layer Normalization
### Results:
* Best Train Accuracy: 72.67
* Best Test Accuracy: 70.68

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/ab13bedc-9fc9-4de4-b68e-2f6bbad012f6)


# Group Normalization
### Results:
* Best Train Accuracy: 73.57
* Best Test Accuracy: 70.56

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/95a9b5ab-2f71-45f2-a071-00d05c312561)



## Misclassified Images

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/ad9591d4-a86d-4ce0-8231-0d482dd33415)

## Findings

* Batch normalization seems to perform better than layer and group normalization.
* This is because batch normalization addresses the Internal Covariate Shift.
* Batch normalization introduces slight regularization effects.
* While the layer normalization does not consider the batch dimension and normalizes across the entire layer.
* For CNNs, where spatial and channel information is important, layer normalization may not capture the statistical variations effectively, which can lead to suboptimal performance when dealing with larger batch sizes.
* Also, Group normalization normalizes within groups of channels but does not consider the spatial dimensions.
* Hence Batch Normalization outperforms Layer and Group Normalization in CNNs.

* Note: A slight overfitting is still there in the model. This needs to be addressed with techniques like Image Augmentation etc
