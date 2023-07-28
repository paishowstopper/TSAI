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
* Best Train Accuracy: 74.2
* Best Test Accuracy: 70.51

![image](https://github.com/paishowstopper/TSAI/assets/26896746/4dd8c70a-4fcd-4ed5-968d-f95ce604291d)

![image](https://github.com/paishowstopper/TSAI/assets/26896746/36eb6125-4d02-45d7-9ddc-a32ab69d6176)

# Layer Normalization
### Results:
* Best Train Accuracy: 74.87
* Best Test Accuracy: 72.83

![image](https://github.com/paishowstopper/TSAI/assets/26896746/c8095afb-99ea-475e-b31b-4c47b11032b6)

![image](https://github.com/paishowstopper/TSAI/assets/26896746/8333ccc7-2810-434b-8721-8530ef65d687)

# Group Normalization
### Results:
* Best Train Accuracy: 74.5
* Best Test Accuracy: 71.03

![image](https://github.com/paishowstopper/TSAI/assets/26896746/52134a8b-d035-49ce-8f8b-1013647f8c99)

![image](https://github.com/paishowstopper/TSAI/assets/26896746/ceaaf975-6fa8-454a-a88d-8bc42a2149db)
