model.py
========

This file has code block 7 from the original file which is the Neural network model class.

utils.py
========

This file has the code blocks 8, 9 and 11. Code block 8 has the variables to capture the training and testing losses and accuracies. Code block 9 has the training and testing functions. Code block 11 is used to plot the training and test accuracies and losses graphically. 

S5.ipynb
========

This file has the code to run the netwrok. Block 1 has all the necessary imports. Block 2 sets the device to GPU (CUDA) if available, else sets it to CPU. Code block 3 declares and initializes the train and test transforms. Train transforms has a few augmentation techniques applied to it (for improved training) apart from setting mean and standard values and transforming to tensor (common to test transform also). MNIST train and test data is loaded to train and test loaders respectively with respective transforms applied. Shuffle is set to true for ensuring that the network does not learn to by heart. Batch size is set to 512. Learning rate is set to 0.01. Model is trained for 20 epochs (result is displayed graphically). Model summary is displayed at the end.