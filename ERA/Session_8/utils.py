import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class_map = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1)
            correct_mask = pred.eq(target)
            incorrect_indices = ~correct_mask

            test_incorrect_pred['images'].extend(data[incorrect_indices])
            test_incorrect_pred['ground_truths'].extend(target[incorrect_indices])
            test_incorrect_pred['predicted_vals'].extend(pred[incorrect_indices])

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def plot_graph():
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

from PIL import Image

def show_incorrect_images():
    num_images = 10
    num_rows = 2
    num_cols = (num_images + 1) // 2  # Adjust the number of columns based on the number of images

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    for i in range(num_images):
        row_idx = i // num_cols
        col_idx = i % num_cols

        img = test_incorrect_pred['images'][i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize the image data
        label = test_incorrect_pred['ground_truths'][i].cpu().item()
        pred = test_incorrect_pred['predicted_vals'][i].cpu().item()

        axs[row_idx, col_idx].imshow(img)
        axs[row_idx, col_idx].set_title(f'GT: {class_map[label]}, Pred: {class_map[pred]}')
        axs[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.show()
