import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from custom_resnet import CustomResNet as Net
from torchsummary import summary

def save_model(model, target_dir, model_name):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def plot_graph(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # Plot Train and Test Loss
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(test_losses, label='Test Loss')
    axs[0].set_title("Train and Test Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot Train and Test Accuracy
    axs[1].plot(train_acc, label='Train Accuracy')
    axs[1].plot(test_acc, label='Test Accuracy')
    axs[1].set_title("Train and Test Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.savefig("loss_accuracy_plot.png")

def show_incorrect_images(test_incorrect_pred, class_map):
    num_images = 10
    num_rows = 2
    num_cols = (num_images + 1) // 2  # Adjust the number of columns based on the number of images

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

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

    plt.savefig("incorrect_images.png")


def model_summary():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))
