import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.models.resnet import ResNet18 as Net
from torchsummary import summary

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@dataclass(frozen=False, init=True)
class GradCamWrapper:
    model: nn.Module
    target_layers: List[nn.Module]
    device: str
    targets: List[int]
    image_tensor: torch.Tensor
    image_numpy: np.ndarray
    reshape_transform: Optional[Callable] = None
    use_cuda: bool = field(init=False)
    target_categories: List[ClassifierOutputTarget] = field(init=False)

    def __post_init__(self) -> None:
        self.use_cuda = self.device == "cuda"
        self.target_categories = [
            ClassifierOutputTarget(target) for target in self.targets
        ]
        self.gradcam = self._init_gradcam_object()

    def _init_gradcam_object(self) -> GradCAM:
        return GradCAM(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=self.use_cuda,
            reshape_transform=self.reshape_transform,
        )

    def _generate_heatmap(self) -> np.ndarray:
        heatmap = self.gradcam(
            input_tensor=self.image_tensor,
            targets=self.target_categories,
        )
        return heatmap

    def display(self, save: bool = False) -> None:
        heatmap = self._generate_heatmap()
        heatmap = heatmap[0, :]
        visualization = show_cam_on_image(self.image_numpy, heatmap, use_rgb=True)
        fig, axes = plt.subplots(figsize=(20, 10), ncols=3)

        axes[0].imshow(self.image_numpy)
        axes[0].axis("off")

        axes[1].imshow(heatmap)
        axes[1].axis("off")

        axes[2].imshow(visualization)
        axes[2].axis("off")

        if save:
            plt.savefig("test.png", bbox_inches='tight')

        plt.show()
      
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

    plt.savefig("results/loss_accuracy_plot.png")

def show_incorrect_images(test_incorrect_pred, class_map, grad_cam=False, model=None):
    num_images = 20
    num_rows = 4
    num_cols = (num_images + 1) // num_rows  # Adjust the number of columns based on the number of images

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    for i in range(num_images):
        row_idx = i // num_cols
        col_idx = i % num_cols

        img = test_incorrect_pred['images'][i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize the image data
        label = test_incorrect_pred['ground_truths'][i].cpu().item()
        pred = test_incorrect_pred['predicted_vals'][i].cpu().item()

        if not grad_cam:
            axs[row_idx, col_idx].imshow(img)
            axs[row_idx, col_idx].set_title(f'GT: {class_map[label]}, Pred: {class_map[pred]}')
            axs[row_idx, col_idx].axis('off')

        if grad_cam:
            input_image = test_incorrect_pred['images'][i].unsqueeze(0)
            # Create a GradCamWrapper object
            cam_wrapper = GradCamWrapper(
                model=model,
                target_layers=[model.layer3[-1]],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                targets=[label],  # Assuming the ground truth is the target class
                image_tensor=input_image,
                image_numpy=img,
            )
  
            # Get the GradCAM heatmap
            heatmap = cam_wrapper._generate_heatmap()
            heatmap = heatmap[0, :]
  
            # Overlay the heatmap on the original image
            visualization = show_cam_on_image(img, heatmap, use_rgb=True)
  
            axs[row_idx, col_idx].imshow(heatmap, cmap='jet', alpha=0.7)
            axs[row_idx, col_idx].imshow(img, alpha=0.5)
            axs[row_idx, col_idx].set_title(f'GT: {class_map[label]}, Pred: {class_map[pred]}')
            axs[row_idx, col_idx].axis('off')

    if grad_cam:
        plt.savefig("results/incorrect_images_with_gradcam.png")
    else:
        plt.savefig("results/incorrect_images.png")


def model_summary():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    summary(model, input_size=(3, 32, 32))
