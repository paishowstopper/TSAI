import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import pytorch_lightning as pl


class CustomResNet(pl.LightningModule):
    def __init__(self, dropout_value=0.01):
        super(CustomResNet, self).__init__()

        self.test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': [], 'grad_cam': []}

        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.resblock1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.maxpoollayer = nn.Sequential(
            nn.MaxPool2d(kernel_size = 4, stride = 4)
            )
        self.fclayer = nn.Sequential(
            nn.Linear(512,10)
            )
        
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        
    def loss_function(self, pred, target):
        criterion = torch.nn.CrossEntropyLoss()
        
        return criterion(pred, target)
        
    def forward(self, x):
        x = self.preplayer(x)
        x = self.layer1(x)
        r1 = self.resblock1(x)
        x = x + r1
        x = self.layer2(x)
        x = self.layer3(x)
        r2 = self.resblock2(x)
        x = x + r2
        x = self.maxpoollayer(x)
        x = x.view((x.shape[0],-1))
        x = self.fclayer(x)
        
        return x

    def get_loss_accuracy(self, batch):
        images, labels = batch
        predictions = self(images)
        loss = self.loss_function(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        
        return loss, accuracy * 100

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.get_loss_accuracy(batch)
        self.log("loss/train", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("acc/train", accuracy, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.get_loss_accuracy(batch)
        self.log("loss/val", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("acc/val", accuracy, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.validation_step(batch, batch_idx)

        return loss
        
    def configure_optimizers(self):
        LEARNING_RATE=0.03
        WEIGHT_DECAY=0
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        self.trainer.fit_loop.setup_data()
        dataloader = self.trainer.train_dataloader

        lr_scheduler = OneCycleLR(
          optimizer,
          max_lr=4.79E-02,
          steps_per_epoch=len(dataloader),
          epochs=24,
          pct_start=5/24,
          div_factor=100,
          three_phase=False,
          final_div_factor=100,
          anneal_strategy='linear'
        )

        scheduler = {"scheduler": lr_scheduler, "interval" : "step"}

        return [optimizer], [scheduler]
        

