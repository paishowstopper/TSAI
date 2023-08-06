import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from src.datamodule import CIFARDataModule
from src.model import CustomResNet


if __name__ == '__main__':

  pl.seed_everything(1, workers=True)

  data_module = CIFARDataModule()
  model = CustomResNet()

  checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", filename='best_model', save_last=True)
  lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")

  trainer = pl.Trainer(
      max_epochs=24,
      deterministic=True,
      logger=True,
      callbacks=[checkpoint, lr_rate_monitor],
      enable_model_summary=False,
      log_every_n_steps=1,
      precision=16
  )
  trainer.fit(model, data_module)
  trainer.test(model, data_module.test_dataloader())
  torch.save(model.state_dict(), 'CustomResNet.pth')
