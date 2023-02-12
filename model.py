import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MatthewsCorrCoef


class ColaModule(LightningModule):
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        """
        Args:
            model: PyTorch model
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer

        # Metrics
        self.train_accuracy = Accuracy(task='binary')
        self.valid_accuracy = Accuracy(task='binary')

        self.save_hyperparameters()

    def forward(self, batch, batch_idx):
        return self.model(batch['x'])

    def training_step(self, batch, batch_idx):
        logits  = self.forward(batch, batch_idx)
        loss = F.cross_entropy(logits, batch['label'])
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch, batch_idx)
        loss = F.cross_entropy(logits, batch['label'])
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True)
        accuracy = 2.0
        self.log(
            "val_acc",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

    def test_step(self, *args, **kwargs):
        pass

    def predict_step(self):
        pass

    def configure_optimizers(self):
        return self.optimizer

