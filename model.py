import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy


class ColaModule(LightningModule):
    def __init__(self, model: nn.Module, head: nn.Module, optimizer: Optimizer):
        """
        Args:
            model: PyTorch model for feature extraction
            head: PyTorch model for classification
            optimizer: PyTorch optimizer
        """
        super().__init__()
        self.model = model
        self.head = head
        self.optimizer = optimizer

        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # This accesses embeddieng of the CLS token output from the last layer
        h_cls = outputs.last_hidden_state[:, 0, :]
        # We pass it through the classification head to get the logits
        logits = self.head(h_cls)
        preds = logits.argmax(dim=-1)
        return logits, preds

    def training_step(self, batch, batch_idx):
        logits, preds = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        train_acc = accuracy(preds, batch["label"], task="binary")

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, mode="test")

    def _shared_eval_step(self, batch, mode):
        logits, preds = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        acc = accuracy(preds, batch["label"], task="binary")
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits, preds = self.forward(batch["input_ids"], batch["attention_mask"])
        probas = F.softmax(logits, dim=-1)
        return probas, preds

    def configure_optimizers(self):
        return self.optimizer
