import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MatthewsCorrCoef


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

        # Metrics
        self.train_accuracy = Accuracy(task="binary")
        self.valid_accuracy = Accuracy(task="binary")

        self.save_hyperparameters()

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
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        train_acc = self.train_accuracy(preds, batch["label"])
        self.log("train/acc", train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, preds = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("valid/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        valid_acc = self.valid_accuracy(preds, batch["label"])
        self.log("valid/acc", valid_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, *args, **kwargs):
        pass

    def predict_step(self):
        pass

    def configure_optimizers(self):
        return self.optimizer
