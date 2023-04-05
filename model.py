import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from pytorch_lightning import LightningModule
import torchmetrics


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

        self.n_classes = 2
        self.task = "binary"

        ncl, tsk = self.n_classes, self.task
        self.train_accuracy = torchmetrics.Accuracy(num_classes=ncl, task=tsk)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=ncl, task=tsk)
        self.test_accuracy = torchmetrics.Accuracy(num_classes=ncl, task=tsk)
        self.val_f1 = torchmetrics.F1(num_classes=ncl, task=tsk)
        self.val_recall_macro = torchmetrics.Recall(num_classes=ncl, task=tsk)
        self.val_recall_micro = torchmetrics.Recall(
            num_classes=ncl, task=tsk, average="micro"
        )
        self.val_precision_macro = torchmetrics.Precision(num_classes=ncl, task=tsk)
        self.val_precision_micro = torchmetrics.Precision(
            num_classes=ncl, task=tsk, average="micro"
        )

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
        self.train_accuracy(preds, batch["label"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/acc",
            self.train_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, mode="val")
        loss, logits, preds = self._shared_eval_step(batch, mode="val")

        self.val_accuracy(preds, batch["label"])
        self.val_f1(preds, batch["label"])
        self.val_recall_macro(preds, batch["label"])
        self.val_recall_micro(preds, batch["label"])
        self.val_precision_macro(preds, batch["label"])
        self.val_precision_micro(preds, batch["label"])

        self.log(
            "val/acc", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("val/f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val/recall_macro",
            self.val_recall_micro,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/recall_micro",
            self.val_recall_micro,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/precision_macro",
            self.val_precision_macro,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/precision_micro",
            self.val_precision_micro,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, mode="val")
        loss, logits, preds = self._shared_eval_step(batch, mode="val")
        self.test_accuracy(preds, batch["label"])
        self.log(
            "test/acc", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True
        )

    def _shared_eval_step(self, batch, mode):
        logits, preds = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log(f"{mode}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, logits, preds

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits, preds = self.forward(batch["input_ids"], batch["attention_mask"])
        probas = F.softmax(logits, dim=-1)
        return probas, preds

    def configure_optimizers(self):
        return self.optimizer
