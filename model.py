from typing import List, Tuple, Dict, Optional

import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from pytorch_lightning import LightningModule
import torch
import torchmetrics
import wandb

from plots import plot_confusion_matrix


class ColaModule(LightningModule):
    def __init__(self, model: nn.Module, head: nn.Module, optimizer: Optimizer) -> None:
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
        self.val_f1 = torchmetrics.F1Score(num_classes=ncl, task=tsk)
        self.val_recall_macro = torchmetrics.Recall(num_classes=ncl, task=tsk)
        self.val_recall_micro = torchmetrics.Recall(
            num_classes=ncl, task=tsk, average="micro"
        )
        self.val_precision_macro = torchmetrics.Precision(num_classes=ncl, task=tsk)
        self.val_precision_micro = torchmetrics.Precision(
            num_classes=ncl, task=tsk, average="micro"
        )

        self.save_hyperparameters(ignore=["model", "head"])

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # Forward pass
        loss, logits, preds = self._shared_eval_step(batch)

        # Update metrics
        self.train_accuracy(preds, batch["label"])

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/acc",
            self.train_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        # Forward pass
        loss, logits, preds = self._shared_eval_step(batch)

        # Update metrics
        self.val_accuracy(preds, batch["label"])
        self.val_f1(preds, batch["label"])
        self.val_recall_macro(preds, batch["label"])
        self.val_recall_micro(preds, batch["label"])
        self.val_precision_macro(preds, batch["label"])
        self.val_precision_micro(preds, batch["label"])

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
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
        return {"logits": logits, "preds": preds, "labels": batch["label"]}

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        labels = torch.cat([batch["labels"] for batch in outputs])
        preds = torch.cat([batch["preds"] for batch in outputs])
        # Log confusion matrix
        plot = plot_confusion_matrix(
            actual=labels.cpu().numpy(),
            predictions=preds.cpu().numpy(),
            figsize=(5, 5),
        )
        self.logger.experiment.log({"Confusion Matrix": wandb.Image(plot)})

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        self._shared_eval_step(batch)
        loss, logits, preds = self._shared_eval_step(batch)
        self.test_accuracy(preds, batch["label"])
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test/acc", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True
        )

    def predict_step(
        self, batch: Dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, preds = self.forward(batch["input_ids"], batch["attention_mask"])
        probas = F.softmax(logits, dim=-1)
        return probas, preds

    def _shared_eval_step(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs forward pass and returns loss, logits and predictions."""
        logits, preds = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        return loss, logits, preds

    def forward(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs forward pass and returns logits and predictions."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # This accesses embedding of the CLS token (index 0) output from the last layer
        h_cls = outputs.last_hidden_state[:, 0, :]
        # We pass it through the classification head to get the logits
        logits = self.head(h_cls)
        preds = logits.argmax(dim=-1)
        return logits, preds

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer
