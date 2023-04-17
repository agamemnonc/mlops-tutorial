from pytorch_lightning import Callback
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
import pandas as pd
import wandb


class SamplesVisualisationLogger(Callback):
    """Callback to log wrong predictions from the first batch in the validation set to
    wandb."""

    def __init__(self, data_module: LightningDataModule) -> None:
        super().__init__()
        self.data_module = data_module

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Get first batch from validation set
        batch = next(iter(self.data_module.val_dataloader()))
        logits, preds = pl_module(batch["input_ids"], batch["attention_mask"])
        labels = batch["label"]
        sentences = batch["sentence"]
        df = pd.DataFrame(
            {
                "Sentence": sentences,
                "Label": labels.numpy(),
                "Prediction": preds.numpy(),
            }
        )
        # Keep only the portion of the dataframe where predictions are wrong
        wrong_preds_df = df[df["Label"] != df["Prediction"]]
        trainer.logger.experiment.log(
            {
                "Wrong predictions": wandb.Table(
                    dataframe=wrong_preds_df,
                    allow_mixed_types=True,
                    columns=["Sentence", "Label", "Prediction"],
                ),
                "step": trainer.global_step,
            }
        )
