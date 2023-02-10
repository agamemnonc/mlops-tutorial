import pytorch_lightning as pl
import datasets
from transformers import AutoTokenizer


class ColaDataModule(pl.LightningDataModule):
    def __init__(self, model_name: str, batch_size: int):
        """
        Args:
            model_name: Model name. It will be used to access the corresponding
            tokenizer.

            batch_size: Batch size for the dataloaders.
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        """Download the data and split into training and validation splits. """
        pass

    def setup(self, stage: str = None):
        """

        Args:
            stage: This argument will be used by the trainer to work out the
            stage we are at: {fit, validate, test, predict}.
        """
        if stage == "fit" or stage is None:
            pass

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None
