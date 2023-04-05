from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datasets
from transformers import PreTrainedTokenizer


class ColaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int,
        max_length: int,
    ):
        """
        Args:
            tokenizer: Pre-trained tokenizer to use to process input text.
            batch_size: Batch size for the dataloaders.
            max_length: Max length for tokenization.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        self.dset = None
        self.data_train, self.data_val, self.data_test, self.data_predict = [None] * 4

    def prepare_data(self):
        """Download and caches the dataset."""
        self.dset = datasets.load_dataset(path="glue", name="cola")

    def setup(self, stage: str = None):
        """
        Tokenizes and formats various data splits.

        Args:
            stage: This argument will be used by the trainer to work out the
            stage we are at: {fit, validate, test, predict}.
        """
        if stage == "fit" or stage is None:
            self.data_train = self.dset["train"].map(self.tokenize_data, batched=True)
            self.data_train.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )
            self.data_val = self.dset["validation"].map(
                self.tokenize_data, batched=True
            )
            self.data_val.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

        if stage == "test":
            self.data_test = self.dset["test"].map(self.tokenize_data, batched=True)
            self.data_test.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

        if stage == "predict":
            self.data_predict = self.dset["predict"].map(
                self.tokenize_data, batched=True
            )
            self.data_predict.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def tokenize_data(self, example):
        """Tokenizes a single example."""
        return self.tokenizer(
            text=example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
