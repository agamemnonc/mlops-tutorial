import os

import torch
import wandb
from transformers import AutoTokenizer, AutoModel
from data import ColaDataModule
from torch.optim import Adam
from model import ColaModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

from callbacks import SamplesVisualisationLogger

if __name__ == "__main__":
    # Config
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    batch_size = 32
    num_workers = 4
    max_length = 512
    lr = 1e-2
    lr_scale = 1  # Learning rate for the base model parameters is lr / lr_scale
    weight_decay = 0
    max_epochs = 5
    max_steps = -1
    fast_dev_run = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datamodule = ColaDataModule(
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
    )

    # Training logic
    model = AutoModel.from_pretrained(model_name)
    head = torch.nn.Linear(model.config.hidden_size, 2)
    optimizer = Adam(
        [
            {
                "params": model.parameters(),
                "lr": lr / lr_scale,
                "weight_decay": weight_decay,
            },
            {"params": head.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]
    )
    colamodule = ColaModule(model=model, head=head, optimizer=optimizer)
    wandb_logger = pl_loggers.WandbLogger(project="mlops-tutorial", save_dir="logs")

    ckpt_callback = pl_callbacks.ModelCheckpoint(
        dirpath="models",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    samples_callback = SamplesVisualisationLogger(data_module=datamodule)
    logger = wandb_logger
    callbacks = [ckpt_callback, samples_callback]

    trainer = Trainer(
        accelerator="cpu" if torch.cuda.device_count() == 0 else "gpu",
        devices=1,
        max_epochs=max_epochs,
        max_steps=max_steps,
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=colamodule, datamodule=datamodule)
    wandb.finish()
