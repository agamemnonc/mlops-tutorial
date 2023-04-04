import torch
from transformers import AutoTokenizer, AutoModel
from data import ColaDataModule
from torch.optim import Adam
from model import ColaModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks


if __name__ == "__main__":
    # Config
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    batch_size = 32
    num_workers = 4
    max_length = 256
    lr = 3e-4
    lr_scale = 3  # Learning rate for the base model parameters is lr / lr_scale
    weight_decay = 1e-4
    max_epochs = 10
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
    logger = pl_loggers.TensorBoardLogger("logs", name="cola")
    ckpt_callback = pl_callbacks.ModelCheckpoint(
        dirpath="models", monitor="val/loss", save_top_k=1, mode="min"
    )
    trainer = Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=max_epochs,
        max_steps=max_steps,
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=[ckpt_callback],
    )
    trainer.fit(model=colamodule, datamodule=datamodule)
