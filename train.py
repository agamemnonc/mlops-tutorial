import torch
from transformers import AutoTokenizer, AutoModel
from data import ColaDataModule
from torch.optim import Adam
from model import ColaModule
from pytorch_lightning import Trainer


if __name__ == "__main__":
    # Config
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    batch_size = 32
    max_length = 256
    lr = 3e-4
    lr_scale = 10  # Learning rate for the base model parameters is lr / lr_scale
    weight_decay = 1e-4
    max_epochs = 10
    max_steps = -1
    fast_dev_run = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datamodule = ColaDataModule(
        tokenizer=tokenizer, batch_size=batch_size, max_length=max_length
    )

    # Main logic
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
    trainer = Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=max_epochs,
        max_steps=max_steps,
        fast_dev_run=fast_dev_run,
        logger=False,
        enable_checkpointing=False,
        callbacks=None,
    )
    trainer.fit(model=colamodule, datamodule=datamodule)
