import torch
from transformers import AutoTokenizer, AutoModel
from data import ColaDataModule
from model import ColaModule
import datasets


if __name__ == "__main__":
    # Config
    test_sentence = "This boy is sitting on a bench"
    model_path = "models/epoch=2-step=804.ckpt"
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    batch_size = 32
    num_workers = 4
    max_length = 512

    # Preparation
    colamodule = ColaModule.load_from_checkpoint(model_path)
    colamodule.eval()
    colamodule.freeze()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datamodule = ColaDataModule(
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
    )
    datamodule.prepare_data()
    label_names = datamodule.dset["train"].features["label"].names

    # Inference logic
    test_sentence_enc = tokenizer(
        text=test_sentence,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits, preds = colamodule.forward(
            input_ids=test_sentence_enc["input_ids"],
            attention_mask=test_sentence_enc["attention_mask"],
        )
        preds_prob = torch.softmax(logits, dim=-1).max().item()
        print(
            f"Prediction for sentence '{test_sentence}': '{label_names[preds]}', with probability {preds_prob:.2f}."
        )
