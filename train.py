import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import os
import pickle

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):  # Reduced max_len for speed
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def main():
    # 🔁 CHANGED: Use XLM-RoBERTa
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv("data/sample_inputs.csv")

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    os.makedirs("models/model", exist_ok=True)
    with open("models/model/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    train_dataset = HateSpeechDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = HateSpeechDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Simulates batch size of 32
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=42,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
        fp16=True  # Mixed-precision training for speed (works fine with XLM-R if your GPU supports it)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    model.save_pretrained("models/model")
    tokenizer.save_pretrained("models/model")

if __name__ == "__main__":
    main()
