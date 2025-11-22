import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class HateSpeechDataset(Dataset):
    def __init__(self, texts, contexts, labels, tokenizer, max_len):
        self.texts = texts
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        main_text = str(self.texts[index])
        context = str(self.contexts[index]) if self.contexts else ""
        # Combine context and main text with [SEP] only if context is not empty
        full_text = f"{context} [SEP] {main_text}".strip() if context else main_text

        inputs = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def load_data(
    data_path="data/sample_inputs.csv",
    tokenizer_name="xlm-roberta-base",
    max_len=128,
    test_size=0.1,
    label_encoder_path="label_encoder.pkl"
):
    # Read CSV data
    df = pd.read_csv(data_path)

    # Ensure 'context' column exists
    if "context" not in df.columns:
        df["context"] = ""

    # Encode labels with LabelEncoder
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    # Save label encoder for inference use
    with open(label_encoder_path, "wb") as f:
        pickle.dump(le, f)

    # Stratified train/validation split
    train_texts, val_texts, train_contexts, val_contexts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["context"].tolist(),
        df["label_id"].tolist(),
        test_size=test_size,
        random_state=42,
        stratify=df["label_id"]
    )

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create dataset objects
    train_dataset = HateSpeechDataset(train_texts, train_contexts, train_labels, tokenizer, max_len)
    val_dataset = HateSpeechDataset(val_texts, val_contexts, val_labels, tokenizer, max_len)

    return train_dataset, val_dataset, le
