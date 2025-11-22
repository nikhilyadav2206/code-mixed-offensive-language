import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from config import MODEL_DIR

# Define label mapping manually (model is binary)
label_map = {0: "non-offensive", 1: "offensive"}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_input(text: str, context: str = None, max_len: int = 256):
    """
    Tokenize input for model. If context is provided, it is concatenated properly.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text input must be a non-empty string.")
    
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"
    combined = f"{context.strip()} {sep_token} {text.strip()}" if context else text.strip()
    
    encoding = tokenizer(
        combined,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    )
    return encoding

def smart_predict(text: str, context: str = None, threshold: float = 0.55):
    """
    Predicts the class label for the given text with optional context.
    Applies fallback if context prediction is weak.
    """
    try:
        inputs = preprocess_input(text, context)
    except ValueError:
        return "Invalid Input", [0.0, 0.0]

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    label_idx = int(np.argmax(probs))
    confidence = float(probs[label_idx])

    # Fallback if context hurts confidence
    if context and confidence < threshold:
        try:
            fallback_inputs = preprocess_input(text)
            fallback_inputs = {k: v.to(device) for k, v in fallback_inputs.items()}
            with torch.no_grad():
                fallback_outputs = model(**fallback_inputs)
                fallback_probs = torch.softmax(fallback_outputs.logits, dim=1).cpu().numpy()[0]

            fallback_idx = int(np.argmax(fallback_probs))
            fallback_conf = float(fallback_probs[fallback_idx])

            if fallback_conf > confidence:
                label_idx = fallback_idx
                probs = fallback_probs
        except Exception:
            pass

    label = label_map[label_idx]
    return label, probs.tolist()
