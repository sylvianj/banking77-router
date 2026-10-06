import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model hosted on Hugging Face Hub — loads from the internet, not local files
MODEL_ID = "Sylvianjoki/banking77-distilbert"

# Robust path to label_names.json — works from any directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_NAMES_PATH = os.path.join(_THIS_DIR, "..", "models", "label_names.json")

MAX_LENGTH = 64


def load_model():
    """Load tokenizer, model, and label names. Downloads from HF Hub on first run."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    with open(LABEL_NAMES_PATH) as f:
        label_names = json.load(f)

    return tokenizer, model, label_names


def predict(text, tokenizer, model, label_names, top_k=3):
    """Return top-k (intent, probability) predictions for a message."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, k=top_k)

    return [
        {"intent": label_names[i], "probability": p}
        for p, i in zip(top_probs.tolist(), top_indices.tolist())
    ]