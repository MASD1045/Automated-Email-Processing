import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

LABELS = ["negative", "neutral", "positive"]

@torch.no_grad()
def analyze_sentiment(text, max_length=512):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    outputs = model(**enc)
    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    label_id = int(np.argmax(probs))

    return {
        "label": LABELS[label_id],
        "confidence": float(probs[label_id]),
        "p_negative": float(probs[0]),
        "p_neutral": float(probs[1]),
        "p_positive": float(probs[2])
    }
