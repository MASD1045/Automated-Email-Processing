import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.cuda.amp import autocast

MODEL_NAME = "philschmid/bart-large-cnn-samsum"

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def get_summary_lengths(input_len):
    max_len = min(120, max(40, int(input_len * 0.3)))
    min_len = max(20, int(max_len * 0.5))
    return min_len, max_len


def summarize_text(text, max_input_tokens=1024):
    if not text.strip():
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    ).to(device)

    input_len = inputs.input_ids.shape[1]
    min_len, max_len = get_summary_lengths(input_len)

    with torch.no_grad():
        with autocast():
            summary_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                min_length=min_len,
                max_length=max_len,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
