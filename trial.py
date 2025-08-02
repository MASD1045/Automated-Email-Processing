from flask import Flask, render_template, redirect, url_for
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer, util
import os

app = Flask(__name__)

# Load BART model and tokenizer for summarization
summarizer_model_name = "philschmid/bart-large-cnn-samsum"
tokenizer = BartTokenizer.from_pretrained(summarizer_model_name)
model = BartForConditionalGeneration.from_pretrained(summarizer_model_name)

# Load sentence transformer model for categorization
embedder = SentenceTransformer('all-MiniLM-L6-v2')
CATEGORIES = ["Work", "Personal", "Finance", "Promotions", "Social"]
category_embeddings = embedder.encode(CATEGORIES, convert_to_tensor=True)

# Load sentiment analysis model and tokenizer
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']


def summarize_email(text):
    """Summarize a single email using the BART model."""
    if not isinstance(text, str) or text.strip() == "":
        return "No content"
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def categorize_emails(df):
    """Assign a category to each email based on similarity to predefined labels."""
    df["category"] = "Uncategorized"
    email_texts = df["body"].fillna("").astype(str).tolist()
    email_embeddings = embedder.encode(email_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(email_embeddings, category_embeddings)

    predicted_categories = []
    for i in range(len(email_texts)):
        best_category_idx = cosine_scores[i].argmax().item()
        predicted_categories.append(CATEGORIES[best_category_idx])

    df["category"] = predicted_categories
    return df


def analyze_sentiment(text):
    """Analyze sentiment of the email body."""
    if not isinstance(text, str) or text.strip() == "":
        return "Unknown"
    encoded_input = sentiment_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = sentiment_model(**encoded_input)
        scores = F.softmax(output.logits, dim=1)
        sentiment = SENTIMENT_LABELS[scores.argmax().item()]
    return sentiment


@app.route("/")
def index():
    try:
        df = pd.read_csv("emails.csv")
        df = df[(df["is_unread"] == True) & (df["body"].notna()) & (df["body"].str.strip() != "")]
        df = categorize_emails(df)
        df["sentiment"] = df["body"].apply(analyze_sentiment)
    except FileNotFoundError:
        df = pd.DataFrame()

    categorized_emails = {}
    if not df.empty:
        categorized = df.groupby("category")
        for cat, group in categorized:
            group = group.reset_index()
            categorized_emails[cat] = group.to_dict(orient="records")

    return render_template("categorize1.html", categorized_emails=categorized_emails)


@app.route("/summarize/<int:email_id>")
def summarize(email_id):
    try:
        df = pd.read_csv("emails.csv")
    except FileNotFoundError:
        return "Email file not found", 404

    df = df.reset_index()
    if email_id >= len(df):
        return "Invalid email ID", 404

    email = df.iloc[email_id]
    body = email.get("body", "")
    summary = summarize_email(body)
    sentiment = analyze_sentiment(body)

    return render_template("summary1.html", email=email, summary=summary, sentiment=sentiment)

@app.route("/add_category")
def add_categoty():
    return render_template("categorize.html")


if __name__ == "__main__":
    app.run(debug=True)
