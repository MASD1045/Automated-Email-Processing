from flask import Flask, render_template, redirect, url_for
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import os

app = Flask(__name__)

# Load BART model and tokenizer for summarization
model_name = "philschmid/bart-large-cnn-samsum"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

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

@app.route("/")
def index():
    """Display original emails from CSV with Summarize button."""
    try:
        df = pd.read_csv("emails.csv")
        filtered_emails = df[
            (df["is_unread"] == True) &
            (df["body"].notna()) &
            (df["body"].str.strip() != "")
        ].copy()
    except FileNotFoundError:
        df = pd.DataFrame()
    
    return render_template("index1.html", emails=filtered_emails.to_dict(orient="records"))

@app.route("/summarize/<int:email_id>")
def summarize(email_id):
    try:
        df = pd.read_csv("emails.csv")
    except FileNotFoundError:
        return "Email file not found", 404

    if email_id >= len(df):
        return "Invalid email ID", 404

    email = df.iloc[email_id]

    body = email.get("body", "")
    if not isinstance(body, str) or body.strip() == "":
        summary = "No content"
    else:
        summary = summarize_email(body)

    return render_template("summary1.html", email=email, summary=summary)

@app.route("/categorized-emails")
def categorized_emails():
    try:
        df = pd.read_csv("predictions\categorized_emails.csv")
    except FileNotFoundError:
        return "Categorized email file not found."

    # You can limit how many emails you want to display
    grouped = df.groupby("Predicted_Category")["body"].apply(list).reset_index()
    return render_template("categorization.html", grouped=grouped)

@app.route("/sentiments-emails")
def sentiments_emails():
    try:
        df = pd.read_csv("actual_sentiments\emails_with_sentiment(roberta_binary).csv")
    except FileNotFoundError:
        return "Categorized email file not found."

    # You can limit how many emails you want to display
    grouped = df.groupby("predicted_sentiment")["body"].apply(list).reset_index()
    return render_template("sentiment.html", grouped=grouped)


if __name__ == "__main__":
    app.run(debug=True)
