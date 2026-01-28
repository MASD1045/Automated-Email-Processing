import os
import csv
import base64
import threading
import re
from flask import Flask, redirect, url_for, render_template, jsonify, request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import BatchHttpRequest
from bs4 import BeautifulSoup, Comment
from readability import Document
from summarizer import summarize_text
from sentiment import analyze_sentiment
from categorization import categorize_emails
from centroids import label_centroids

app = Flask(__name__)
app.secret_key = "replace_with_a_real_secret"
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # allow HTTP for testing

# config
CLIENT_SECRETS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CSV_FILE = "emails.csv" 
MAX_EMAILS = 10000
BATCH_SIZE = 100

# progress state
progress = {"total": 0, "fetched": 0, "status": "not_started"}
emails = []


# ----------------------------------------------------
# HARD CLEAN METHOD — Removes EVERYTHING except text
# ----------------------------------------------------
def hard_clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # remove scripts, styles and metadata
    for tag in soup(["script", "style", "meta", "link", "title", "head", "svg", "xml"]):
        tag.decompose()

    # remove comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        c.extract()

    # remove inline CSS
    for tag in soup():
        for attr in ["style", "class", "id"]:
            if attr in tag.attrs:
                del tag.attrs[attr]

    # unwrap tables
    for tag in soup.find_all(["table", "tr", "td", "tbody", "thead", "tfoot"]):
        tag.unwrap()

    # extract readable text
    text = soup.get_text(separator="\n")

    # cleanup spacing
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = "\n".join(lines)

    return cleaned.strip()


# ---------------- REMOVE EXTRA SPACES ----------------
def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join(line for line in text.split("\n") if line.strip())

    return text.strip()


# ---------------- CLEAN HTML + EXTRACT MAIN CONTENT ----------------
def extract_main_text(payload):
    html = ""

    # direct html
    if payload.get("body", {}).get("data"):
        html = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
    else:
        # find HTML inside multipart
        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
                html = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                break

    if not html:
        return ""

    doc = Document(html)
    main_html = doc.summary()

    cleaned = hard_clean_html(main_html)
    return cleaned


# ---------------- PREFER PLAIN TEXT ----------------
def extract_body(payload):
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            raw = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
            return clean_text(raw)

    raw_html_text = extract_main_text(payload)
    return clean_text(raw_html_text)


# ---------------- Gmail Helpers ----------------
def get_service_from_token():
    if not os.path.exists(TOKEN_FILE):
        return None
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    return build("gmail", "v1", credentials=creds)


# ---------------- EXTRACT ONLY CATEGORY ----------------
def extract_category_only(labels):
    """
    Extracts Gmail category from labelIds list, e.g. CATEGORY_UPDATES -> UPDATES
    """
    for label in labels:
        if label.startswith("CATEGORY_"):
            return label.split("_", 1)[1]
    return ""


# ---------------- BACKGROUND EMAIL FETCHER ----------------
def background_fetch_emails():
    global progress, emails

    try:
        svc = get_service_from_token()
        if not svc:
            progress["status"] = "error"
            return

        progress["status"] = "fetching"
        emails = []

        # collect message IDs
        message_ids = []
        page_token = None

        while len(message_ids) < MAX_EMAILS:
            resp = svc.users().messages().list(
                userId="me",
                maxResults=500,
                pageToken=page_token
            ).execute()

            msgs = resp.get("messages", [])
            page_token = resp.get("nextPageToken")

            message_ids.extend([m["id"] for m in msgs])

            if not page_token:
                break

        message_ids = message_ids[:MAX_EMAILS]

        progress["total"] = len(message_ids)
        progress["fetched"] = 0

        # batch callback
        def batch_callback(request_id, response, exception):
            global emails, progress

            if exception:
                emails.append({
                    "ID": str(request_id),
                    "From": "",
                    "To": "",
                    "Subject": "",
                    "Message-ID": "",
                    "Labels": "",
                    "Body": ""
                })
            else:
                payload = response.get("payload", {})
                headers = payload.get("headers", [])

                def get_h(n):
                    for h in headers:
                        if h.get("name", "").lower() == n.lower():
                            return h.get("value", "")
                    return ""

                # extract only Gmail category
                labels = response.get("labelIds", [])
                category = extract_category_only(labels)

                body = extract_body(payload)

                emails.append({
                    "ID": response.get("id", ""),
                    "From": get_h("From"),
                    "To": get_h("To"),
                    "Subject": get_h("Subject"),
                    "Message-ID": get_h("Message-ID"),
                    "Labels": category,  # only category
                    "Body": body
                })

            progress["fetched"] += 1

        # execute batches
        for i in range(0, len(message_ids), BATCH_SIZE):
            batch = svc.new_batch_http_request(callback=batch_callback)
            chunk = message_ids[i:i + BATCH_SIZE]

            for mid in chunk:
                batch.add(
                    svc.users().messages().get(userId="me", id=mid, format="full"),
                    request_id=mid
                )

            batch.execute()

        # save CSV automatically
        keys = ["ID", "From", "To", "Subject", "Message-ID", "Labels", "Body"]
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(emails)

        progress["status"] = "done"

    except Exception as exc:
        print("Background fetch error:", exc)
        progress["status"] = "error"


# ---------------- Flask Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/authorize")
def authorize():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("oauth2callback", _external=True),
    )
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
    return redirect(auth_url)


@app.route("/oauth2callback")
def oauth2callback():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("oauth2callback", _external=True),
    )
    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    return redirect(url_for("fetch_emails_route"))

@app.route("/fetch_emails")
def fetch_emails_route():
    if progress["status"] == "done":
        return redirect(url_for("emails_page"))

    if progress["status"] != "fetching":
        threading.Thread(
            target=background_fetch_emails,
            daemon=True
        ).start()

    return render_template("fetching.html")


@app.route("/status")
def status():
    return jsonify(progress)

@app.route("/emails")
def emails_page():
    if progress["status"] != "done":
        return "Emails still fetching. Please wait."
    return render_template("emails.html", emails=emails)

@app.route("/summarize_email", methods=["POST"])
def summarize_email():
    email_id = request.form.get("email_id")

    for email in emails:
        if email["ID"] == email_id:
            email["Summary"] = summarize_text(email["Body"])
            break

    return redirect(url_for("emails_page"))

@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment_route():
    email_id = request.form.get("email_id")

    for email in emails:
        if email["ID"] == email_id:
            if email.get("Body") and email["Body"].strip():
                email["Sentiment"] = analyze_sentiment(email["Body"])
            break

    return redirect(url_for("sentiment_page"))

@app.route("/sentiment")
def sentiment_page():
    filtered_emails = [email for email in emails if email.get("Body") and email["Body"].strip()]
    return render_template("sentiment.html", emails=filtered_emails)


@app.route("/categorize", methods=["GET", "POST"])
def categorize():
    ask_examples = False
    unknown_categories = []
    results = None

    if request.method == "POST":
        categories = request.form.get("categories", "")
        user_categories = [c.strip().upper() for c in categories.split(",")]

        # Collect example inputs if present
        user_examples = {}
        for key in request.form:
            if key.startswith("examples_"):
                cat = key.replace("examples_", "")
                examples = request.form[key].split("||")
                user_examples[cat] = [e.strip() for e in examples if e.strip()]

        results, unknown_categories = categorize_emails(
            emails,
            label_centroids,
            user_categories,
            user_examples if user_examples else None
        )

        if unknown_categories and not user_examples:
            ask_examples = True
        else:
            ask_examples = False
        
        results = [r for r in results if r["Predicted_Category"] in user_categories]

    return render_template(
        "categorization.html",
        ask_examples=ask_examples,
        unknown_categories=unknown_categories,
        categories=request.form.get("categories", ""),
        results=results
    )

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
