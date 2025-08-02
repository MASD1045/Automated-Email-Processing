import os.path
import csv
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def authenticate_gmail():
    """Authenticate user and return Gmail service."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=8080)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def fetch_all_emails():
    """Fetch as many emails as possible and save them per user."""
    service = authenticate_gmail()
    try:
        profile = service.users().getProfile(userId="me").execute()
        user_email = profile["emailAddress"]
        filename = f"emailss.csv"

        print(f"Fetching emails for {user_email}...")
        results = service.users().messages().list(userId="me", maxResults=500).execute()
        messages = results.get("messages", [])

        emails = []
        for msg in messages:
            msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
            headers = msg_data["payload"]["headers"]
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
            sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")
            snippet = msg_data.get("snippet", "")
            
            # Decode email body (if available)
            body = ""
            if "parts" in msg_data["payload"]:
                for part in msg_data["payload"]["parts"]:
                    if part["mimeType"] == "text/plain":
                        body_data = part["body"].get("data", "")
                        if body_data:
                            body = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")

            emails.append([sender, subject, snippet, body])

        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Sender", "Subject", "Snippet", "Body"])
            writer.writerows(emails)

        print(f"Emails saved")

    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    fetch_all_emails()
