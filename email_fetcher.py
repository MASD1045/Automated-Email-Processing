import os
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
    """Fetch emails from Gmail and save them as emails.csv."""
    service = authenticate_gmail()
    try:
        profile = service.users().getProfile(userId="me").execute()
        user_email = profile["emailAddress"]
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

            email_status = msg_data["labelIds"]
            is_unread = "UNREAD" in email_status

            emails.append({
                "sender": sender,
                "subject": subject,
                "snippet": snippet,
                "body": body,
                "is_unread": is_unread
            })
        
        # Save emails to CSV file
        with open("emails.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["sender", "subject", "snippet", "body", "is_unread"])
            writer.writeheader()  # Write the header row
            for email in emails:
                writer.writerow(email)  # Write each email to the CSV file

        print("Emails saved as emails.csv")
        return emails

    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

if __name__ == "__main__":
    fetch_all_emails()
