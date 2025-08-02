import pandas as pd
import re

def parse_into_emails(messages):
    """Processes raw emails into structured format."""
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from': map_to_list(emails, 'from'),
        'subject': map_to_list(emails, 'subject'),
        'date': map_to_list(emails, 'date')
    }

def parse_raw_message(raw_message):
    """Extracts fields from raw email text."""
    lines = raw_message.split('\n')
    email = {}
    message = ''
    
    keys_to_extract = ['from', 'to', 'subject', 'date']
    
    for line in lines:
        if ':' not in line:  
            # Handle body text
            message += line.strip() + " "  # Preserve spaces between lines
            email['body'] = message.strip()
        else:
            pairs = line.split(':', 1)  # Split only at the first colon
            key = pairs[0].lower().strip()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val

    return email

def map_to_list(emails, key):
    """Extracts a list of values for a given key."""
    return [email.get(key, '') for email in emails]

# Load dataset
email_subset = pd.read_csv("emails.csv")  # Update file name

# Process emails
email_df = pd.DataFrame(parse_into_emails(email_subset["message"]))

# Remove unnecessary \n and extra spaces
email_df = email_df.map(lambda x: re.sub(r'\s+', ' ', str(x)).strip() if isinstance(x, str) else x)

# Save to a new CSV file
email_df.to_csv("processed_enron_emails.csv", index=False)

print("Email parsing complete! Saved as processed_enron_emails.csv")
print(email_df.head())
