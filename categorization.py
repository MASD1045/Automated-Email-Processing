import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib

# Load your emails from JSON
def load_emails(path="emails.csv"):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

# Prepare data
df = load_emails()
df = df[df['label'].isin(['Work', 'Personal', 'Spam'])]  # Example categories
df['text'] = df['subject'] + ' ' + df['body']

# Encode using BERT
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & good
X = model.encode(df['text'].tolist(), show_progress_bar=True)

# Encode labels
label_map = {label: i for i, label in enumerate(df['label'].unique())}
y = df['label'].map(label_map)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# Save model
joblib.dump(clf, 'model/logistic_model.pkl')
joblib.dump(label_map, 'model/label_map.pkl')
