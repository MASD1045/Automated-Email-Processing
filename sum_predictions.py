import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# 🔁 Load all prediction CSVs from the "sentiments" folder
folder_path = "actual_sentiments/"  # <-- changed to the "sentiments" folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 📁 Load predictions into a dictionary
model_outputs = {}
model_name_mapping = {
    'emails_with_sentiment(albert_binary)': 'Albert',
    'emails_with_sentiment(bert_binary)': 'BERT',
    'emails_with_sentiment(distilbert)': 'DistilBERT',
    'emails_with_sentiment(roberta_binary)': 'RoBERTa',
    'emails_with_sentiment(xlnet_binary)': 'XLNet'
}

# Load predictions
for file in csv_files:
    model_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    if "predicted_sentiment" in df.columns:
        model_outputs[model_name] = df["predicted_sentiment"]  # ✅ Fixed: Get actual Series, not string
    else:
        print(f"⚠️ 'predicted_sentiment' column not found in {file}")

# ✅ Sanity check
lengths = [len(preds) for preds in model_outputs.values()]
assert len(set(lengths)) == 1, "All models must have predictions for the same number of emails!"

# 📊 Create pairwise agreement matrix
model_names = list(model_outputs.keys())
agreement_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)

for model_a in model_names:
    for model_b in model_names:
        if model_a == model_b:
            agreement_matrix.loc[model_a, model_b] = 1.0
        else:
            a_preds = model_outputs[model_a]
            b_preds = model_outputs[model_b]
            agreement = (a_preds == b_preds).mean()
            agreement_matrix.loc[model_a, model_b] = round(agreement, 4)

# 🏷 Rename model labels using friendly names
custom_model_names = [model_name_mapping.get(name, name) for name in model_names]
agreement_matrix.index = custom_model_names
agreement_matrix.columns = custom_model_names

# 📈 Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix.astype(float), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Pairwise Agreement Rate Between Different Models for Sentiment Analysis")
plt.tight_layout()

# 💾 Save the plot to a fileKs
plt.savefig("pairwise_agreement_heatmap.png", dpi=300)
print("✅ Heatmap saved as 'pairwise_agreement_heatmap.png'.")

plt.show()
