import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# 🔁 Load all prediction CSVs from a folder
folder_path = "predictions/"  # <-- change this to your actual folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 📁 Load predictions into a dictionary
model_outputs = {}
model_name_mapping = {
    'categorized_emails_with_bert': 'BERT',  # Example: Map 'bert_model' to 'BERT Model'
    'categorized_emails_with_distilbert': 'DistilBERT',  # Example: Map 'distilbert_model' to 'DistilBERT Model'
    'categorized_emails_with_roberta': 'RoBERTa',  # Example: Map 'roberta_model' to 'RoBERTa Model'
    'categorized_emails_with_xlnet': 'XLNet',  # Example: Map 'xlnet_model' to 'XLNet Model'
    'categorized_emails_with_albert': 'Albert',
    'categorized_emails': 'Sentence BERT',
    'categorized_emails_with_bart': 'BART'
    # Add other mappings as necessary
}

# Load predictions
for file in csv_files:
    model_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    if "Predicted_Category" in df.columns:
        model_outputs[model_name] = df["Predicted_Category"]
    else:
        print(f"⚠️ 'Predicted_Category' column not found in {file}")

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

# Use the mapping to rename the axes (rows and columns) to more user-friendly model names
custom_model_names = [model_name_mapping.get(name, name) for name in model_names]
agreement_matrix.index = custom_model_names
agreement_matrix.columns = custom_model_names

# 📈 Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix.astype(float), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Pairwise Agreement Rate Between Different Models for Categorization")
plt.tight_layout()

# 💾 Save the plot to a file
plt.savefig("pairwise_agreement_heatmap.png", dpi=300)  # You can change format here
print("✅ Heatmap saved as 'pairwise_agreement_heatmap.png'.")

plt.show()
