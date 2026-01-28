import pickle

CENTROID = "gmail_centroids.pkl"

with open(CENTROID, "rb") as f:
    label_centroids = pickle.load(f)

print("✅ Gmail centroids loaded")
print("Available labels:", list(label_centroids.keys()))
