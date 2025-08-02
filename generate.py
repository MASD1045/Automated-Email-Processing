import pandas as pd

# Sample positive and negative email bodies (20 examples for each category)
positive_examples = [
    "I absolutely love the service! Everything is amazing.",
    "The product exceeded my expectations, very happy with it.",
    "Great job! Excellent work, highly recommended.",
    "I'm extremely satisfied with my purchase. Will buy again.",
    "The team was wonderful to work with, will definitely return.",
    "Fantastic customer support! They were very helpful and friendly.",
    "Highly recommend this product. It works like a charm!",
    "I’m so pleased with my experience. Everything went smoothly.",
    "What a great purchase! It was exactly what I needed.",
    "Wonderful product! I’m so glad I bought it.",
    "This is by far the best experience I’ve had with any company.",
    "The item arrived earlier than expected, and it’s perfect.",
    "I can’t believe how easy it was to make a purchase. Very happy.",
    "The quality of the product is excellent. It’s exactly what I wanted.",
    "Totally impressed with the customer service. Very responsive!",
    "I’ll definitely be coming back to this store in the future.",
    "I love how efficient the service was. Thank you for making this so easy.",
    "The product was as described, and I’m very happy with it.",
    "Amazing! I’m so glad I made this purchase, it’s been fantastic.",
    "The support team was outstanding in helping me resolve an issue."
]

negative_examples = [
    "I’m very disappointed with the product. It didn’t work as expected.",
    "Horrible experience, will never buy from here again.",
    "The customer support was terrible. I will not recommend this.",
    "The service was awful. It took too long and wasn’t worth it.",
    "I regret buying this. It broke after a week.",
    "Terrible quality. Not at all like the description on the website.",
    "The delivery was delayed, and I received the wrong item.",
    "This product was a waste of money. It’s defective.",
    "I’m not happy with my purchase at all. Will return it.",
    "Very dissatisfied with the product. It didn’t meet my expectations.",
    "I can’t believe how bad the customer service was. I will never use this company again.",
    "I’ve had issues since I got this. Definitely won’t buy again.",
    "The quality is poor, and I am very upset with my purchase.",
    "The order was wrong and no one is responding to my messages.",
    "I waited for weeks for this and it’s totally not what I wanted.",
    "The product broke after only a few uses. So frustrating!",
    "Very unhappy with my experience. I’ll be looking elsewhere next time.",
    "I wouldn’t recommend this to anyone. Terrible experience.",
    "I’m extremely disappointed. The item did not perform as advertised.",
    "This is by far the worst purchase I’ve made."
]

# Combine positive and negative examples
email_bodies = positive_examples + negative_examples
labels = ["positive"] * 20 + ["negative"] * 20  # 20 positive and 20 negative labels

# Create DataFrame
df = pd.DataFrame({
    "body": email_bodies,
    "sentiment": labels
})

# Save to CSV
df.to_csv("email_sentiment_data.csv", index=False, encoding="utf-8")
print("Synthetic dataset with 40 emails (20 positive and 20 negative) has been generated and saved as 'synthetic_email_sentiment_data.csv'.")
