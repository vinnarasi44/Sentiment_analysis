import joblib
import re

# Load trained model
model = joblib.load("model/sentiment_model.pkl")

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Get user input
user_text = input("Enter your review: ")

# Clean input
cleaned_text = clean_text(user_text)

# Predict sentiment
prediction = model.predict([cleaned_text])

# Display result
print("\nSentiment Analysis Result:")
print("Input:", user_text)
print("Prediction:", prediction[0])