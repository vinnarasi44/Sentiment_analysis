import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Load Dataset

data = pd.read_csv("dataset/sentiment_data.csv")

print("Dataset Loaded Successfully!")


# Step 2: Text Cleaning

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


data["text"] = data["text"].apply(clean_text)


# Step 3: Separate Input and Output

X = data["text"]
y = data["sentiment"]


# Step 4: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Step 5: Create ML Pipeline

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Step 6: Train Model

model.fit(X_train, y_train)

print("Model Training Completed!")


# Step 7: Test Model

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))


# Step 8: Save Model

joblib.dump(model, "model/sentiment_model.pkl")

print("\nModel Saved Successfully!")