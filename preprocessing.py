import re

def clean_text(text):

    # Convert text into lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Sample text
text = "I LOVE this Product!!!"

# Clean the text
result = clean_text(text)

print("Original Text:", text)
print("Cleaned Text:", result)