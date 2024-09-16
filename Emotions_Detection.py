# Importing dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string


# Loading Dataset 

df = pd.read_csv('emotions.csv')

# Checking if there is null Values

df.isnull().sum()

# Preprocessing Text 

lemm = nltk.WordNetLemmatizer()
nltk.download('punkt')

def preprocess_text(texts):
    # Check if input is a single string or a list of strings
    if isinstance(texts, str):
        texts = [texts]

    # Process each text in the list
    processed_texts = []
    for text in texts:
        text = text.lower()                                               # Convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = nltk.word_tokenize(text)                                   # Tokenize the text
        processed_texts.append(' '.join(text))                            # Join tokens back to a single string

    return processed_texts

process_text = df['text'].apply(preprocess_text)


# Spliting the Data 

x = df['text']
y = df['label']

# Feature Extraction using TF-IDF

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)

# Spliting the data into test and train set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Training the data on Logistic Regression 

model = LogisticRegression()
model.fit(x_train, y_train)

# Model Evaluation 

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Make predictions on the test data
y_pred = model.predict(x_test)

# Evaluate the model
print(f"Accuracy: , {accuracy_score(y_test, y_pred)*100:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Testing Model on Custom Input 

text = ["The constant noise outside is making me very angry."]
text_processed = preprocess_text(text)
text_tfidf = vectorizer.transform(text_processed)
prediction = model.predict(text_tfidf)

emotions = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Assuming prediction is a list or array with the first element being the predicted label
predicted_emotion = emotions.get(prediction[0], "Unknown emotion")

# Print the predicted emotion
print(predicted_emotion)

# Optionally, return or use the predicted emotion elsewhere
prediction[0]
