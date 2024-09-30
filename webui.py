import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and TF-IDF vectorizer
model = joblib.load(r'Emotions-Detection-From_Text/model.pkl')
tfidf_vectorizer = joblib.load(r'C:\Users\ali\Documents\ENV WORKS\Emotion Detection\Emotions-Detection-From_Text\vectorizer.pkl')  # Assuming you saved the vectorizer

# Emotion mapping
emotions = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Function to preprocess text using the TF-IDF vectorizer
def preprocess_text(text):
    # Preprocessing and vectorization using the loaded TF-IDF vectorizer
    transformed_text = tfidf_vectorizer.transform([text])
    return transformed_text

# Streamlit UI
st.title("Emotion Prediction from Text")

# User input for text
user_input = st.text_area("Enter the text for emotion prediction:")

# When the user clicks the "Predict" button
if st.button("Predict"):
    if user_input:
        # Preprocess the text
        processed_text = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(processed_text)

        # Get the numeric prediction and emotion label
        predicted_numeric_value = prediction[0]
        predicted_emotion = emotions.get(predicted_numeric_value, "Unknown emotion")

        # Display results
        st.write(f"Predicted Numeric Value: {predicted_numeric_value}")
        st.write(f"Predicted Emotion: {predicted_emotion}")
    else:
        st.write("Please enter some text to get a prediction.")
