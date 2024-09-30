# Emotion Detection using Logistic Regression

This project demonstrates emotion detection from text using Natural Language Processing (NLP) techniques and a machine learning model based on Logistic Regression. The model predicts one of the following emotions: **sadness**, **joy**, **love**, **anger**, **fear**, and **surprise**.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Custom Prediction](#custom-prediction)
- [Contribution](#Contributing)


## Project Overview

This project involves preprocessing text data, training a logistic regression model, and predicting the emotions present in a given text. The workflow includes:
1. Text preprocessing (lowercasing, removing punctuation, and tokenization).
2. Feature extraction using **TF-IDF**.
3. Training a **Logistic Regression** model.
4. Evaluating the model on test data.
5. Predicting emotions for custom input.

## Technologies Used

- **Python**: Core programming language.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **Matplotlib & Seaborn**: For data visualization.
- **NLTK**: For natural language processing and text preprocessing.
- **Scikit-learn**: For machine learning model development and evaluation.

## Dataset

The dataset used in this project (`emotions.csv`) contains text data along with corresponding emotion labels. Each record has:
- `text`: The input text.
- `label`: The emotion label corresponding to the text.

Make sure to place the dataset file in the project directory.
## Usage : 

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/emotion-detection.git
   

2. Navigate:
   ```bash
   cd emotion-detection

## Contributing
If you'd like to contribute, please fork the repository and make changes as needed. Submit a pull request when you're done.




