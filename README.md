# sentiment-Analysis-by-Simple-RNN

## 🎬 IMDB Movie Review Sentiment Analysis using Simple RNN
This project demonstrates a sentiment analysis model built using a Simple RNN architecture to classify IMDB movie reviews as positive or negative. The model is trained on the IMDB dataset available through TensorFlow Datasets, and is deployed via a Streamlit web application for interactive user input and real-time predictions.

## 📌 Project Highlights
Dataset: IMDB movie reviews dataset from TensorFlow Datasets.
## Preprocessing:
Tokenized and padded sequences using pad_sequences from Keras.
Maximum vocabulary size: 10,000
Input sequence length: 500
## Model Architecture:
Embedding Layer: Converts words into dense vectors of dimension 128.
Simple RNN Layer: Captures sequential dependencies in the text.
Dense Output Layer: Outputs a binary classification (positive or negative sentiment).
## Training:
Implemented EarlyStopping to prevent overfitting and optimize training time.
Model is saved for future inference and reuse.
## Deployment:
Built a Streamlit app to allow users to input custom movie reviews.
## The app returns:
Predicted Sentiment (Positive/Negative)
Confidence Score (Probability)
## 🚀 How It Works
User enters a movie review in the Streamlit interface.
The input is preprocessed and passed to the trained RNN model.
The model predicts the sentiment and displays the result along with the probability score.
## 📂 Use Case
## This application is ideal for:

1. Understanding audience sentiment from movie reviews.
2. Demonstrating the effectiveness of RNNs in natural language processing tasks.
3. Serving as a foundation for more advanced NLP models like LSTM, GRU, or Transformers.
