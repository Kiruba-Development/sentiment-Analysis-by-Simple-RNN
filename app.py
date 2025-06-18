import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model


## Load the model

model = load_model('SimpleRNN/simple_rnn_imdb.h5')

## Load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index ={value : key for key, value in word_index.items()}


## Step 2 helper funtion
## Function to decode reviews
def decode_review(encoded_review):
    return  ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


## Funtion to preprocess the use Text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ## Predict the result with funtion

# def predict_sentiment(review):

#     response = preprocess_text(review)
#     prediction = model.predict(response)

#     if prediction[0] > 0.5:
#         result = "Positive"
#         return prediction[0][0], result
#     else:
#         result = "Negative"
#         return prediction[0][0], result
        
       


## Streamlit

st.title("Sentiment analysis of IMDB movie review by using the Simple RNN")

review = st.text_area("Enter the review")

button = st.button("Classify")

if button:
    if len(review) < 5 :
        st.write("Review is not enough to classify")

    else:

        preprocces_input = preprocess_text(review)

        prediction = model.predict(preprocces_input)

        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        prob = prediction[0][0]

        ## Display the result

        st.write(f'The movie got {sentiment} Review and the probability is {prob}')

else:
    st.write("Please write the movie review")
