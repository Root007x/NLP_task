import streamlit as st 
import pickle
from pathlib import Path
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")


VECTOR_MODEL = Path("model/vectorizer.pkl")
MODEL = Path("model/model.pkl")

tf_idf = pickle.load(open(VECTOR_MODEL,"rb"))
model = pickle.load(open(MODEL, "rb"))

list_stopwords = set(stopwords.words("english")) 

# define preprocessing function
def text_cleaning(text : str):
    # remove html
    text = re.sub(re.compile("<.*?>"), '', text)
    
    # remove punctuation
    text = text.translate(str.maketrans("","",string.punctuation))
    
    # remove numbers
    text = re.sub(r'\d+','',text)
    
    # lowercase
    text = text.lower()
    
    # remove stopwatch
    text = " ".join([word for word in text.split() if word not in list_stopwords])
        
    return text



st.title("Movie Review Classifier")

# take input from user
input_review = st.text_area("Enter the movie review")

# preprocessed text
transformed_input = text_cleaning(input_review)

if st.button("Classify Review"):
    # vectorize the text
    vector_input = tf_idf.transform([transformed_input])
    # predict
    result = model.predict(vector_input)[0]

    if result == 1:
        st.success("Review sentiment : Positive")
    else:
        st.success("Review sentiment : Negative")




