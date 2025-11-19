import streamlit as st
import pickle
import numpy as np
import tensorflow

# -----------------------------
# Define the same class used during saving
# -----------------------------
class SimpleModel:
    def __init__(self, keras_model=None):
        self.keras_model = keras_model

    def predict(self, X):
        preds = self.keras_model.predict(X)
        return preds.argmax(axis=1)

# -----------------------------
# Load files
# -----------------------------
@st.cache_resource
def load_model():
    with open("simple_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

@st.cache_resource
def load_preprocessor():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor

model = load_model()
tokenizer = load_tokenizer()

import spacy
import re
import pickle

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text):
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower().strip()
        return text

    def tokenize_spacy(self, text):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop]

    def transform(self, texts):
        cleaned = [self.clean_text(t) for t in texts]
        tokens = [" ".join(self.tokenize_spacy(t)) for t in cleaned]
        return tokens

preprocessor = TextPreprocessor()
# -----------------------------
# UI
# -----------------------------
st.title("💬 Sentiment Prediction App")
st.write("Enter your text below to predict sentiment as Positive, Neutral, or Negative.")

text = st.text_area("Enter text here:")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text!")
    else:
        processed = preprocessor.transform([text]) if hasattr(preprocessor, "transform") else [text]
        seq = tokenizer.texts_to_sequences(processed)

        seq = np.array(seq, dtype=np.float32).reshape(1, -1)

        pred = model.predict(seq)[0]

        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        result = label_map.get(pred, "Unknown")

        st.success(f"Prediction: **{result}**")
