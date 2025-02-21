import streamlit as st
import re
import nltk
import requests
import pickle
from newspaper import Article
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from transformers import pipeline
from utils import preprocess_text, balance_dataset

#  No longer pretrained model (switched from DistilBERT to fine-tuned model on Fake News dataset)
with open("models/fake_news_model.pkl", "rb") as model_file:
    model, vectorizer = pickle.load(model_file)

def extract_article_content(url):
    """
    Given a URL, scrape and extract the main text of the article.
    Uses newspaper3k for convenience.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        # Maybe log here
        return None

def main():
    st.title("Fake News Detector")
    st.markdown("Enter a news article URL to classify it as real or fake.")

    url = st.text_input("Enter the URL of the news article:")

    if url:
        st.write("**Step 1:** Extracting article text...")
        article_content = extract_article_content(url)

        if not article_content:
            st.error("Failed to scrape the article. Please check the URL or try another.")
            return

        st.write("**Step 2:** Preprocessing the article...")
        clean_text = preprocess_text(article_content)

        # Show small sample of the cleaned text
        if len(clean_text.split()) > 0:
            st.success("Preprocessing complete!")
            st.write("**Sample of processed text:**")
            st.write(" ".join(clean_text.split()[:50]) + " ...")
        else:
            st.error("Article text is empty after preprocessing. Possibly a non-English or invalid article.")
            return

        st.write("**Step 3:** Classifying the article...")

        # Vectorize text using our trained vectorizer
        vectorized_text = vectorizer.transform([clean_text])

        # Perform classification and confidence estimation
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        confidence = max(probabilities)

        label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

        st.success(f"**Predicted Label:** {label}")
        st.info(f"**Confidence:** {confidence:.2f}")


if __name__ == "__main__":
    main()
