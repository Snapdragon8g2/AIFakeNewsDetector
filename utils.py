# Text preprocessing and data loading
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Clean, tokenize, and remove stop words from raw article text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower().strip()

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)

# Balance dataset by downsampling/oversampling
def balance_dataset(fake, true, sample_size=None):
    if sample_size:
        fake = fake.sample(n=sample_size, random_state=42)
        true = true.sample(n=sample_size, random_state=42)

    print(f"Fake news: {len(fake)}, True news: {len(true)}")
    combined_data = pd.concat([fake, true], ignore_index=True)
    print(f"Combined data size: {len(combined_data)}")
    
    return combined_data.sample(frac=1, random_state=42)