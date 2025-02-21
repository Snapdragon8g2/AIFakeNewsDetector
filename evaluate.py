import pickle
import pandas as pd
from sklearn.metrics import classification_report

# Load the model and vectorizer from train.py
with open("models/fake_news_model.pkl", "rb") as model_file:
    model, vectorizer = pickle.load(model_file)

def evaluate_model():
    # Load and preprocess test data
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")
    fake['label'] = 0
    true['label'] = 1

    data = pd.concat([fake, true], ignore_index=True)
    X_test = vectorizer.transform(data['text'])
    y_test = data['label']

    # Evaluate
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    evaluate_model()
