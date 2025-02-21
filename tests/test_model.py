import pytest
import joblib
import os
import numpy as np

MODEL_PATH = "models/fake_news_model.pkl"

@pytest.fixture
def model():
    """Load the trained model"""
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        model, vectorizer = joblib.load(f)
    return model, vectorizer

def test_model_loading(model):
    """Test that the model loads correctly"""
    model, vectorizer = model
    assert model is not None
    assert vectorizer is not None

def test_model_prediction(model):
    """Test that the model makes predictions"""
    model, vectorizer = model
    sample_text = ["This is a fake news article."]

    vectorized_text = vectorizer.transform(sample_text)
    prediction = model.predict(vectorized_text)

    # Convert np.int64 to Python int before assertion to get rid of warning
    assert isinstance(int(prediction[0]), int)
    assert prediction[0] in [0, 1]
