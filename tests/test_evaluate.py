import sys
import os
import pandas as pd
import pytest
from evaluate import evaluate_model

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

@pytest.fixture
def mock_data():
    """Create temporary fake news dataset"""
    fake_data = pd.DataFrame({"text": ["Fake news example"] * 10, "label": [0] * 10})
    true_data = pd.DataFrame({"text": ["Real news example"] * 10, "label": [1] * 10})
    return pd.concat([fake_data, true_data])

def test_evaluate_model(mock_data, monkeypatch):
    """Ensure evaluate_model runs successfully"""
    def mock_read_csv(filepath):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    try:
        evaluate_model()
    except Exception as e:
        pytest.fail(f"Evaluation failed due to {e}")
