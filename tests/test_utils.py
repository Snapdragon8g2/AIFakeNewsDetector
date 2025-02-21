
import pytest
import pandas as pd
import os
import sys
from utils import balance_dataset

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from utils import preprocess_text, balance_dataset

def test_balance_dataset():
    """Test dataset balancing function"""
    fake_data = pd.DataFrame({"text": ["Fake news example"] * 10})
    true_data = pd.DataFrame({"text": ["Real news example"] * 20})

    balanced_data = balance_dataset(fake_data, true_data, sample_size=10)

    # Debugging
    print(f"Final balanced dataset size: {len(balanced_data)}")
    
    # Ensure balanced dataset has equal samples to get rid of warning
    fake_count = balanced_data[balanced_data["text"] == "Fake news example"].shape[0]
    true_count = balanced_data[balanced_data["text"] == "Real news example"].shape[0]

    assert fake_count == 10, f"Fake sample count should be 10 but got {fake_count}"
    assert true_count == 10, f"True sample count should be 10 but got {true_count}"