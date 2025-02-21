import pytest
import subprocess
import time
import requests

def test_app_runs():
    """Test if the Streamlit app starts and is accessible"""

    process = subprocess.Popen(["streamlit", "run", "app.py", "--server.headless", "true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time.sleep(20)

    try:
        response = requests.get("http://localhost:8501")
        assert response.status_code == 200, "Streamlit did not start successfully."

    except requests.exceptions.ConnectionError:
        pytest.fail("Streamlit app did not start correctly.")

    finally:
        process.kill()  # Terminate after process or test errors
