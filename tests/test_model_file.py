import os

def test_model_file_exists():
    """
    CI test: checks if model file exists after running main.py.
    """
    assert os.path.exists("models/hotel_cancellation_model.joblib"), \
        "Model file not generated! Run main.py before tests."
