import pytest
from predictor import predict_disease, classes, load_image_for_model, mobilenet_model
from tensorflow.keras.applications.mobilenet import preprocess_input


# Sample test image
TEST_IMAGE = r"C:\Users\YV\Downloads\redlust.jpg"


# ==========================
# Test 1: Prediction runs
# ==========================
def test_prediction_runs():

    disease, confidence = predict_disease(TEST_IMAGE)

    assert disease is not None
    assert confidence is not None


# ==========================
# Test 2: Correct return types
# ==========================
def test_prediction_types():

    disease, confidence = predict_disease(TEST_IMAGE)

    assert isinstance(disease, str)
    assert isinstance(confidence, float)


# ==========================
# Test 3: Class must exist
# ==========================
def test_prediction_class_valid():

    disease, _ = predict_disease(TEST_IMAGE)

    assert disease in classes


# ==========================
# Test 4: Confidence range
# ==========================
def test_confidence_range():

    _, confidence = predict_disease(TEST_IMAGE)

    assert 0 <= confidence <= 1


# ==========================
# Test 5: Image preprocessing
# ==========================
def test_image_preprocessing():

    img = load_image_for_model(
        TEST_IMAGE,
        mobilenet_model,
        preprocess_input
    )

    assert img.shape[0] == 1