import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add project root so imports from SIC work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SIC.api.main import app, load_model

# Force model load (pytest may not always trigger startup events)
load_model()

# Create a TestClient instance
client = TestClient(app)


def test_health_check():
    """
    Tests if the /health endpoint is working correctly.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    """
    Tests if the /model_info endpoint returns valid model metadata.
    """
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()

    assert "model_type" in data
    assert "classes" in data
    assert isinstance(data["classes"], list)


def test_predict_endpoint():
    """
    Tests the /predict endpoint with a sample image.
    """
    image_path = Path("SIC/data/seg_test/seg_test/buildings/20064.jpg")

    # Ensure the image exists
    assert image_path.is_file(), f"Test image not found at {image_path}"

    with open(image_path, "rb") as image_file:
        files = {"file": (image_path.name, image_file, "image/jpeg")}
        response = client.post("/predict", files=files)

    # Assert that the request was successful
    assert response.status_code == 200

    data = response.json()
    # Validate expected response keys
    for key in ["predicted_class", "confidence_score", "uncertainty_estimate", "description"]:
        assert key in data

    # Optional: Check expected class (only if you are confident about the label)
    # assert data["predicted_class"] == "buildings"
