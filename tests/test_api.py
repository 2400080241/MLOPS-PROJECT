\
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_api_is_running():
    response = client.get("/docs")
    assert response.status_code == 200


def test_crop_prediction():
    payload = {
        "nitrogen": 90,
        "phosphorus": 42,
        "potassium": 43,
        "temperature": 25.5,
        "humidity": 80.0,
        "soil_ph": 6.5,
        "rainfall": 200.0,
        "soil_moisture": 30.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert "recommended_crop" in result
    assert isinstance(result["recommended_crop"], str)
    assert result["recommended_crop"] != ""
