#Imports testing utilities and the application for integration testing
import pytest
from fastapi.testclient import TestClient
from app import app

#Defines a standard valid payload for reuse across multiple test cases
valid_payload = {
    "pickup_hour": 14,
    "pickup_day_of_week": 3,
    "is_weekend": False,
    "trip_duration_minutes": 15.5,
    "trip_speed_mph": 20.0,
    "log_trip_distance": 1.6,
    "fare_per_mile": 2.5,
    "fare_per_minute": 0.8,
    "PU_Borough": "Manhattan",
    "DO_Borough": "Brooklyn",
    "passenger_count": 2,
    "trip_distance": 5.0,
    "fare_amount": 15.0,
    "total_amount": 18.0
}

#Sets up a client fixture to trigger startup events and model loading
@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

#Verifies the health check endpoint returns success and confirms model loading
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model loaded"] is True

#Verifies that a valid single prediction request returns the expected JSON structure
def test_predict_valid(client):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "prediction_id" in data
    assert data["model_version"] == "1"

#Verifies that the batch endpoint correctly processes multiple records
def test_predict_batch(client):
    batch_payload = {"records": [valid_payload, valid_payload]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2

#Verifies that the API rejects invalid input values with an HTTP 422 error
def test_predict_invalid_input(client):
    invalid_payload = valid_payload.copy()
    invalid_payload["pickup_hour"] = 25
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

#Verifies that the API correctly rejects trips with a distance of zero
def test_predict_edge_case_zero_distance(client):
    edge_case_payload = valid_payload.copy()
    edge_case_payload["trip_distance"] = 0
    response = client.post("/predict", json=edge_case_payload)
    assert response.status_code == 422