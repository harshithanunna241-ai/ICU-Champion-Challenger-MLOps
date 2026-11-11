
"""Tests for ICU deterioration prediction API endpoints."""

import numpy as np
import pytest

from fastapi.testclient import TestClient

from src.api.main import app
from src.api import prediction_routes


client = TestClient(app)


class MockChampionModel:
    """Mock binary classification model for API testing."""

    def predict_proba(self, features):
        return np.array([[0.20, 0.80]])


def valid_observation_request():
    """Return synthetic timestamped ICU observations."""

    return {
        "subject_id": 1001,
        "stay_id": 2001,
        "prediction_time": "2026-11-10T12:00:00",
        "observations": [
            {
                "charttime": "2026-11-10T07:00:00",
                "variable": "heart_rate",
                "value": 85,
            },
            {
                "charttime": "2026-11-10T08:00:00",
                "variable": "systolic_bp",
                "value": 120,
            },
            {
                "charttime": "2026-11-10T08:00:00",
                "variable": "diastolic_bp",
                "value": 80,
            },
            {
                "charttime": "2026-11-10T09:00:00",
                "variable": "temperature",
                "value": 37.0,
            },
            {
                "charttime": "2026-11-10T10:00:00",
                "variable": "spo2",
                "value": 98,
            },
            {
                "charttime": "2026-11-10T11:00:00",
                "variable": "respiratory_rate",
                "value": 18,
            },
        ],
    }


def mock_champion():
    """Return a mock Champion model and its registered feature schema."""

    return (
        MockChampionModel(),
        "Mock Champion",
        ["heart_rate_mean"],
    )


def test_observation_prediction_success(monkeypatch):
    """Valid observations should produce a prediction."""

    monkeypatch.setattr(
        prediction_routes,
        "load_champion_model",
        mock_champion,
    )

    response = client.post(
        "/predict/from-observations",
        json=valid_observation_request(),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["subject_id"] == 1001
    assert result["stay_id"] == 2001
    assert result["prediction"] == 1
    assert result["deterioration_probability"] == 0.8
    assert result["model_name"] == "Mock Champion"
    assert result["status"] == "success"


def test_observation_prediction_missing_identifiers():
    """Missing patient identifiers should be rejected."""

    request = valid_observation_request()
    del request["subject_id"]

    response = client.post(
        "/predict/from-observations",
        json=request,
    )

    assert response.status_code == 422


def test_observation_prediction_invalid_variable():
    """Unsupported observation variables should be rejected."""

    request = valid_observation_request()
    request["observations"][0]["variable"] = "invalid_variable"

    response = client.post(
        "/predict/from-observations",
        json=request,
    )

    assert response.status_code == 422


def test_observation_prediction_invalid_measurement():
    """Non-finite measurement values should be rejected."""

    request = valid_observation_request()
    request["observations"][0]["value"] = "NaN"

    response = client.post(
        "/predict/from-observations",
        json=request,
    )

    assert response.status_code == 422


def test_observation_prediction_no_lookback_data(monkeypatch):
    """Observations outside the six-hour window should be rejected."""

    monkeypatch.setattr(
        prediction_routes,
        "load_champion_model",
        mock_champion,
    )

    request = valid_observation_request()

    for observation in request["observations"]:
        observation["charttime"] = "2026-11-10T05:00:00"

    response = client.post(
        "/predict/from-observations",
        json=request,
    )

    assert response.status_code == 400
    assert "non-finite" in response.json()["detail"]


def test_observation_prediction_without_champion(monkeypatch):
    """Missing Champion model should return HTTP 503."""

    def raise_missing_champion():
        raise RuntimeError(
            "No Champion model is available in the model registry."
        )

    monkeypatch.setattr(
        prediction_routes,
        "load_champion_model",
        raise_missing_champion,
    )

    response = client.post(
        "/predict/from-observations",
        json=valid_observation_request(),
    )

    assert response.status_code == 503
    assert "No Champion model" in response.json()["detail"]
