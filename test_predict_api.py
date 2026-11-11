
"""Integration tests for the ICU prediction API."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def main():
    print("\n========== PREDICTION API TEST ==========\n")

    print("Testing prediction endpoint...")

    mock_result = {
        "prediction": 1,
        "deterioration_probability": 0.82,
        "model_name": "Test Logistic Regression",
        "status": "success",
    }

    with patch(
        "src.api.prediction_routes.predict_deterioration",
        return_value=mock_result,
    ):

        response = client.post(
            "/predict",
            json={
                "features": {
                    "heart_rate_mean": 115.0,
                    "spo2_mean": 90.0,
                }
            },
        )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 1
    assert result["deterioration_probability"] == 0.82
    assert result["model_name"] == "Test Logistic Regression"

    print("Prediction endpoint: SUCCESS")
    print("HTTP status:", response.status_code)

    print("\nTesting invalid request...")

    response = client.post(
        "/predict",
        json={
            "features": {}
        },
    )

    assert response.status_code == 422

    print("Invalid request validation: SUCCESS")

    print("\nTesting missing feature error...")

    with patch(
        "src.api.prediction_routes.predict_deterioration",
        side_effect=ValueError("Missing required features"),
    ):

        response = client.post(
            "/predict",
            json={
                "features": {
                    "heart_rate_mean": 100.0
                }
            },
        )

    assert response.status_code == 400

    print("Missing feature error handling: SUCCESS")

    print("\nTesting unavailable model error...")

    with patch(
        "src.api.prediction_routes.predict_deterioration",
        side_effect=RuntimeError("No Champion model available"),
    ):

        response = client.post(
            "/predict",
            json={
                "features": {
                    "heart_rate_mean": 100.0
                }
            },
        )

    assert response.status_code == 503

    print("Unavailable model error handling: SUCCESS")

    print("\nTesting Swagger registration...")

    schema = client.get("/openapi.json").json()

    assert "/predict" in schema["paths"]
    assert "post" in schema["paths"]["/predict"]

    print("Swagger prediction route registration: SUCCESS")

    print("\n========== PREDICTION API TEST PASSED ==========")


if __name__ == "__main__":
    main()
