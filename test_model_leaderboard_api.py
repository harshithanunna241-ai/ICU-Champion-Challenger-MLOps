
"""Tests for the model performance leaderboard API."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def main():
    print("\n========== MODEL LEADERBOARD API TEST ==========\n")

    mock_models = [
        {
            "model_name": "Logistic Regression",
            "version": "1.0",
            "stage": "champion",
            "metrics": {
                "auroc": 0.85,
                "auprc": 0.80,
                "brier_score": 0.20,
            },
        },
        {
            "model_name": "Random Forest",
            "version": "1.0",
            "stage": "challenger",
            "metrics": {
                "auroc": 0.92,
                "auprc": 0.90,
                "brier_score": 0.12,
            },
        },
        {
            "model_name": "XGBoost",
            "version": "1.0",
            "stage": "challenger",
            "metrics": {
                "auroc": 0.89,
                "auprc": 0.86,
                "brier_score": 0.15,
            },
        },
    ]

    print("Testing leaderboard endpoint...")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=mock_models,
    ):
        response = client.get("/models/leaderboard")

    assert response.status_code == 200

    result = response.json()

    assert result["total_models"] == 3
    assert result["ranking_metric"] == "AUPRC"

    leaderboard = result["leaderboard"]

    assert leaderboard[0]["model_name"] == "Random Forest"
    assert leaderboard[0]["auprc"] == 0.90
    assert leaderboard[0]["rank"] == 1

    assert leaderboard[1]["model_name"] == "XGBoost"
    assert leaderboard[1]["rank"] == 2

    assert leaderboard[2]["model_name"] == "Logistic Regression"
    assert leaderboard[2]["rank"] == 3

    print("Leaderboard generation: SUCCESS")
    print("AUPRC ranking: SUCCESS")

    print("\nTesting empty registry...")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=[],
    ):
        response = client.get("/models/leaderboard")

    assert response.status_code == 200
    assert response.json()["total_models"] == 0

    print("Empty registry handling: SUCCESS")

    print("\nTesting invalid metric handling...")

    invalid_models = [
        {
            "model_name": "Invalid Model",
            "version": "1.0",
            "stage": "challenger",
            "metrics": {"auprc": 1.5},
        },
        {
            "model_name": "Missing Metric Model",
            "version": "1.0",
            "stage": "challenger",
            "metrics": {},
        },
    ]

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=invalid_models,
    ):
        response = client.get("/models/leaderboard")

    assert response.status_code == 200
    assert response.json()["total_models"] == 0

    print("Invalid metric handling: SUCCESS")

    print("\nTesting Swagger registration...")

    schema = client.get("/openapi.json").json()

    assert "/models/leaderboard" in schema["paths"]
    assert "get" in schema["paths"]["/models/leaderboard"]

    print("Swagger registration: SUCCESS")

    print("\n========== MODEL LEADERBOARD API TEST PASSED ==========")


if __name__ == "__main__":
    main()
