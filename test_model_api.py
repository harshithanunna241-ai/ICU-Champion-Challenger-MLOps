
"""Tests for the model registry API."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def main():
    print("\n========== MODEL REGISTRY API TEST ==========\n")

    mock_models = [
        {
            "model_name": "Logistic Regression",
            "version": "1.0",
            "stage": "champion",
            "feature_columns": ["heart_rate_mean", "spo2_mean"],
            "metrics": {
                "auroc": 0.90,
                "auprc": 0.88,
                "brier_score": 0.12,
            },
        },
        {
            "model_name": "Random Forest",
            "version": "1.0",
            "stage": "challenger",
            "feature_columns": ["heart_rate_mean", "spo2_mean"],
            "metrics": {
                "auroc": 0.92,
                "auprc": 0.89,
                "brier_score": 0.11,
            },
        },
    ]

    print("Testing registered models endpoint...")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=mock_models,
    ):
        response = client.get("/models/")

    assert response.status_code == 200

    result = response.json()

    assert result["total_models"] == 2
    assert len(result["models"]) == 2
    assert result["models"][0]["model_name"] == "Logistic Regression"
    assert result["models"][0]["stage"] == "champion"

    print("Registered models retrieval: SUCCESS")
    print("Total models:", result["total_models"])

    print("\nTesting empty registry...")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=[],
    ):
        response = client.get("/models/")

    assert response.status_code == 200
    assert response.json()["total_models"] == 0

    print("Empty registry handling: SUCCESS")

    print("\nTesting registry error handling...")

    with patch(
        "src.api.model_routes.list_registered_models",
        side_effect=RuntimeError("Registry unavailable"),
    ):
        response = client.get("/models/")

    assert response.status_code == 500

    print("Registry error handling: SUCCESS")

    print("\nTesting Swagger registration...")

    schema = client.get("/openapi.json").json()

    assert "/models/" in schema["paths"]
    assert "get" in schema["paths"]["/models/"]

    print("Swagger route registration: SUCCESS")

    print("\n========== MODEL REGISTRY API TEST PASSED ==========")


if __name__ == "__main__":
    main()
