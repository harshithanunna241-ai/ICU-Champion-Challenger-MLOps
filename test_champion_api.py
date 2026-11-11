
"""Tests for the Champion model API."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def main():
    print("\n========== CHAMPION MODEL API TEST ==========\n")

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
                "auprc": 0.90,
                "brier_score": 0.11,
            },
        },
    ]

    print("Testing Champion retrieval...")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=mock_models,
    ):
        response = client.get("/models/champion")

    assert response.status_code == 200

    result = response.json()
    champion = result["champion"]

    assert champion["model_name"] == "Logistic Regression"
    assert champion["stage"] == "champion"
    assert champion["metrics"]["auprc"] == 0.88

    print("Champion retrieval: SUCCESS")
    print("Champion:", champion["model_name"])

    print("\nTesting empty registry...")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=[],
    ):
        response = client.get("/models/champion")

    assert response.status_code == 404

    print("No Champion handling: SUCCESS")

    print("\nTesting multiple Champions...")

    multiple_champions = [
        mock_models[0],
        {
            **mock_models[1],
            "stage": "champion",
        },
    ]

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=multiple_champions,
    ):
        response = client.get("/models/champion")

    assert response.status_code == 409

    print("Multiple Champion validation: SUCCESS")

    print("\nTesting Swagger registration...")

    schema = client.get("/openapi.json").json()

    assert "/models/champion" in schema["paths"]
    assert "get" in schema["paths"]["/models/champion"]

    print("Swagger registration: SUCCESS")

    print("\n========== CHAMPION MODEL API TEST PASSED ==========")


if __name__ == "__main__":
    main()
