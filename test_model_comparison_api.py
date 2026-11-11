
"""Tests for the Model Comparison API."""

from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.main import app


client = TestClient(app)


def sample_registry():
    return [
        {
            "model_name": "Logistic Regression",
            "version": "1",
            "stage": "champion",
            "metrics": {
                "auroc": 0.88,
                "auprc": 0.80,
                "recall": 0.75,
                "brier_score": 0.20,
            },
        },
        {
            "model_name": "Random Forest",
            "version": "2",
            "stage": "challenger",
            "metrics": {
                "auroc": 0.91,
                "auprc": 0.85,
                "recall": 0.82,
                "brier_score": 0.17,
            },
        },
        {
            "model_name": "XGBoost",
            "version": "1",
            "stage": "challenger",
            "metrics": {
                "auroc": 0.90,
                "auprc": 0.83,
                "recall": 0.80,
                "brier_score": 0.18,
            },
        },
    ]


def main():

    print("\n========== MODEL COMPARISON API TEST ==========\n")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=sample_registry(),
    ):

        print("Testing Champion and Challenger comparison...")

        response = client.get("/models/comparison")

        assert response.status_code == 200

        result = response.json()

        assert result["status"] == "success"
        assert result["champion"]["model_name"] == "Logistic Regression"
        assert result["total_comparisons"] == 2

        print("Model comparison: SUCCESS")

        print("\nTesting Random Forest metric differences...")

        random_forest = next(
            model
            for model in result["comparisons"]
            if model["model_name"] == "Random Forest"
        )

        differences = random_forest["difference_from_champion"]

        assert differences["auroc"] == 0.03
        assert differences["auprc"] == 0.05
        assert differences["recall"] == 0.07
        assert differences["brier_score"] == -0.03

        print("Random Forest metric differences: SUCCESS")

        print("\nTesting XGBoost comparison...")

        xgboost = next(
            model
            for model in result["comparisons"]
            if model["model_name"] == "XGBoost"
        )

        assert xgboost["difference_from_champion"]["auprc"] == 0.03

        print("XGBoost comparison: SUCCESS")

    print("\nTesting empty registry...")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=[],
    ):

        response = client.get("/models/comparison")

        assert response.status_code == 404

    print("No Champion handling: SUCCESS")

    print("\nTesting multiple Champions...")

    duplicate_champions = sample_registry() + [
        {
            "model_name": "Extra Model",
            "version": "3",
            "stage": "champion",
            "metrics": {"auprc": 0.90},
        }
    ]

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=duplicate_champions,
    ):

        response = client.get("/models/comparison")

        assert response.status_code == 409

    print("Multiple Champion validation: SUCCESS")

    print("\nTesting Swagger registration...")

    schema = client.get("/openapi.json").json()

    assert "/models/comparison" in schema["paths"]

    print("Swagger registration: SUCCESS")

    print("\n========== MODEL COMPARISON API TEST PASSED ==========\n")


if __name__ == "__main__":
    main()
