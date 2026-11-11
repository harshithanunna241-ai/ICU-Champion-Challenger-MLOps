
"""Tests for the Model Version API."""

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
                "auprc": 0.80,
                "auroc": 0.88,
                "recall": 0.75,
            },
        },
        {
            "model_name": "Random Forest",
            "version": "2",
            "stage": "challenger",
            "metrics": {
                "auprc": 0.85,
                "auroc": 0.91,
                "recall": 0.82,
            },
        },
    ]


def main():

    print("\n========== MODEL VERSION API TEST ==========\n")

    with patch(
        "src.api.model_routes.list_registered_models",
        return_value=sample_registry(),
    ):

        print("Testing existing model version...")

        response = client.get("/models/Random Forest/2")

        assert response.status_code == 200

        result = response.json()

        assert result["status"] == "success"
        assert result["model"]["model_name"] == "Random Forest"
        assert result["model"]["version"] == "2"

        print("Model version retrieval: SUCCESS")
        print("Model:", result["model"]["model_name"])
        print("Version:", result["model"]["version"])

        print("\nTesting missing model version...")

        response = client.get("/models/Random Forest/99")

        assert response.status_code == 404

        print("Missing model handling: SUCCESS")

        print("\nTesting model name case-insensitivity...")

        response = client.get("/models/random forest/2")

        assert response.status_code == 200

        print("Case-insensitive retrieval: SUCCESS")

        print("\nTesting multiple matching versions...")

        duplicate_records = sample_registry() + [
            {
                "model_name": "Random Forest",
                "version": "2",
                "stage": "archived",
                "metrics": {"auprc": 0.70},
            }
        ]

        with patch(
            "src.api.model_routes.list_registered_models",
            return_value=duplicate_records,
        ):

            response = client.get("/models/Random Forest/2")

            assert response.status_code == 409

        print("Duplicate version validation: SUCCESS")

    print("\nTesting Swagger registration...")

    schema = client.get("/openapi.json").json()

    assert "/models/{model_name}/{version}" in schema["paths"]

    print("Swagger registration: SUCCESS")

    print("\n========== MODEL VERSION API TEST PASSED ==========\n")


if __name__ == "__main__":
    main()
