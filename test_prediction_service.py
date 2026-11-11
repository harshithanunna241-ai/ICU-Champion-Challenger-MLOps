
"""Tests for the ICU prediction service."""

import tempfile

import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.registry.model_registry import register_model
from src.api.prediction_service import (
    load_champion_model,
    predict_deterioration,
)


def main():
    print("\n========== PREDICTION SERVICE TEST ==========\n")

    feature_columns = [
        "heart_rate_mean",
        "spo2_mean",
    ]

    X = np.array([
        [70, 99],
        [75, 98],
        [110, 91],
        [120, 88],
        [80, 97],
        [115, 90],
        [65, 99],
        [130, 85],
    ])

    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(random_state=42)),
    ])

    model.fit(X, y)

    with tempfile.TemporaryDirectory() as temp_dir:

        print("Registering test Champion model...")

        register_model(
            model=model,
            model_name="Test Logistic Regression",
            version="1.0",
            metrics={
                "auroc": 0.90,
                "auprc": 0.88,
                "brier_score": 0.12,
            },
            feature_columns=feature_columns,
            stage="champion",
            registry_dir=temp_dir,
        )

        print("Champion registration: SUCCESS")

        print("\nTesting Champion model loading...")

        loaded_model, model_name, columns = load_champion_model(
            registry_dir=temp_dir
        )

        assert loaded_model is not None
        assert model_name == "Test Logistic Regression"
        assert columns == feature_columns

        print("Champion model loading: SUCCESS")

        print("\nTesting deterioration prediction...")

        result = predict_deterioration(
            features={
                "heart_rate_mean": 115,
                "spo2_mean": 90,
            },
            registry_dir=temp_dir,
        )

        assert result["prediction"] in [0, 1]
        assert 0 <= result["deterioration_probability"] <= 1
        assert result["model_name"] == "Test Logistic Regression"
        assert result["status"] == "success"

        print("Prediction generation: SUCCESS")
        print("Prediction:", result["prediction"])
        print(
            "Probability:",
            round(result["deterioration_probability"], 4)
        )

        print("\nTesting missing feature validation...")

        try:
            predict_deterioration(
                features={"heart_rate_mean": 100},
                registry_dir=temp_dir,
            )

            raise AssertionError(
                "Missing feature was not rejected."
            )

        except ValueError as exc:
            assert "Missing required features" in str(exc)

        print("Missing feature validation: SUCCESS")

    print("\n========== PREDICTION SERVICE TEST PASSED ==========")


if __name__ == "__main__":
    main()
