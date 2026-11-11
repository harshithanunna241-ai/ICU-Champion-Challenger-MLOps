
"""Test calibration and threshold selection for three ML models."""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from src.models.train_logistic import train_logistic_model
from src.models.train_random_forest import train_random_forest_model
from src.models.train_xgboost import train_xgboost_model

from src.models.calibration import (
    calibrate_model,
    select_classification_threshold,
    evaluate_calibration,
)


def main():

    print("\n========== THREE-MODEL CALIBRATION TEST ==========")

    # 1. Create synthetic dataset
    X, y = make_classification(
        n_samples=600,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        weights=[0.65, 0.35],
        random_state=42,
    )

    feature_columns = [
        "heart_rate_mean",
        "systolic_bp_mean",
        "diastolic_bp_mean",
        "mean_bp_mean",
        "respiratory_rate_mean",
        "temperature_mean",
        "spo2_mean",
        "glucose_mean",
    ]

    dataset = pd.DataFrame(X, columns=feature_columns)
    dataset["deterioration_event"] = y

    print("\nSynthetic dataset created.")
    print("Total records:", len(dataset))
    print("Positive records:", int(y.sum()))
    print("Negative records:", int(len(y) - y.sum()))

    # 2. Split dataset
    train_data, validation_data = train_test_split(
        dataset,
        test_size=0.25,
        stratify=dataset["deterioration_event"],
        random_state=42,
    )

    print("\nTraining records:", len(train_data))
    print("Validation records:", len(validation_data))

    # 3. Train three models
    print("\nTraining Logistic Regression...")
    logistic_model, logistic_features = train_logistic_model(train_data)

    print("Training Random Forest...")
    rf_model, rf_features = train_random_forest_model(train_data)

    print("Training XGBoost...")
    xgb_model, xgb_features = train_xgboost_model(train_data)

    models = {
        "Logistic Regression": (logistic_model, logistic_features),
        "Random Forest": (rf_model, rf_features),
        "XGBoost": (xgb_model, xgb_features),
    }

    # 4. Calibrate and evaluate models
    for model_name, (model, features) in models.items():

        print(f"\n========== {model_name} ==========")

        # Calibrate model
        calibrated_model = calibrate_model(
            model=model,
            train_data=train_data,
            feature_columns=features,
            method="sigmoid",
            cv=5,
        )

        print("Model calibration: Completed")

        # Evaluate calibration
        calibration_results = evaluate_calibration(
            model=calibrated_model,
            validation_data=validation_data,
            feature_columns=features,
        )

        print(
            "Brier Score:",
            round(calibration_results["brier_score"], 4),
        )

        print(
            "Log Loss:",
            round(calibration_results["log_loss"], 4),
        )

        # Generate validation probabilities
        validation_probabilities = calibrated_model.predict_proba(
            validation_data[features]
        )[:, 1]

        # Select classification threshold
        threshold_results = select_classification_threshold(
            y_true=validation_data["deterioration_event"],
            y_prob=validation_probabilities,
            objective="f1",
        )

        print("\nThreshold Selection Results:")
        print(threshold_results)

        # Extract threshold from returned dictionary
        if isinstance(threshold_results, dict):
            threshold = threshold_results["threshold"]
        else:
            threshold = threshold_results

        print(
            "Selected Threshold:",
            round(float(threshold), 4),
        )

        # Validation checks
        assert np.isfinite(calibration_results["brier_score"])
        assert np.isfinite(calibration_results["log_loss"])
        assert 0 <= float(threshold) <= 1
        assert np.all(np.isfinite(validation_probabilities))

        print(f"{model_name} Calibration: SUCCESS")

    print("\n==============================================")
    print("THREE-MODEL CALIBRATION TEST: SUCCESS")
    print("==============================================")


if __name__ == "__main__":
    main()
