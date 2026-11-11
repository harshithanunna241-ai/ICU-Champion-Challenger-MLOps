
"""Test XGBoost training and evaluation with synthetic ICU data."""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.models.train_xgboost import (
    train_xgboost_model,
    evaluate_xgboost_model,
)


def main():
    print("\n========== XGBOOST MODEL INTEGRATION TEST ==========")

    np.random.seed(42)

    total_records = 200

    heart_rate = np.random.normal(100, 15, total_records)
    systolic_bp = np.random.normal(115, 18, total_records)
    spo2 = np.random.normal(96, 3, total_records)

    risk_score = (
        (heart_rate > 110).astype(int)
        + (systolic_bp < 100).astype(int)
        + (spo2 < 93).astype(int)
    )

    deterioration_event = (risk_score >= 2).astype(int)

    dataset = pd.DataFrame({
        "subject_id": np.arange(1000, 1000 + total_records),
        "stay_id": np.arange(2000, 2000 + total_records),
        "heart_rate_mean": heart_rate,
        "systolic_bp_mean": systolic_bp,
        "spo2_mean": spo2,
        "temperature_mean": np.random.normal(37, 0.8, total_records),
        "deterioration_event": deterioration_event,
    })

    # Introduce some missing feature values for imputation testing.
    dataset.loc[dataset.index[::15], "spo2_mean"] = np.nan

    train_data, test_data = train_test_split(
        dataset,
        test_size=0.20,
        random_state=42,
        stratify=dataset["deterioration_event"],
    )

    print(f"Total records: {len(dataset)}")
    print(f"Training records: {len(train_data)}")
    print(f"Testing records: {len(test_data)}")

    print("\nTraining XGBoost model...")

    model, feature_columns = train_xgboost_model(train_data)

    print("\nEvaluating XGBoost model...")

    metrics = evaluate_xgboost_model(
        model=model,
        test_data=test_data,
        feature_columns=feature_columns,
    )

    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    ]

    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert np.isfinite(metrics[metric]), f"Invalid metric: {metric}"

    print("\nXGBOOST MODEL INTEGRATION TEST: SUCCESS")
    print("Model training, missing-value handling and evaluation passed.")


if __name__ == "__main__":
    main()
