
"""Integration test for three-model calibrated evaluation."""

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.evaluation.evaluate_three_models_calibrated import (
    evaluate_three_models_calibrated,
    create_calibrated_evaluation_report,
)


def main():

    print("\n========== THREE-MODEL CALIBRATED EVALUATION TEST ==========")

    # 1. Generate synthetic dataset
    X, y = make_classification(
        n_samples=900,
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

    # 2. Split into training, threshold-selection and evaluation data
    train_data, remaining_data = train_test_split(
        dataset,
        test_size=0.40,
        stratify=dataset["deterioration_event"],
        random_state=42,
    )

    threshold_data, evaluation_data = train_test_split(
        remaining_data,
        test_size=0.50,
        stratify=remaining_data["deterioration_event"],
        random_state=42,
    )

    print("\nTraining records:", len(train_data))
    print("Threshold selection records:", len(threshold_data))
    print("Evaluation records:", len(evaluation_data))

    # 3. Run integrated three-model evaluation
    comparison = evaluate_three_models_calibrated(
        train_data=train_data,
        threshold_data=threshold_data,
        evaluation_data=evaluation_data,
        target_column="deterioration_event",
        calibration_method="sigmoid",
        cv=5,
    )

    # 4. Display model comparison
    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    # 5. Create evaluation report
    report = create_calibrated_evaluation_report(comparison)

    print("\n========== CALIBRATED EVALUATION REPORT ==========")
    print(report.to_string(index=False))

    # 6. Validate results
    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }

    assert set(comparison["model"]) == expected_models
    assert len(comparison) == 3

    required_metrics = [
        "threshold",
        "brier_score",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    ]

    for metric in required_metrics:
        assert np.isfinite(comparison[metric]).all(), (
            f"Invalid values found in {metric}"
        )

    for metric in [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "brier_score",
    ]:
        assert comparison[metric].between(0, 1).all()

    assert (comparison["log_loss"] >= 0).all()

    assert len(report) == 3
    assert report["f1_score"].is_monotonic_decreasing

    print("\n==============================================")
    print("THREE-MODEL CALIBRATED EVALUATION: SUCCESS")
    print("==============================================")


if __name__ == "__main__":
    main()
