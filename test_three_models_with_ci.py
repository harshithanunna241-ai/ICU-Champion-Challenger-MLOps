
"""Integration test for three-model confidence interval evaluation."""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.models.train_logistic import train_logistic_model
from src.models.train_random_forest import train_random_forest_model
from src.models.train_xgboost import train_xgboost_model

from src.evaluation.evaluate_three_models_with_ci import (
    evaluate_three_models_with_ci,
    create_confidence_interval_table,
)


def main():

    print("\n========== THREE-MODEL CONFIDENCE INTERVAL TEST ==========")

    # --------------------------------------------------
    # 1. Generate synthetic dataset
    # --------------------------------------------------

    rng = np.random.default_rng(42)

    total_records = 300

    data = pd.DataFrame({
        "subject_id": np.arange(1, total_records + 1),
        "stay_id": np.arange(1001, 1001 + total_records),
        "heart_rate_mean": rng.normal(95, 20, total_records),
        "systolic_bp_mean": rng.normal(115, 18, total_records),
        "diastolic_bp_mean": rng.normal(75, 12, total_records),
        "temperature_mean": rng.normal(37, 1, total_records),
        "spo2_mean": rng.normal(96, 3, total_records),
        "respiratory_rate_mean": rng.normal(20, 5, total_records),
    })

    # Generate a synthetic deterioration risk score
    risk_score = (
        (data["heart_rate_mean"] > 110).astype(int)
        + (data["systolic_bp_mean"] < 100).astype(int)
        + (data["spo2_mean"] < 93).astype(int)
        + (data["temperature_mean"] > 38.5).astype(int)
        + (data["respiratory_rate_mean"] > 24).astype(int)
    )

    data["deterioration_event"] = (
        risk_score >= 2
    ).astype(int)

    # Add some missing feature values
    data.loc[
        rng.choice(data.index, 15, replace=False),
        "spo2_mean",
    ] = np.nan

    data.loc[
        rng.choice(data.index, 15, replace=False),
        "temperature_mean",
    ] = np.nan

    print("\nSynthetic dataset created.")
    print("Total records:", len(data))

    print(
        "Positive records:",
        int(data["deterioration_event"].sum()),
    )

    print(
        "Negative records:",
        int((data["deterioration_event"] == 0).sum()),
    )

    # --------------------------------------------------
    # 2. Split dataset
    # --------------------------------------------------

    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["deterioration_event"],
    )

    print("\nTraining records:", len(train_data))
    print("Testing records:", len(test_data))

    # --------------------------------------------------
    # 3. Train Logistic Regression
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic_model, logistic_features = train_logistic_model(
        train_data
    )

    print("Logistic Regression training: SUCCESS")

    # --------------------------------------------------
    # 4. Train Random Forest
    # --------------------------------------------------

    print("\nTraining Random Forest...")

    random_forest_model, random_forest_features = (
        train_random_forest_model(train_data)
    )

    print("Random Forest training: SUCCESS")

    # --------------------------------------------------
    # 5. Train XGBoost
    # --------------------------------------------------

    print("\nTraining XGBoost...")

    xgboost_model, xgboost_features = train_xgboost_model(
        train_data
    )

    print("XGBoost training: SUCCESS")

    # --------------------------------------------------
    # 6. Calculate confidence intervals
    # --------------------------------------------------

    print("\nEvaluating all three models...")

    results = evaluate_three_models_with_ci(
        logistic_model=logistic_model,
        random_forest_model=random_forest_model,
        xgboost_model=xgboost_model,
        test_data=test_data,
        logistic_features=logistic_features,
        random_forest_features=random_forest_features,
        xgboost_features=xgboost_features,
        n_bootstraps=1000,
        confidence_level=0.95,
        threshold=0.5,
    )

    # --------------------------------------------------
    # 7. Create comparison table
    # --------------------------------------------------

    comparison_table = create_confidence_interval_table(
        results
    )

    print("\n========== CONFIDENCE INTERVAL RESULTS ==========")

    print(
        comparison_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------
    # 8. Verify results
    # --------------------------------------------------

    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }

    assert set(results.keys()) == expected_models

    assert len(comparison_table) == 15

    for model_name, model_result in results.items():

        metrics = model_result["confidence_intervals"]["metrics"]

        assert len(metrics) == 5

        for metric_name, values in metrics.items():

            assert 0 <= values["estimate"] <= 1

            assert 0 <= values["lower_bound"] <= 1

            assert 0 <= values["upper_bound"] <= 1

            assert (
                values["lower_bound"]
                <= values["upper_bound"]
            )

            assert (
                values["valid_bootstrap_samples"] >= 100
            )

        print(f"{model_name}: verification SUCCESS")

    print(
        "\nTHREE-MODEL CONFIDENCE INTERVAL TEST: SUCCESS"
    )


if __name__ == "__main__":
    main()
