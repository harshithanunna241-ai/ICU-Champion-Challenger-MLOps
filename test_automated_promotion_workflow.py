
"""Test the complete automated promotion and audit workflow."""

import tempfile

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.orchestration.automated_promotion import (
    run_automated_promotion,
)


def main():

    print("\n========== AUTOMATED PROMOTION WORKFLOW TEST ==========")

    rng = np.random.default_rng(42)

    total_records = 100

    heart_rate = rng.normal(95, 15, total_records)
    systolic_bp = rng.normal(120, 18, total_records)
    diastolic_bp = rng.normal(75, 10, total_records)
    temperature = rng.normal(37, 0.7, total_records)
    spo2 = rng.normal(96, 2, total_records)
    respiratory_rate = rng.normal(20, 4, total_records)

    risk_score = (
        (heart_rate > 100).astype(int)
        + (systolic_bp < 110).astype(int)
        + (spo2 < 94).astype(int)
        + (temperature > 38).astype(int)
        + (respiratory_rate > 24).astype(int)
    )

    target = (risk_score >= 2).astype(int)

    features = [
        "heart_rate_mean",
        "systolic_bp_mean",
        "diastolic_bp_mean",
        "temperature_mean",
        "spo2_mean",
        "respiratory_rate_mean",
    ]

    dataset = pd.DataFrame({
        "heart_rate_mean": heart_rate,
        "systolic_bp_mean": systolic_bp,
        "diastolic_bp_mean": diastolic_bp,
        "temperature_mean": temperature,
        "spo2_mean": spo2,
        "respiratory_rate_mean": respiratory_rate,
        "deterioration_event": target,
    })

    # Separate training and held-out testing records.
    from sklearn.model_selection import train_test_split

    train_data, test_data = train_test_split(
        dataset,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    X_train = train_data[features]
    y_train = train_data["deterioration_event"]

    # Train Champion.
    champion_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    champion_model.fit(X_train, y_train)

    print("\nChampion training: SUCCESS")

    # Train Challenger.
    challenger_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )

    challenger_model.fit(X_train, y_train)

    print("Challenger training: SUCCESS")

    print("\nHeld-out testing records:", len(test_data))

    # Execute complete workflow.
    with tempfile.TemporaryDirectory() as temp_dir:

        results = run_automated_promotion(
            champion_model=champion_model,
            challenger_model=challenger_model,
            test_data=test_data,
            champion_features=features,
            challenger_features=features,
            audit_dir=temp_dir,
        )

        if "comparison" not in results:
            raise AssertionError("Model comparison is missing.")

        if "promotion_result" not in results:
            raise AssertionError("Promotion result is missing.")

        if "audit_result" not in results:
            raise AssertionError("Audit result is missing.")

        if not results["audit_result"]["audit_file"]:
            raise AssertionError("Audit file path is missing.")

    print("\n========== FINAL TEST REPORT ==========")

    print("Model comparison: SUCCESS")
    print("Promotion policy: SUCCESS")
    print("Audit logging: SUCCESS")
    print("Complete automated workflow: SUCCESS")

    print("\nAll workflow tests passed successfully.")
    print("======================================================")


if __name__ == "__main__":
    main()
