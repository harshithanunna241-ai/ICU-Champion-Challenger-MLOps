
"""Test integration of existing ML models with MLflow."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.orchestration.train_and_track import (
    train_and_track_models,
)


def main():

    print("\n========== MLFLOW MODEL INTEGRATION TEST ==========")

    # Step 1: Generate reproducible synthetic data.
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

    deterioration_event = (risk_score >= 2).astype(int)

    dataset = pd.DataFrame({
        "subject_id": range(1001, 1001 + total_records),
        "heart_rate_mean": heart_rate,
        "systolic_bp_mean": systolic_bp,
        "diastolic_bp_mean": diastolic_bp,
        "temperature_mean": temperature,
        "spo2_mean": spo2,
        "respiratory_rate_mean": respiratory_rate,
        "deterioration_event": deterioration_event,
    })

    print("\nSynthetic dataset created.")
    print("Total records:", len(dataset))
    print(
        "Positive records:",
        int(dataset["deterioration_event"].sum()),
    )
    print(
        "Negative records:",
        int((dataset["deterioration_event"] == 0).sum()),
    )

    # Step 2: Split data.
    train_data, test_data = train_test_split(
        dataset,
        test_size=0.20,
        random_state=42,
        stratify=dataset["deterioration_event"],
    )

    print("\nTraining records:", len(train_data))
    print("Testing records:", len(test_data))

    # Step 3: Configure persistent SQLite tracking.
    project_root = Path(__file__).resolve().parent

    tracking_dir = (
        project_root / "artifacts" / "mlflow_test"
    )

    tracking_dir.mkdir(parents=True, exist_ok=True)

    database_path = tracking_dir / "mlflow_test.db"

    tracking_uri = (
        f"sqlite:///{database_path.resolve().as_posix()}"
    )

    print("\nTracking database:", database_path)

    # Step 4: Train, evaluate and track both models.
    results = train_and_track_models(
        train_data=train_data,
        test_data=test_data,
        experiment_name="ICU Model Integration Test",
        tracking_uri=tracking_uri,
    )

    # Step 5: Final test report.
    print("\n========== INTEGRATION TEST REPORT ==========")

    for model_name, result in results.items():

        print(f"\nModel: {model_name}")

        print(
            "Run ID:",
            result["tracking"]["run_id"],
        )

        print(
            "Metrics recorded:",
            len(result["metrics"]),
        )

    print("\nMLflow model integration completed successfully.")

    print("================================================")


if __name__ == "__main__":
    main()
