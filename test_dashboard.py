
"""Test centralized ICU monitoring dashboard integration."""

import numpy as np
import pandas as pd

from src.monitoring.dashboard import (
    build_monitoring_report,
    print_monitoring_report,
)


def main():
    print("\n========== ICU DASHBOARD INTEGRATION TEST ==========")

    rng = np.random.default_rng(42)
    n = 500

    # 1. Reference patient feature data
    reference_data = pd.DataFrame({
        "heart_rate_mean": rng.normal(72, 5, n),
        "spo2_mean": rng.normal(97, 1, n),
        "systolic_bp_mean": rng.normal(120, 10, n),
    })

    # 2. Current patient feature data with heart rate drift
    current_data = pd.DataFrame({
        "heart_rate_mean": rng.normal(95, 5, n),
        "spo2_mean": rng.normal(97, 1, n),
        "systolic_bp_mean": rng.normal(120, 10, n),
    })

    # 3. Reference model predictions
    reference_predictions = pd.DataFrame({
        "predicted_probability": rng.beta(2, 8, n)
    })

    # 4. Current model predictions with prediction drift
    current_predictions = pd.DataFrame({
        "predicted_probability": rng.beta(6, 4, n)
    })

    # 5. Evaluation data with actual outcomes
    y_true = np.array([0, 1] * (n // 2))
    rng.shuffle(y_true)

    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.60, 0.95, n),
        rng.uniform(0.05, 0.40, n),
    )

    evaluation_data = pd.DataFrame({
        "deterioration_event": y_true,
        "predicted_probability": y_prob,
    })

    print("\nSynthetic monitoring datasets created.")

    # 6. Build centralized monitoring report
    report = build_monitoring_report(
        reference_data=reference_data,
        current_data=current_data,
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        evaluation_data=evaluation_data,
        feature_columns=[
            "heart_rate_mean",
            "spo2_mean",
            "systolic_bp_mean",
        ],
        probability_column="predicted_probability",
        target_column="deterioration_event",
        decision_threshold=0.50,
        psi_threshold=0.20,
        minimum_auroc=0.75,
        minimum_auprc=0.75,
        maximum_brier_score=0.25,
    )

    # 7. Display dashboard report
    print_monitoring_report(report)

    # 8. Validate report structure
    expected_sections = {
        "feature_drift",
        "prediction_drift",
        "model_performance",
        "alerts",
        "overall_status",
    }

    assert set(report.keys()) == expected_sections

    assert len(report["feature_drift"]) == 3

    assert report["prediction_drift"]["status"] == "DRIFT DETECTED"

    assert report["model_performance"]["total_records"] == 500

    assert report["alerts"]["total_alerts"] >= 1

    assert report["overall_status"] == "ATTENTION REQUIRED"

    # 9. Validate monitoring alert integration
    alert_categories = [
        alert["category"]
        for alert in report["alerts"]["alerts"]
    ]

    assert "FEATURE DRIFT" in alert_categories
    assert "PREDICTION DRIFT" in alert_categories

    print("\n========== DASHBOARD TEST RESULTS ==========")
    print("Feature drift integration: PASSED")
    print("Prediction drift integration: PASSED")
    print("Model performance integration: PASSED")
    print("Alert generation integration: PASSED")
    print("Overall dashboard integration: PASSED")

    print("\nAll dashboard integration tests passed successfully.")


if __name__ == "__main__":
    main()
