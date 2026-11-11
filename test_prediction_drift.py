
"""Test prediction drift monitoring."""

import numpy as np
import pandas as pd

from src.monitoring.prediction_drift import monitor_prediction_drift


def main():
    print("\n========== PREDICTION DRIFT TEST ==========")

    rng = np.random.default_rng(42)
    n = 500

    # Reference model predictions
    reference_predictions = pd.DataFrame({
        "predicted_probability": rng.beta(2, 8, n)
    })

    # Current model predictions with a shifted distribution
    current_predictions = pd.DataFrame({
        "predicted_probability": rng.beta(6, 4, n)
    })

    # Introduce missing prediction values
    current_predictions.loc[:9, "predicted_probability"] = np.nan

    print("Reference records:", len(reference_predictions))
    print("Current records:", len(current_predictions))

    # Run prediction drift monitoring
    result = monitor_prediction_drift(
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        probability_column="predicted_probability",
        psi_threshold=0.20,
        decision_threshold=0.50,
    )

    print("\n========== PREDICTION DRIFT REPORT ==========")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Validate results
    assert result["reference_records"] == 500
    assert result["current_records"] == 490

    assert result["psi"] > 0.20
    assert result["status"] == "DRIFT DETECTED"

    assert result["current_missing_rate"] > 0
    assert result["reference_missing_rate"] == 0

    assert 0 <= result["reference_mean_probability"] <= 1
    assert 0 <= result["current_mean_probability"] <= 1

    assert (
        result["current_positive_prediction_rate"]
        > result["reference_positive_prediction_rate"]
    )

    print("\nAll prediction drift tests passed successfully.")


if __name__ == "__main__":
    main()
