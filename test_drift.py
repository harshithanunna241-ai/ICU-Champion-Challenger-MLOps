
"""Test feature drift detection using synthetic ICU data."""

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    calculate_psi,
    detect_feature_drift,
    detect_data_drift,
)


def main():
    print("\n========== DATA DRIFT TEST ==========")

    rng = np.random.default_rng(42)
    n = 500

    # Reference data: baseline patient observations
    reference_data = pd.DataFrame({
        "heart_rate_mean": rng.normal(72, 5, n),
        "spo2_mean": rng.normal(97, 1, n),
        "systolic_bp_mean": rng.normal(120, 10, n),
    })

    # Current data: heart rate distribution has shifted
    current_data = pd.DataFrame({
        "heart_rate_mean": rng.normal(95, 5, n),
        "spo2_mean": rng.normal(97, 1, n),
        "systolic_bp_mean": rng.normal(120, 10, n),
    })

    # Introduce missing observations
    current_data.loc[:19, "spo2_mean"] = np.nan

    print("Reference records:", len(reference_data))
    print("Current records:", len(current_data))

    # Test 1: Population Stability Index
    psi = calculate_psi(
        reference_data["heart_rate_mean"],
        current_data["heart_rate_mean"],
    )

    print("\nHeart rate PSI:", round(psi, 4))

    assert np.isfinite(psi), "PSI must be finite."
    assert psi > 0, "Expected a positive PSI for shifted data."

    # Test 2: Individual feature drift
    feature_result = detect_feature_drift(
        reference_data,
        current_data,
        feature="heart_rate_mean",
    )

    print("\nIndividual feature result:")
    print(feature_result)

    assert feature_result["psi"] > 0.20
    assert feature_result["drift_status"] == "DRIFT DETECTED"

    # Test 3: Multiple feature drift detection
    report = detect_data_drift(
        reference_data,
        current_data,
        feature_columns=[
            "heart_rate_mean",
            "spo2_mean",
            "systolic_bp_mean",
        ],
    )

    print("\nComplete drift report:")
    print(report.to_string(index=False))

    assert len(report) == 3
    assert set(report["feature"]) == {
        "heart_rate_mean",
        "spo2_mean",
        "systolic_bp_mean",
    }

    # Test 4: Missing-value monitoring
    spo2_result = report[
        report["feature"] == "spo2_mean"
    ].iloc[0]

    assert spo2_result["current_missing_rate"] > 0

    print("\nAll data drift tests passed successfully.")


if __name__ == "__main__":
    main()
