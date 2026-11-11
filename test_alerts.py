
"""Test monitoring alert rules for the ICU prediction platform."""

import pandas as pd

from src.monitoring.alerts import generate_monitoring_alerts


def main():
    print("\n========== MONITORING ALERT TEST ==========")

    # Synthetic feature drift report
    feature_drift_report = pd.DataFrame([
        {
            "feature": "heart_rate_mean",
            "drift_status": "DRIFT DETECTED",
            "psi": 0.35,
        },
        {
            "feature": "spo2_mean",
            "drift_status": "STABLE",
            "psi": 0.04,
        },
        {
            "feature": "systolic_bp_mean",
            "drift_status": "MODERATE DRIFT",
            "psi": 0.15,
        },
    ])

    # Synthetic prediction drift report
    prediction_drift_report = {
        "psi": 0.32,
        "status": "DRIFT DETECTED",
    }

    # Synthetic model performance report
    performance_report = {
        "auroc": 0.68,
        "auprc": 0.65,
        "brier_score": 0.30,
    }

    print("\nGenerating monitoring alerts...")

    result = generate_monitoring_alerts(
        feature_drift_report=feature_drift_report,
        prediction_drift_report=prediction_drift_report,
        performance_report=performance_report,
        minimum_auroc=0.75,
        minimum_auprc=0.75,
        maximum_brier_score=0.25,
    )

    print("\n========== ALERT REPORT ==========")
    print("Total alerts:", result["total_alerts"])
    print("Overall status:", result["status"])

    for index, alert in enumerate(result["alerts"], start=1):
        print(f"\nAlert {index}")
        print("Category:", alert["category"])
        print("Severity:", alert["severity"])
        print("Message:", alert["message"])

    # Expected alerts:
    # 1 feature drift
    # 1 prediction drift
    # 1 AUROC failure
    # 1 AUPRC failure
    # 1 Brier score failure

    assert result["total_alerts"] == 5
    assert result["status"] == "ALERTS GENERATED"

    categories = [alert["category"] for alert in result["alerts"]]

    assert "FEATURE DRIFT" in categories
    assert "PREDICTION DRIFT" in categories
    assert "MODEL PERFORMANCE" in categories
    assert "MODEL CALIBRATION" in categories

    # Test all-clear condition
    stable_features = pd.DataFrame([
        {
            "feature": "heart_rate_mean",
            "drift_status": "STABLE",
        }
    ])

    stable_predictions = {
        "status": "STABLE",
    }

    good_performance = {
        "auroc": 0.90,
        "auprc": 0.85,
        "brier_score": 0.10,
    }

    clear_result = generate_monitoring_alerts(
        feature_drift_report=stable_features,
        prediction_drift_report=stable_predictions,
        performance_report=good_performance,
    )

    assert clear_result["total_alerts"] == 0
    assert clear_result["status"] == "ALL CLEAR"

    print("\nAll-clear condition: PASSED")

    # Test invalid threshold
    try:
        generate_monitoring_alerts(
            minimum_auroc=1.5
        )
    except ValueError:
        print("Invalid threshold validation: PASSED")
    else:
        raise AssertionError("Invalid threshold test failed.")

    print("\nAll monitoring alert tests passed successfully.")


if __name__ == "__main__":
    main()
