
"""Test historical ICU monitoring trend analysis."""

import pandas as pd

from src.monitoring.trends import (
    analyze_monitoring_trends,
    summarize_monitoring_trends,
)


def main():
    print("\n========== MONITORING TRENDS TEST ==========")

    # Create synthetic historical monitoring records
    history = pd.DataFrame([
        {
            "timestamp": "2026-11-01T10:00:00+00:00",
            "overall_status": "ATTENTION REQUIRED",
            "feature_drift": [
                {
                    "feature": "heart_rate_mean",
                    "drift_status": "DRIFT DETECTED",
                    "psi": 0.30,
                },
                {
                    "feature": "spo2_mean",
                    "drift_status": "STABLE",
                    "psi": 0.04,
                },
            ],
            "prediction_drift": {
                "psi": 0.28,
                "status": "DRIFT DETECTED",
            },
            "model_performance": {
                "auroc": 0.80,
                "auprc": 0.78,
                "brier_score": 0.20,
            },
            "alerts": {
                "total_alerts": 3,
                "alerts": [],
            },
        },
        {
            "timestamp": "2026-11-03T10:00:00+00:00",
            "overall_status": "ATTENTION REQUIRED",
            "feature_drift": [
                {
                    "feature": "heart_rate_mean",
                    "drift_status": "DRIFT DETECTED",
                    "psi": 0.35,
                },
                {
                    "feature": "spo2_mean",
                    "drift_status": "DRIFT DETECTED",
                    "psi": 0.22,
                },
            ],
            "prediction_drift": {
                "psi": 0.32,
                "status": "DRIFT DETECTED",
            },
            "model_performance": {
                "auroc": 0.75,
                "auprc": 0.73,
                "brier_score": 0.24,
            },
            "alerts": {
                "total_alerts": 4,
                "alerts": [],
            },
        },
        {
            "timestamp": "2026-11-05T10:00:00+00:00",
            "overall_status": "ALL CLEAR",
            "feature_drift": [
                {
                    "feature": "heart_rate_mean",
                    "drift_status": "STABLE",
                    "psi": 0.05,
                },
                {
                    "feature": "spo2_mean",
                    "drift_status": "STABLE",
                    "psi": 0.03,
                },
            ],
            "prediction_drift": {
                "psi": 0.06,
                "status": "STABLE",
            },
            "model_performance": {
                "auroc": 0.90,
                "auprc": 0.88,
                "brier_score": 0.12,
            },
            "alerts": {
                "total_alerts": 0,
                "alerts": [],
            },
        },
    ])

    print("Historical records:", len(history))

    # Test 1: Analyze monitoring trends
    trends = analyze_monitoring_trends(history)

    print("\n========== HISTORICAL TREND REPORT ==========")
    print(trends.to_string(index=False))

    assert len(trends) == 3
    assert trends["auroc"].tolist() == [0.80, 0.75, 0.90]
    assert trends["total_alerts"].tolist() == [3, 4, 0]

    assert trends.iloc[0]["feature_drift_count"] == 1
    assert trends.iloc[1]["feature_drift_count"] == 2
    assert trends.iloc[2]["feature_drift_count"] == 0

    print("\nTrend analysis validation: PASSED")

    # Test 2: Summarize historical trends
    summary = summarize_monitoring_trends(trends)

    print("\n========== MONITORING TREND SUMMARY ==========")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    assert summary["total_monitoring_runs"] == 3
    assert summary["total_alerts"] == 7
    assert summary["total_feature_drift_events"] == 3

    assert abs(summary["average_auroc"] - (0.80 + 0.75 + 0.90) / 3) < 1e-9

    assert summary["latest_auroc"] == 0.90
    assert summary["latest_auprc"] == 0.88
    assert summary["latest_brier_score"] == 0.12

    print("\nTrend summary validation: PASSED")

    # Test 3: Empty history validation
    try:
        analyze_monitoring_trends(pd.DataFrame())
    except ValueError:
        print("Empty history validation: PASSED")
    else:
        raise AssertionError("Empty history validation failed.")

    print("\nAll monitoring trend tests passed successfully.")


if __name__ == "__main__":
    main()
