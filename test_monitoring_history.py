
"""Test historical monitoring report storage and retrieval."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.monitoring.history import (
    save_monitoring_report,
    load_monitoring_history,
)


def create_sample_report():
    """Create a synthetic monitoring report."""

    return {
        "feature_drift": pd.DataFrame([
            {
                "feature": "heart_rate_mean",
                "psi": 0.35,
                "drift_status": "DRIFT DETECTED",
            },
            {
                "feature": "spo2_mean",
                "psi": 0.04,
                "drift_status": "STABLE",
            },
        ]),
        "prediction_drift": {
            "psi": 0.32,
            "status": "DRIFT DETECTED",
        },
        "model_performance": {
            "auroc": 0.90,
            "auprc": 0.85,
            "brier_score": 0.10,
        },
        "alerts": {
            "total_alerts": 2,
            "status": "ALERTS GENERATED",
            "alerts": [
                {
                    "category": "FEATURE DRIFT",
                    "severity": "WARNING",
                    "message": "Drift detected in heart_rate_mean.",
                },
                {
                    "category": "PREDICTION DRIFT",
                    "severity": "WARNING",
                    "message": "Prediction distribution has drifted.",
                },
            ],
        },
        "overall_status": "ATTENTION REQUIRED",
    }


def main():
    print("\n========== MONITORING HISTORY TEST ==========")

    report = create_sample_report()

    with tempfile.TemporaryDirectory() as temp_dir:
        history_path = Path(temp_dir) / "monitoring_history.jsonl"

        # Test 1: Save monitoring report
        save_result = save_monitoring_report(
            report=report,
            output_path=history_path,
        )

        print("\nSave result:")
        print(save_result)

        assert save_result["saved"] is True
        assert history_path.exists()
        assert save_result["overall_status"] == "ATTENTION REQUIRED"

        # Test 2: Verify JSON Lines format
        with history_path.open("r", encoding="utf-8") as file:
            first_line = file.readline()

        stored_record = json.loads(first_line)

        assert "timestamp" in stored_record
        assert stored_record["overall_status"] == "ATTENTION REQUIRED"
        assert len(stored_record["feature_drift"]) == 2

        print("\nJSON storage validation: PASSED")

        # Test 3: Load monitoring history
        history = load_monitoring_history(history_path)

        print("\nLoaded monitoring history:")
        print(history.to_string(index=False))

        assert len(history) == 1
        assert history.iloc[0]["overall_status"] == "ATTENTION REQUIRED"

        print("\nHistory retrieval validation: PASSED")

        # Test 4: Save another report
        second_report = create_sample_report()
        second_report["overall_status"] = "ALL CLEAR"
        second_report["alerts"] = {
            "total_alerts": 0,
            "status": "ALL CLEAR",
            "alerts": [],
        }

        save_monitoring_report(
            report=second_report,
            output_path=history_path,
        )

        updated_history = load_monitoring_history(history_path)

        assert len(updated_history) == 2
        assert updated_history.iloc[1]["overall_status"] == "ALL CLEAR"

        print("\nMultiple report storage validation: PASSED")

        # Test 5: Invalid report validation
        try:
            save_monitoring_report(
                report={"overall_status": "ALL CLEAR"},
                output_path=history_path,
            )
        except ValueError:
            print("Invalid report validation: PASSED")
        else:
            raise AssertionError("Invalid report test failed.")

    print("\nAll monitoring history tests passed successfully.")


if __name__ == "__main__":
    main()
