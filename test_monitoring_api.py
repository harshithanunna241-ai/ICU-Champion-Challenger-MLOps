
"""Integration tests for ICU monitoring API endpoints."""

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.monitoring_routes as monitoring_routes


def create_test_history(history_path):
    """Create monitoring history matching trends.py schema."""

    records = [
        {
            "timestamp": "2026-11-05T10:00:00+00:00",
            "overall_status": "HEALTHY",

            "model_performance": {
                "auroc": 0.90,
                "auprc": 0.85,
                "brier_score": 0.15,
            },

            "feature_drift": [],

            "prediction_drift": {
                "psi": 0.08,
                "status": "STABLE",
            },

            "alerts": {
                "total_alerts": 0,
                "items": [],
            },
        },

        {
            "timestamp": "2026-11-06T10:00:00+00:00",
            "overall_status": "ATTENTION REQUIRED",

            "model_performance": {
                "auroc": 0.88,
                "auprc": 0.82,
                "brier_score": 0.17,
            },

            "feature_drift": [
                {
                    "feature": "heart_rate_mean",
                    "drift_status": "DRIFT DETECTED",
                }
            ],

            "prediction_drift": {
                "psi": 0.24,
                "status": "DRIFT DETECTED",
            },

            "alerts": {
                "total_alerts": 1,
                "items": [
                    {
                        "severity": "WARNING",
                        "message": "Feature drift detected",
                    }
                ],
            },
        },
    ]

    # Write proper JSON Lines records.
    with open(history_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    return records


def main():
    print("\n========== MONITORING API INTEGRATION TEST ==========\n")

    with tempfile.TemporaryDirectory() as temp_dir:

        history_path = (
            Path(temp_dir) / "monitoring_history.jsonl"
        )

        create_test_history(history_path)

        original_history_path = (
            monitoring_routes.DEFAULT_HISTORY_PATH
        )

        monitoring_routes.DEFAULT_HISTORY_PATH = history_path

        try:
            app = FastAPI()
            app.include_router(monitoring_routes.router)

            client = TestClient(app)

            # -----------------------------------------
            # Test 1: Health endpoint
            # -----------------------------------------

            print("Testing monitoring health endpoint...")

            response = client.get("/monitoring/health")

            assert response.status_code == 200, response.text

            result = response.json()

            assert result["status"] == "healthy"
            assert result["service"] == "ICU Monitoring API"

            print("Health endpoint: SUCCESS")

            # -----------------------------------------
            # Test 2: History endpoint
            # -----------------------------------------

            print("\nTesting monitoring history endpoint...")

            response = client.get("/monitoring/history")

            assert response.status_code == 200, response.text

            result = response.json()

            assert result["total_records"] == 2
            assert len(result["records"]) == 2

            print("History endpoint: SUCCESS")

            # -----------------------------------------
            # Test 3: History limit
            # -----------------------------------------

            print("\nTesting history limit...")

            response = client.get(
                "/monitoring/history?limit=1"
            )

            assert response.status_code == 200, response.text

            result = response.json()

            assert result["total_records"] == 1

            print("History limit: SUCCESS")

            # -----------------------------------------
            # Test 4: Invalid history limit
            # -----------------------------------------

            print("\nTesting invalid history limit...")

            response = client.get(
                "/monitoring/history?limit=0"
            )

            assert response.status_code == 422

            print("Invalid history limit: SUCCESS")

            # -----------------------------------------
            # Test 5: Monitoring trends
            # -----------------------------------------

            print("\nTesting monitoring trends endpoint...")

            response = client.get("/monitoring/trends")

            if response.status_code != 200:
                print("Trends API error:", response.text)

            assert response.status_code == 200, response.text

            result = response.json()

            assert result["total_records"] == 2
            assert len(result["trends"]) == 2

            summary = result["summary"]

            assert summary["total_monitoring_runs"] == 2
            assert summary["total_alerts"] == 1
            assert summary["total_feature_drift_events"] == 1

            print("Trends endpoint: SUCCESS")
            print("Monitoring runs:", summary["total_monitoring_runs"])
            print("Total alerts:", summary["total_alerts"])
            print(
                "Feature drift events:",
                summary["total_feature_drift_events"],
            )

            # -----------------------------------------
            # Test 6: Empty history
            # -----------------------------------------

            print("\nTesting empty monitoring history...")

            empty_history_path = (
                Path(temp_dir) / "empty_history.jsonl"
            )

            empty_history_path.write_text(
                "",
                encoding="utf-8",
            )

            monitoring_routes.DEFAULT_HISTORY_PATH = (
                empty_history_path
            )

            response = client.get("/monitoring/history")

            assert response.status_code == 200, response.text

            result = response.json()

            assert result["total_records"] == 0
            assert result["records"] == []

            print("Empty history handling: SUCCESS")

        finally:
            monitoring_routes.DEFAULT_HISTORY_PATH = (
                original_history_path
            )

    print(
        "\n========== MONITORING API TEST PASSED =========="
    )


if __name__ == "__main__":
    main()
