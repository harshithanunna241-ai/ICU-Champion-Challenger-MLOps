
"""Tests for the Model Registry Audit API."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.audit_routes import load_audit_events


client = TestClient(app)


def sample_audit_events():
    return [
        {
            "model_name": "Logistic Regression",
            "event_type": "registration",
            "version": "1",
            "timestamp": "2026-11-07T10:00:00",
        },
        {
            "model_name": "Random Forest",
            "event_type": "registration",
            "version": "2",
            "timestamp": "2026-11-07T10:30:00",
        },
        {
            "model_name": "Random Forest",
            "event_type": "promotion",
            "version": "2",
            "timestamp": "2026-11-07T11:00:00",
        },
        {
            "model_name": "Logistic Regression",
            "event_type": "archival",
            "version": "1",
            "timestamp": "2026-11-07T11:30:00",
        },
    ]


def main():

    print("\n========== MODEL REGISTRY AUDIT API TEST ==========\n")

    print("Testing audit health endpoint...")

    response = client.get("/audit/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    print("Audit health: SUCCESS")

    print("\nTesting audit event retrieval...")

    with patch(
        "src.api.audit_routes.load_audit_events",
        return_value=sample_audit_events(),
    ):

        response = client.get("/audit/events")

        assert response.status_code == 200

        result = response.json()

        assert result["total_events"] == 4
        assert len(result["events"]) == 4

    print("Audit event retrieval: SUCCESS")

    print("\nTesting model name filtering...")

    with patch(
        "src.api.audit_routes.load_audit_events",
        return_value=sample_audit_events(),
    ):

        response = client.get(
            "/audit/events",
            params={"model_name": "Random Forest"},
        )

        assert response.status_code == 200

        result = response.json()

        assert result["total_events"] == 2

        assert all(
            event["model_name"] == "Random Forest"
            for event in result["events"]
        )

    print("Model name filtering: SUCCESS")

    print("\nTesting event type filtering...")

    with patch(
        "src.api.audit_routes.load_audit_events",
        return_value=sample_audit_events(),
    ):

        response = client.get(
            "/audit/events",
            params={"event_type": "promotion"},
        )

        assert response.status_code == 200
        assert response.json()["total_events"] == 1

    print("Event type filtering: SUCCESS")

    print("\nTesting audit event limit...")

    with patch(
        "src.api.audit_routes.load_audit_events",
        return_value=sample_audit_events(),
    ):

        response = client.get(
            "/audit/events",
            params={"limit": 2},
        )

        assert response.status_code == 200
        assert response.json()["total_events"] == 2

    print("Audit event limit: SUCCESS")

    print("\nTesting audit summary...")

    with patch(
        "src.api.audit_routes.load_audit_events",
        return_value=sample_audit_events(),
    ):

        response = client.get("/audit/summary")

        assert response.status_code == 200

        result = response.json()

        assert result["total_events"] == 4
        assert result["event_counts"]["registration"] == 2
        assert result["event_counts"]["promotion"] == 1
        assert result["event_counts"]["archival"] == 1

    print("Audit summary: SUCCESS")

    print("\nTesting audit JSONL file loading...")

    with tempfile.TemporaryDirectory() as temp_dir:

        audit_file = Path(temp_dir) / "audit_log.jsonl"

        events = sample_audit_events()

        with audit_file.open("w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event) + "\n")

        loaded_events = load_audit_events(audit_file)

        assert len(loaded_events) == 4
        assert loaded_events[0]["model_name"] == "Logistic Regression"

    print("Audit JSONL loading: SUCCESS")

    print("\nTesting missing audit file...")

    with tempfile.TemporaryDirectory() as temp_dir:

        missing_file = Path(temp_dir) / "missing.jsonl"

        loaded_events = load_audit_events(missing_file)

        assert loaded_events == []

    print("Missing audit file handling: SUCCESS")

    print("\nTesting Swagger registration...")

    schema = client.get("/openapi.json").json()

    assert "/audit/health" in schema["paths"]
    assert "/audit/events" in schema["paths"]
    assert "/audit/summary" in schema["paths"]

    print("Swagger registration: SUCCESS")

    print(
        "\n========== MODEL REGISTRY AUDIT API TEST PASSED ==========\n"
    )


if __name__ == "__main__":
    main()
