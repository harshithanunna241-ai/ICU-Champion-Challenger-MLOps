
"""Integration test for live audit logging and FastAPI endpoints."""

import json

from fastapi.testclient import TestClient

import src.api.audit_routes as audit_routes
from src.api.main import app
from src.registry.audit_logger import log_audit_event


def response_text(response):
    """Convert an API response into searchable text."""
    return json.dumps(response.json(), default=str).lower()


def main():
    print("\n========== LIVE AUDIT API INTEGRATION TEST ==========\n")

    audit_file = audit_routes.DEFAULT_AUDIT_PATH.parent / "test_live_audit_api.jsonl"

    audit_file.parent.mkdir(parents=True, exist_ok=True)

    if audit_file.exists():
        audit_file.unlink()

    # Make the API read from the test audit file.
    audit_routes.DEFAULT_AUDIT_PATH = audit_file

    # Generate actual audit events.
    print("Creating audit events...")

    log_audit_event(
        event_type="registration",
        model_name="Logistic Regression",
        version="1",
        details={"source": "integration_test"},
        audit_path=audit_file,
    )

    log_audit_event(
        event_type="promotion",
        model_name="Logistic Regression",
        version="1",
        details={"stage": "champion"},
        audit_path=audit_file,
    )

    log_audit_event(
        event_type="registration",
        model_name="Random Forest",
        version="1",
        details={"source": "integration_test"},
        audit_path=audit_file,
    )

    log_audit_event(
        event_type="archival",
        model_name="Random Forest",
        version="1",
        details={"reason": "test_archival"},
        audit_path=audit_file,
    )

    assert audit_file.exists(), "Audit log file was not created."

    print("Audit event creation: SUCCESS")

    # Test FastAPI endpoints.
    client = TestClient(app)

    print("\nTesting audit health endpoint...")
    health_response = client.get("/audit/health")

    assert health_response.status_code == 200
    print("Audit health endpoint: SUCCESS")

    print("\nTesting audit events endpoint...")
    events_response = client.get("/audit/events")

    assert events_response.status_code == 200

    events_text = response_text(events_response)

    assert "logistic regression" in events_text
    assert "random forest" in events_text
    assert "registration" in events_text
    assert "promotion" in events_text
    assert "archival" in events_text

    print("Audit events endpoint: SUCCESS")

    print("\nTesting model-name filter...")
    filtered_response = client.get(
        "/audit/events",
        params={"model_name": "Logistic Regression"},
    )

    assert filtered_response.status_code == 200

    filtered_text = response_text(filtered_response)

    assert "logistic regression" in filtered_text
    assert "random forest" not in filtered_text

    print("Model-name filter: SUCCESS")

    print("\nTesting event-type filter...")
    event_filter_response = client.get(
        "/audit/events",
        params={"event_type": "promotion"},
    )

    assert event_filter_response.status_code == 200

    event_filter_text = response_text(event_filter_response)

    assert "promotion" in event_filter_text
    assert "registration" not in event_filter_text

    print("Event-type filter: SUCCESS")

    print("\nTesting event limit...")
    limit_response = client.get(
        "/audit/events",
        params={"limit": 1},
    )

    assert limit_response.status_code == 200

    limit_data = limit_response.json()

    if isinstance(limit_data, list):
        assert len(limit_data) <= 1
    elif isinstance(limit_data, dict):
        event_items = limit_data.get("events")
        if isinstance(event_items, list):
            assert len(event_items) <= 1

    print("Event limit: SUCCESS")

    print("\nTesting audit summary endpoint...")
    summary_response = client.get("/audit/summary")

    assert summary_response.status_code == 200

    print("Audit summary endpoint: SUCCESS")

    print("\n========== LIVE AUDIT API INTEGRATION TEST PASSED ==========")


if __name__ == "__main__":
    main()
