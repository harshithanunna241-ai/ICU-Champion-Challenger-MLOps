
"""Tests for the Model Registry Audit Logger."""

import json
import tempfile
from pathlib import Path

from src.registry.audit_logger import log_audit_event


def main():

    print("\n========== MODEL REGISTRY AUDIT LOGGER TEST ==========\n")

    with tempfile.TemporaryDirectory() as temp_dir:

        audit_file = Path(temp_dir) / "audit_log.jsonl"

        print("Testing model registration event...")

        registration = log_audit_event(
            event_type="registration",
            model_name="Logistic Regression",
            version="1",
            details={
                "auprc": 0.80,
                "auroc": 0.88,
            },
            audit_path=audit_file,
        )

        assert registration["event_type"] == "registration"
        assert registration["model_name"] == "Logistic Regression"
        assert registration["version"] == "1"

        print("Registration event: SUCCESS")

        print("\nTesting model promotion event...")

        promotion = log_audit_event(
            event_type="promotion",
            model_name="Random Forest",
            version="2",
            details={
                "previous_stage": "challenger",
                "new_stage": "champion",
            },
            audit_path=audit_file,
        )

        assert promotion["event_type"] == "promotion"
        assert promotion["details"]["new_stage"] == "champion"

        print("Promotion event: SUCCESS")

        print("\nTesting model archival event...")

        archival = log_audit_event(
            event_type="archival",
            model_name="Logistic Regression",
            version="1",
            details={
                "reason": "Replaced by a newer model",
            },
            audit_path=audit_file,
        )

        assert archival["event_type"] == "archival"

        print("Archival event: SUCCESS")

        print("\nTesting UTC timestamp...")

        assert registration["timestamp"].endswith("+00:00")

        print("UTC timestamp: SUCCESS")

        print("\nTesting unique event IDs...")

        event_ids = {
            registration["event_id"],
            promotion["event_id"],
            archival["event_id"],
        }

        assert len(event_ids) == 3

        print("Unique event IDs: SUCCESS")

        print("\nTesting JSONL file creation...")

        assert audit_file.exists()

        with audit_file.open("r", encoding="utf-8") as file:
            lines = file.readlines()

        assert len(lines) == 3

        print("JSONL file creation: SUCCESS")

        print("\nTesting JSONL data integrity...")

        loaded_events = [
            json.loads(line)
            for line in lines
        ]

        assert len(loaded_events) == 3
        assert loaded_events[0]["event_type"] == "registration"
        assert loaded_events[1]["event_type"] == "promotion"
        assert loaded_events[2]["event_type"] == "archival"

        print("JSONL data integrity: SUCCESS")

        print("\nTesting invalid event type...")

        try:
            log_audit_event(
                event_type="invalid_event",
                model_name="Test Model",
                version="1",
                audit_path=audit_file,
            )
            raise AssertionError("Invalid event type was accepted.")

        except ValueError:
            print("Invalid event validation: SUCCESS")

        print("\nTesting empty model name...")

        try:
            log_audit_event(
                event_type="registration",
                model_name="",
                version="1",
                audit_path=audit_file,
            )
            raise AssertionError("Empty model name was accepted.")

        except ValueError:
            print("Empty model name validation: SUCCESS")

        print("\nTesting invalid details...")

        try:
            log_audit_event(
                event_type="registration",
                model_name="Test Model",
                version="1",
                details="invalid",
                audit_path=audit_file,
            )
            raise AssertionError("Invalid details were accepted.")

        except ValueError:
            print("Details validation: SUCCESS")

        assert len(audit_file.read_text(encoding="utf-8").splitlines()) == 3

        print("\n========== MODEL REGISTRY AUDIT LOGGER TEST PASSED ==========\n")


if __name__ == "__main__":
    main()
