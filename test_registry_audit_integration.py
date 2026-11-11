
"""Integration tests for model registry and automatic audit logging."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.registry.model_registry import (
    register_model,
    promote_model,
    archive_model,
    list_registered_models,
)
from src.registry.audit_logger import log_audit_event


def main():

    print("\n========== REGISTRY AUDIT INTEGRATION TEST ==========\n")

    with tempfile.TemporaryDirectory() as temp_dir:

        registry_dir = Path(temp_dir) / "model_registry"
        audit_file = Path(temp_dir) / "audit_log.jsonl"

        # Redirect automatic audit logging to the temporary test file.
        def test_audit_logger(
            event_type,
            model_name,
            version,
            details=None,
            audit_path=None,
        ):
            return log_audit_event(
                event_type=event_type,
                model_name=model_name,
                version=version,
                details=details,
                audit_path=audit_file,
            )

        with patch(
            "src.registry.model_registry.log_audit_event",
            side_effect=test_audit_logger,
        ):

            print("Testing Logistic Regression registration...")

            logistic_model = LogisticRegression()

            register_model(
                model=logistic_model,
                model_name="Logistic Regression",
                version="1",
                metrics={
                    "auroc": 0.88,
                    "auprc": 0.80,
                    "recall": 0.75,
                    "brier_score": 0.20,
                },
                feature_columns=["heart_rate", "mean_bp"],
                stage="candidate",
                registry_dir=registry_dir,
            )

            print("Logistic Regression registration: SUCCESS")

            print("\nTesting Random Forest registration...")

            random_forest_model = RandomForestClassifier(
                n_estimators=10,
                random_state=42,
            )

            register_model(
                model=random_forest_model,
                model_name="Random Forest",
                version="1",
                metrics={
                    "auroc": 0.91,
                    "auprc": 0.85,
                    "recall": 0.82,
                    "brier_score": 0.17,
                },
                feature_columns=["heart_rate", "mean_bp"],
                stage="candidate",
                registry_dir=registry_dir,
            )

            print("Random Forest registration: SUCCESS")

            print("\nTesting third model registration...")

            extra_model = LogisticRegression()

            register_model(
                model=extra_model,
                model_name="Extra Model",
                version="1",
                metrics={
                    "auroc": 0.82,
                    "auprc": 0.78,
                    "recall": 0.70,
                    "brier_score": 0.22,
                },
                feature_columns=["heart_rate", "mean_bp"],
                stage="candidate",
                registry_dir=registry_dir,
            )

            print("Third model registration: SUCCESS")

            print("\nTesting first Champion promotion...")

            promote_model(
                model_name="Logistic Regression",
                version="1",
                registry_dir=registry_dir,
            )

            records = list_registered_models(registry_dir)

            champion = [
                record for record in records
                if record["stage"] == "champion"
            ]

            assert len(champion) == 1
            assert champion[0]["model_name"] == "Logistic Regression"

            print("First Champion promotion: SUCCESS")

            print("\nTesting second Champion promotion...")

            promote_model(
                model_name="Random Forest",
                version="1",
                registry_dir=registry_dir,
            )

            records = list_registered_models(registry_dir)

            champion = [
                record for record in records
                if record["stage"] == "champion"
            ]

            archived_logistic = [
                record for record in records
                if (
                    record["model_name"] == "Logistic Regression"
                    and record["stage"] == "archived"
                )
            ]

            assert len(champion) == 1
            assert champion[0]["model_name"] == "Random Forest"
            assert len(archived_logistic) == 1

            print("Champion replacement and automatic archival: SUCCESS")

            print("\nTesting candidate archival...")

            archive_model(
                model_name="Extra Model",
                version="1",
                registry_dir=registry_dir,
            )

            records = list_registered_models(registry_dir)

            extra_record = next(
                record for record in records
                if record["model_name"] == "Extra Model"
            )

            assert extra_record["stage"] == "archived"

            print("Candidate archival: SUCCESS")

            print("\nTesting audit log creation...")

            assert audit_file.exists()

            with audit_file.open("r", encoding="utf-8") as file:
                events = [
                    json.loads(line)
                    for line in file
                    if line.strip()
                ]

            assert len(events) == 7

            print("Audit log creation: SUCCESS")

            print("\nTesting registration audit events...")

            registration_events = [
                event for event in events
                if event["event_type"] == "registration"
            ]

            assert len(registration_events) == 3

            print("Registration audit events: SUCCESS")

            print("\nTesting promotion audit events...")

            promotion_events = [
                event for event in events
                if event["event_type"] == "promotion"
            ]

            assert len(promotion_events) == 2

            print("Promotion audit events: SUCCESS")

            print("\nTesting archival audit events...")

            archival_events = [
                event for event in events
                if event["event_type"] == "archival"
            ]

            assert len(archival_events) == 2

            print("Archival audit events: SUCCESS")

            print("\nTesting previous Champion archival details...")

            previous_champion_event = next(
                event for event in archival_events
                if event["model_name"] == "Logistic Regression"
            )

            assert previous_champion_event["version"] == "1"

            print("Previous Champion audit details: SUCCESS")

            print("\nTesting final Champion state...")

            assert champion[0]["model_name"] == "Random Forest"

            print("Final Champion verification: SUCCESS")

    print(
        "\n========== REGISTRY AUDIT INTEGRATION TEST PASSED ==========\n"
    )


if __name__ == "__main__":
    main()
