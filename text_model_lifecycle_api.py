
"""Tests for model lifecycle API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import model_lifecycle_routes

client = TestClient(app)


@pytest.fixture
def mock_registry(monkeypatch):
    """Provide an isolated in-memory model registry."""

    records = [
        {
            "model_name": "Random_Forest",
            "version": "1",
            "stage": "candidate",
        }
    ]

    def mock_list_registered_models(registry_dir=None):
        return records

    def mock_promote_model(
        model_name,
        version,
        registry_dir=None,
        human_approved=False,
    ):
        if not human_approved:
            raise ValueError("Human approval is required.")

        for record in records:
            if (
                record["model_name"] == model_name
                and record["version"] == version
            ):
                record["stage"] = "champion"

    monkeypatch.setattr(
        model_lifecycle_routes.registry,
        "list_registered_models",
        mock_list_registered_models,
    )

    monkeypatch.setattr(
        model_lifecycle_routes.registry,
        "promote_model",
        mock_promote_model,
    )

    monkeypatch.setattr(
        model_lifecycle_routes,
        "REGISTRY_DIR_OVERRIDE",
        "test_registry",
    )

    return records


def test_promotion_requires_approval():
    """Missing approval should be rejected."""

    response = client.post(
        "/models/Random_Forest/1/promote",
        json={},
    )

    assert response.status_code == 422


def test_promotion_rejects_false_approval():
    """False approval should not promote a model."""

    response = client.post(
        "/models/Random_Forest/1/promote",
        json={"human_approved": False},
    )

    assert response.status_code == 403


def test_promotion_rejects_non_boolean_approval():
    """Approval must be a strict boolean."""

    response = client.post(
        "/models/Random_Forest/1/promote",
        json={"human_approved": "true"},
    )

    assert response.status_code == 422


def test_promotion_succeeds_with_approval(mock_registry):
    """Approved promotion should update the registry."""

    response = client.post(
        "/models/Random_Forest/1/promote",
        json={"human_approved": True},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "success"
    assert result["model"]["stage"] == "champion"

    assert mock_registry[0]["stage"] == "champion"


def test_promotion_rejects_unknown_model(mock_registry):
    """Unknown models should return 404."""

    response = client.post(
        "/models/Unknown_Model/1/promote",
        json={"human_approved": True},
    )

    assert response.status_code == 404
