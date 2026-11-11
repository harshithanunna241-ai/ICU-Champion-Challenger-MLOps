
from unittest.mock import MagicMock, call

import src.orchestration.train_and_track as workflow


def make_test_model():
    """Create a lightweight mock model."""

    model = MagicMock()
    estimator = MagicMock()
    estimator.get_params.return_value = {"max_iter": 1000}
    model.steps = [("classifier", estimator)]

    return model


def make_metrics():
    return {
        "accuracy": 0.90,
        "precision": 0.85,
        "recall": 0.80,
        "f1_score": 0.82,
        "auroc": 0.95,
    }


def configure_workflow(monkeypatch):
    """Mock all model training and tracking operations."""

    model = make_test_model()
    metrics = make_metrics()

    for name in (
        "train_logistic_model",
        "train_random_forest_model",
        "train_xgboost_model",
    ):
        monkeypatch.setattr(
            workflow,
            name,
            lambda data, model=model: (model, ["heart_rate"]),
        )

    for name in (
        "evaluate_logistic_model",
        "evaluate_random_forest_model",
        "evaluate_xgboost_model",
    ):
        monkeypatch.setattr(
            workflow,
            name,
            lambda model, data, columns, metrics=metrics: metrics,
        )

    monkeypatch.setattr(
        workflow,
        "track_model_run",
        lambda **kwargs: {"run_id": "test-run"},
    )

    return model


def test_training_does_not_register_candidates_by_default(monkeypatch):
    configure_workflow(monkeypatch)

    register_mock = MagicMock()
    monkeypatch.setattr(
        workflow,
        "register_model",
        register_mock,
    )

    results = workflow.train_and_track_models(
        train_data=object(),
        test_data=object(),
    )

    assert len(results) == 3
    assert register_mock.call_count == 0

    for result in results.values():
        assert result["registry"] is None
        assert result["tracking"]["run_id"] == "test-run"


def test_training_registers_all_models_as_candidates(monkeypatch):
    configure_workflow(monkeypatch)

    register_mock = MagicMock(
        side_effect=[
            {"stage": "candidate"},
            {"stage": "candidate"},
            {"stage": "candidate"},
        ]
    )

    monkeypatch.setattr(
        workflow,
        "register_model",
        register_mock,
    )

    results = workflow.train_and_track_models(
        train_data=object(),
        test_data=object(),
        register_candidates=True,
        registry_dir="test_registry",
    )

    assert len(results) == 3
    assert register_mock.call_count == 3

    for result in results.values():
        assert result["registry"]["stage"] == "candidate"

    for registered_call in register_mock.call_args_list:
        assert registered_call.kwargs["stage"] == "candidate"
        assert registered_call.kwargs["version"] == "v1"
        assert registered_call.kwargs["registry_dir"] == "test_registry"
