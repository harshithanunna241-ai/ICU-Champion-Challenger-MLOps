
"""Integration tests for monitoring and retraining decisions."""

import pytest

import src.monitoring.service as service


def create_monitoring_report(
    auroc=0.60,
    auprc=0.60,
    recall=0.40,
    brier_score=0.35,
):
    return {
        "feature_drift": [],
        "prediction_drift": {
            "status": "DRIFT DETECTED",
            "psi": 0.50,
        },
        "model_performance": {
            "auroc": auroc,
            "auprc": auprc,
            "recall": recall,
            "brier_score": brier_score,
        },
        "alerts": {
            "total_alerts": 1,
            "alerts": [],
        },
        "overall_status": "ATTENTION REQUIRED",
    }


@pytest.fixture
def mock_monitoring_components(monkeypatch):
    report = create_monitoring_report()

    monkeypatch.setattr(
        service,
        "build_monitoring_report",
        lambda **kwargs: report,
    )

    monkeypatch.setattr(
        service,
        "save_monitoring_report",
        lambda **kwargs: {"saved": True},
    )

    monkeypatch.setattr(
        service,
        "load_monitoring_history",
        lambda path: [{"overall_status": "ATTENTION REQUIRED"}],
    )

    monkeypatch.setattr(
        service,
        "analyze_monitoring_trends",
        lambda history: {"trend": "stable"},
    )

    monkeypatch.setattr(
        service,
        "summarize_monitoring_trends",
        lambda trends: {"summary": "Test summary"},
    )

    return report


def test_immature_labels_do_not_trigger_retraining(
    tmp_path,
    mock_monitoring_components,
):
    result = service.run_monitoring_service(
        reference_data=None,
        current_data=None,
        reference_predictions=None,
        current_predictions=None,
        evaluation_data=None,
        history_path=tmp_path / "history.jsonl",
        labels_matured=False,
    )

    decision = result["retraining_decision"]

    assert decision["retrain_required"] is False
    assert decision["decision"] == "NO_RETRAINING"
    assert decision["labels_matured"] is False
    assert result["monitoring_report"] is mock_monitoring_components


def test_matured_labels_with_degraded_performance_recommend_retraining(
    tmp_path,
    mock_monitoring_components,
):
    result = service.run_monitoring_service(
        reference_data=None,
        current_data=None,
        reference_predictions=None,
        current_predictions=None,
        evaluation_data=None,
        history_path=tmp_path / "history.jsonl",
        labels_matured=True,
    )

    decision = result["retraining_decision"]

    assert decision["retrain_required"] is True
    assert decision["decision"] == "RETRAINING_RECOMMENDED"
    assert decision["labels_matured"] is True
    assert decision["human_approval_required"] is True
    assert decision["automatic_retraining"] is False
    assert decision["automatic_promotion"] is False


def test_monitoring_outputs_are_preserved(
    tmp_path,
    mock_monitoring_components,
):
    result = service.run_monitoring_service(
        reference_data=None,
        current_data=None,
        reference_predictions=None,
        current_predictions=None,
        evaluation_data=None,
        history_path=tmp_path / "history.jsonl",
    )

    assert "monitoring_report" in result
    assert "retraining_decision" in result
    assert "storage_result" in result
    assert "history_records" in result
    assert "trends" in result
    assert "trend_summary" in result

    assert result["storage_result"]["saved"] is True
    assert result["history_records"] == 1


def test_non_boolean_label_maturity_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="labels_matured"):
        service.run_monitoring_service(
            reference_data=None,
            current_data=None,
            reference_predictions=None,
            current_predictions=None,
            evaluation_data=None,
            history_path=tmp_path / "history.jsonl",
            labels_matured="yes",
        )
