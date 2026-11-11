
"""Unit tests for automated Champion-Challenger promotion workflow."""

import pytest

from src.orchestration.automated_promotion import run_automated_promotion


@pytest.fixture
def model_data():
    """Provide synthetic model comparison inputs."""
    return {
        "champion_model": object(),
        "challenger_model": object(),
        "test_data": object(),
        "champion_features": ["age", "heart_rate"],
        "challenger_features": ["age", "heart_rate"],
    }


@pytest.fixture
def comparison_result():
    """Provide synthetic Champion and Challenger metrics."""
    return {
        "champion": {
            "model_name": "Champion",
            "auprc": 0.80,
            "recall": 0.70,
            "brier_score": 0.20,
        },
        "challenger": {
            "model_name": "Challenger",
            "auprc": 0.90,
            "recall": 0.75,
            "brier_score": 0.15,
        },
    }


@pytest.fixture
def eligible_promotion_result():
    """Provide an eligible decision requiring human approval."""
    return {
        "promotion_eligible": True,
        "promotion_approved": False,
        "human_approval_required": True,
        "human_approved": False,
        "decision": "ELIGIBLE - HUMAN APPROVAL REQUIRED",
        "current_champion": "Champion",
        "challenger": "Challenger",
        "reasons": [],
    }


def test_successful_automated_promotion_workflow(
    monkeypatch,
    model_data,
    comparison_result,
    eligible_promotion_result,
):
    """Test successful comparison, evaluation and audit."""

    audit_result = {
        "audit_file": "test_audit.jsonl",
        "status": "saved",
    }

    def mock_compare_models(**kwargs):
        return comparison_result

    def mock_evaluate_promotion(**kwargs):
        return eligible_promotion_result

    def mock_save_audit(**kwargs):
        return audit_result

    monkeypatch.setattr(
        "src.orchestration.automated_promotion.compare_models",
        mock_compare_models,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.evaluate_promotion",
        mock_evaluate_promotion,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.save_promotion_audit",
        mock_save_audit,
    )

    result = run_automated_promotion(**model_data)

    assert result["comparison"] == comparison_result
    assert result["promotion_result"] == eligible_promotion_result
    assert result["audit_result"] == audit_result


def test_workflow_requires_human_approval(
    monkeypatch,
    model_data,
    comparison_result,
    eligible_promotion_result,
):
    """An eligible Challenger must not be automatically approved."""

    monkeypatch.setattr(
        "src.orchestration.automated_promotion.compare_models",
        lambda **kwargs: comparison_result,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.evaluate_promotion",
        lambda **kwargs: eligible_promotion_result,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.save_promotion_audit",
        lambda **kwargs: {"audit_file": "test_audit.jsonl"},
    )

    result = run_automated_promotion(**model_data)

    promotion = result["promotion_result"]

    assert promotion["promotion_eligible"] is True
    assert promotion["human_approval_required"] is True
    assert promotion["human_approved"] is False
    assert promotion["promotion_approved"] is False


def test_workflow_passes_correct_inputs_to_comparison(
    monkeypatch,
    model_data,
    comparison_result,
    eligible_promotion_result,
):
    """Verify that comparison receives all supplied model inputs."""

    captured = {}

    def mock_compare_models(**kwargs):
        captured.update(kwargs)
        return comparison_result

    monkeypatch.setattr(
        "src.orchestration.automated_promotion.compare_models",
        mock_compare_models,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.evaluate_promotion",
        lambda **kwargs: eligible_promotion_result,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.save_promotion_audit",
        lambda **kwargs: {"audit_file": "test_audit.jsonl"},
    )

    run_automated_promotion(**model_data)

    assert captured["champion_model"] is model_data["champion_model"]
    assert captured["challenger_model"] is model_data["challenger_model"]
    assert captured["test_data"] is model_data["test_data"]
    assert captured["champion_features"] == ["age", "heart_rate"]
    assert captured["challenger_features"] == ["age", "heart_rate"]


def test_workflow_saves_audit_with_correct_metrics(
    monkeypatch,
    model_data,
    comparison_result,
    eligible_promotion_result,
):
    """Verify that the audit receives the evaluation results."""

    captured = {}

    def mock_save_audit(**kwargs):
        captured.update(kwargs)
        return {"audit_file": "test_audit.jsonl"}

    monkeypatch.setattr(
        "src.orchestration.automated_promotion.compare_models",
        lambda **kwargs: comparison_result,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.evaluate_promotion",
        lambda **kwargs: eligible_promotion_result,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.save_promotion_audit",
        mock_save_audit,
    )

    run_automated_promotion(**model_data)

    assert captured["promotion_result"] == eligible_promotion_result
    assert captured["champion_metrics"] == comparison_result["champion"]
    assert captured["challenger_metrics"] == comparison_result["challenger"]


def test_workflow_passes_audit_directory(
    monkeypatch,
    model_data,
    comparison_result,
    eligible_promotion_result,
    tmp_path,
):
    """Verify that the requested audit directory is forwarded."""

    captured = {}

    def mock_save_audit(**kwargs):
        captured.update(kwargs)
        return {"audit_file": "test_audit.jsonl"}

    monkeypatch.setattr(
        "src.orchestration.automated_promotion.compare_models",
        lambda **kwargs: comparison_result,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.evaluate_promotion",
        lambda **kwargs: eligible_promotion_result,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.save_promotion_audit",
        mock_save_audit,
    )

    run_automated_promotion(
        **model_data,
        audit_dir=tmp_path,
    )

    assert captured["audit_dir"] == tmp_path


def test_comparison_failure_stops_workflow(
    monkeypatch,
    model_data,
):
    """An evaluation must not run when model comparison fails."""

    def mock_compare_models(**kwargs):
        raise ValueError("Comparison failed")

    monkeypatch.setattr(
        "src.orchestration.automated_promotion.compare_models",
        mock_compare_models,
    )

    with pytest.raises(ValueError, match="Comparison failed"):
        run_automated_promotion(**model_data)


def test_audit_failure_is_propagated(
    monkeypatch,
    model_data,
    comparison_result,
    eligible_promotion_result,
):
    """An audit failure must not be silently ignored."""

    monkeypatch.setattr(
        "src.orchestration.automated_promotion.compare_models",
        lambda **kwargs: comparison_result,
    )
    monkeypatch.setattr(
        "src.orchestration.automated_promotion.evaluate_promotion",
        lambda **kwargs: eligible_promotion_result,
    )

    def mock_save_audit(**kwargs):
        raise OSError("Audit storage failed")

    monkeypatch.setattr(
        "src.orchestration.automated_promotion.save_promotion_audit",
        mock_save_audit,
    )

    with pytest.raises(OSError, match="Audit storage failed"):
        run_automated_promotion(**model_data)
