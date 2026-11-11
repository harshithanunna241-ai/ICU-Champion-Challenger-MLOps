
"""Unit tests for registry-based Champion-Challenger promotion."""

import pytest

from src.orchestration.registry_promotion import run_registry_promotion


@pytest.fixture
def registry_records():
    """Provide synthetic Champion and Candidate registry records."""
    return [
        {
            "model_name": "Champion",
            "version": "1",
            "stage": "champion",
            "feature_columns": ["age", "heart_rate"],
        },
        {
            "model_name": "Challenger",
            "version": "2",
            "stage": "candidate",
            "feature_columns": ["age", "heart_rate"],
        },
    ]


@pytest.fixture
def comparison_result():
    """Provide synthetic model comparison metrics."""
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
    """Provide an eligible promotion awaiting human approval."""
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


@pytest.fixture
def approved_promotion_result():
    """Provide an explicitly approved promotion result."""
    return {
        "promotion_eligible": True,
        "promotion_approved": True,
        "human_approval_required": False,
        "human_approved": True,
        "decision": "PROMOTION APPROVED",
        "current_champion": "Champion",
        "challenger": "Challenger",
        "reasons": [],
    }


def setup_successful_workflow(
    monkeypatch,
    registry_records,
    comparison_result,
    promotion_result,
):
    """Mock registry, comparison, policy and audit operations."""

    loaded_models = []

    def mock_load_registered_model(**kwargs):
        loaded_models.append(kwargs)
        return object()

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.list_registered_models",
        lambda **kwargs: registry_records,
    )

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.load_registered_model",
        mock_load_registered_model,
    )

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.compare_models",
        lambda **kwargs: comparison_result,
    )

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.evaluate_promotion",
        lambda **kwargs: promotion_result,
    )

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.save_promotion_audit",
        lambda **kwargs: {"audit_file": "test_audit.jsonl"},
    )

    return loaded_models


def test_successful_registry_workflow_without_approval(
    monkeypatch,
    registry_records,
    comparison_result,
    eligible_promotion_result,
):
    """Verify that an eligible Challenger is not automatically promoted."""

    setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        eligible_promotion_result,
    )

    promotion_calls = []

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.promote_model",
        lambda **kwargs: promotion_calls.append(kwargs),
    )

    result = run_registry_promotion(test_data=object())

    assert result["comparison"] == comparison_result
    assert result["promotion_result"] == eligible_promotion_result
    assert result["audit_result"]["audit_file"] == "test_audit.jsonl"
    assert result["registry_result"] is None
    assert result["promotion_executed"] is False
    assert promotion_calls == []


def test_registry_promotion_executes_only_when_approved(
    monkeypatch,
    registry_records,
    comparison_result,
    approved_promotion_result,
):
    """Verify that explicit approval allows registry promotion."""

    setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        approved_promotion_result,
    )

    expected_registry_result = {"status": "promoted"}
    promotion_calls = []

    def mock_promote_model(**kwargs):
        promotion_calls.append(kwargs)
        return expected_registry_result

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.promote_model",
        mock_promote_model,
    )

    result = run_registry_promotion(test_data=object())

    assert result["registry_result"] == expected_registry_result
    assert result["promotion_executed"] is True
    assert len(promotion_calls) == 1

    assert promotion_calls[0]["model_name"] == "Challenger"
    assert promotion_calls[0]["version"] == "2"


def test_registry_requires_exactly_one_champion(
    monkeypatch,
    registry_records,
):
    """Reject a registry without exactly one Champion."""

    records = [
        record
        for record in registry_records
        if record["stage"] != "champion"
    ]

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.list_registered_models",
        lambda **kwargs: records,
    )

    with pytest.raises(
        ValueError,
        match="exactly one Champion",
    ):
        run_registry_promotion(test_data=object())


def test_registry_rejects_multiple_champions(
    monkeypatch,
    registry_records,
):
    """Reject a registry containing multiple Champions."""

    records = registry_records + [
        {
            "model_name": "SecondChampion",
            "version": "3",
            "stage": "champion",
            "feature_columns": ["age"],
        }
    ]

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.list_registered_models",
        lambda **kwargs: records,
    )

    with pytest.raises(
        ValueError,
        match="exactly one Champion",
    ):
        run_registry_promotion(test_data=object())


def test_registry_requires_exactly_one_challenger(
    monkeypatch,
    registry_records,
):
    """Reject a registry without exactly one Candidate."""

    records = [
        record
        for record in registry_records
        if record["stage"] != "candidate"
    ]

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.list_registered_models",
        lambda **kwargs: records,
    )

    with pytest.raises(
        ValueError,
        match="exactly one Challenger",
    ):
        run_registry_promotion(test_data=object())


def test_registry_rejects_multiple_challengers(
    monkeypatch,
    registry_records,
):
    """Reject a registry containing multiple Candidates."""

    records = registry_records + [
        {
            "model_name": "SecondChallenger",
            "version": "3",
            "stage": "candidate",
            "feature_columns": ["age"],
        }
    ]

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.list_registered_models",
        lambda **kwargs: records,
    )

    with pytest.raises(
        ValueError,
        match="exactly one Challenger",
    ):
        run_registry_promotion(test_data=object())


def test_registry_loads_both_registered_models(
    monkeypatch,
    registry_records,
    comparison_result,
    eligible_promotion_result,
):
    """Verify that the correct model names and versions are loaded."""

    loaded_models = setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        eligible_promotion_result,
    )

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.promote_model",
        lambda **kwargs: None,
    )

    run_registry_promotion(test_data=object())

    assert len(loaded_models) == 2

    assert loaded_models[0]["model_name"] == "Champion"
    assert loaded_models[0]["version"] == "1"

    assert loaded_models[1]["model_name"] == "Challenger"
    assert loaded_models[1]["version"] == "2"


def test_registry_passes_correct_features_to_comparison(
    monkeypatch,
    registry_records,
    comparison_result,
    eligible_promotion_result,
):
    """Verify that registered feature columns reach comparison."""

    captured = {}

    setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        eligible_promotion_result,
    )

    def mock_compare_models(**kwargs):
        captured.update(kwargs)
        return comparison_result

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.compare_models",
        mock_compare_models,
    )

    run_registry_promotion(test_data="synthetic_test_data")

    assert captured["test_data"] == "synthetic_test_data"
    assert captured["champion_features"] == ["age", "heart_rate"]
    assert captured["challenger_features"] == ["age", "heart_rate"]


def test_registry_saves_audit_with_correct_metrics(
    monkeypatch,
    registry_records,
    comparison_result,
    eligible_promotion_result,
):
    """Verify that the audit receives the comparison metrics."""

    captured = {}

    setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        eligible_promotion_result,
    )

    def mock_save_audit(**kwargs):
        captured.update(kwargs)
        return {"audit_file": "test_audit.jsonl"}

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.save_promotion_audit",
        mock_save_audit,
    )

    run_registry_promotion(test_data=object())

    assert captured["promotion_result"] == eligible_promotion_result
    assert captured["champion_metrics"] == comparison_result["champion"]
    assert captured["challenger_metrics"] == comparison_result["challenger"]


def test_registry_forwards_custom_directories(
    monkeypatch,
    registry_records,
    comparison_result,
    eligible_promotion_result,
    tmp_path,
):
    """Verify that custom registry and audit directories are forwarded."""

    captured = {}

    def mock_list_registered_models(**kwargs):
        captured["registry_dir"] = kwargs["registry_dir"]
        return registry_records

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.list_registered_models",
        mock_list_registered_models,
    )

    setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        eligible_promotion_result,
    )

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.list_registered_models",
        mock_list_registered_models,
    )

    def mock_save_audit(**kwargs):
        captured["audit_dir"] = kwargs["audit_dir"]
        return {"audit_file": "test_audit.jsonl"}

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.save_promotion_audit",
        mock_save_audit,
    )

    run_registry_promotion(
        test_data=object(),
        registry_dir=tmp_path / "registry",
        audit_dir=tmp_path / "audit",
    )

    assert captured["registry_dir"] == tmp_path / "registry"
    assert captured["audit_dir"] == tmp_path / "audit"


def test_model_loading_failure_is_propagated(
    monkeypatch,
    registry_records,
):
    """Verify that a model loading error is not silently ignored."""

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.list_registered_models",
        lambda **kwargs: registry_records,
    )

    def mock_load_registered_model(**kwargs):
        raise FileNotFoundError("Model artifact not found")

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.load_registered_model",
        mock_load_registered_model,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found",
    ):
        run_registry_promotion(test_data=object())


def test_comparison_failure_is_propagated(
    monkeypatch,
    registry_records,
    comparison_result,
    eligible_promotion_result,
):
    """Verify that comparison errors stop the workflow."""

    setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        eligible_promotion_result,
    )

    def mock_compare_models(**kwargs):
        raise ValueError("Model comparison failed")

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.compare_models",
        mock_compare_models,
    )

    with pytest.raises(
        ValueError,
        match="Model comparison failed",
    ):
        run_registry_promotion(test_data=object())


def test_audit_failure_prevents_promotion(
    monkeypatch,
    registry_records,
    comparison_result,
    approved_promotion_result,
):
    """Verify that registry promotion cannot proceed without audit success."""

    setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        approved_promotion_result,
    )

    promotion_calls = []

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.promote_model",
        lambda **kwargs: promotion_calls.append(kwargs),
    )

    def mock_save_audit(**kwargs):
        raise OSError("Audit storage failed")

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.save_promotion_audit",
        mock_save_audit,
    )

    with pytest.raises(
        OSError,
        match="Audit storage failed",
    ):
        run_registry_promotion(test_data=object())

    assert promotion_calls == []


def test_registry_promotion_failure_is_propagated(
    monkeypatch,
    registry_records,
    comparison_result,
    approved_promotion_result,
):
    """Verify that a registry promotion error is propagated."""

    setup_successful_workflow(
        monkeypatch,
        registry_records,
        comparison_result,
        approved_promotion_result,
    )

    def mock_promote_model(**kwargs):
        raise RuntimeError("Registry promotion failed")

    monkeypatch.setattr(
        "src.orchestration.registry_promotion.promote_model",
        mock_promote_model,
    )

    with pytest.raises(
        RuntimeError,
        match="Registry promotion failed",
    ):
        run_registry_promotion(test_data=object())
