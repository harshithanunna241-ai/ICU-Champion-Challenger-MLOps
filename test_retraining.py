
"""Tests for safe ICU model retraining decisions."""

import pytest

from src.orchestration.retraining import should_retrain


DEFAULT_POLICY = {
    "minimum_auroc": 0.75,
    "minimum_auprc": 0.75,
    "minimum_recall": 0.60,
    "maximum_brier_score": 0.25,
}


def create_monitoring_summary(
    auroc=0.90,
    auprc=0.85,
    recall=0.80,
    brier_score=0.15,
    labels_matured=True,
    feature_drift=None,
    prediction_drift=None,
):
    """Create a monitoring report for testing."""

    return {
        "labels_matured": labels_matured,
        "monitoring_report": {
            "model_performance": {
                "auroc": auroc,
                "auprc": auprc,
                "recall": recall,
                "brier_score": brier_score,
            },
            "feature_drift": feature_drift or [],
            "prediction_drift": prediction_drift or {
                "status": "NO DRIFT",
                "psi": 0.05,
            },
        },
    }


def test_good_performance_does_not_trigger_retraining():
    summary = create_monitoring_summary()

    result = should_retrain(summary, DEFAULT_POLICY)

    assert result["retrain_required"] is False
    assert result["decision"] == "NO_RETRAINING"
    assert result["triggered_metrics"] == []


def test_unmatured_labels_do_not_trigger_retraining():
    summary = create_monitoring_summary(
        auroc=0.40,
        auprc=0.30,
        labels_matured=False,
    )

    result = should_retrain(summary, DEFAULT_POLICY)

    assert result["retrain_required"] is False
    assert result["decision"] == "NO_RETRAINING"
    assert result["labels_matured"] is False


def test_feature_drift_alone_does_not_trigger_retraining():
    summary = create_monitoring_summary(
        feature_drift=[
            {"feature": "heart_rate", "drift_status": "DRIFT DETECTED"}
        ]
    )

    result = should_retrain(summary, DEFAULT_POLICY)

    assert result["retrain_required"] is False
    assert result["decision"] == "NO_RETRAINING"


def test_prediction_drift_alone_does_not_trigger_retraining():
    summary = create_monitoring_summary(
        prediction_drift={
            "status": "DRIFT DETECTED",
            "psi": 0.50,
        }
    )

    result = should_retrain(summary, DEFAULT_POLICY)

    assert result["retrain_required"] is False
    assert result["decision"] == "NO_RETRAINING"


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("auroc", 0.60),
        ("auprc", 0.60),
        ("recall", 0.40),
        ("brier_score", 0.35),
    ],
)
def test_performance_degradation_recommends_retraining(
    metric,
    value,
):
    summary = create_monitoring_summary(**{metric: value})

    result = should_retrain(summary, DEFAULT_POLICY)

    assert result["retrain_required"] is True
    assert result["decision"] == "RETRAINING_RECOMMENDED"
    assert any(
        item["metric"] == metric
        for item in result["triggered_metrics"]
    )


def test_multiple_degraded_metrics_are_reported():
    summary = create_monitoring_summary(
        auroc=0.60,
        auprc=0.50,
        recall=0.40,
        brier_score=0.35,
    )

    result = should_retrain(summary, DEFAULT_POLICY)

    assert result["retrain_required"] is True
    assert len(result["triggered_metrics"]) == 4


def test_integrated_monitoring_report_is_supported():
    report = create_monitoring_summary()["monitoring_report"]
    report["labels_matured"] = True

    result = should_retrain(report, DEFAULT_POLICY)

    assert result["retrain_required"] is False


def test_missing_performance_metric_is_rejected():
    summary = create_monitoring_summary()
    del summary["monitoring_report"]["model_performance"]["auprc"]

    with pytest.raises(ValueError, match="auprc"):
        should_retrain(summary, DEFAULT_POLICY)


def test_invalid_performance_metric_is_rejected():
    summary = create_monitoring_summary(auroc=float("nan"))

    with pytest.raises(ValueError, match="auroc"):
        should_retrain(summary, DEFAULT_POLICY)


def test_invalid_policy_threshold_is_rejected():
    invalid_policy = {
        "minimum_auprc": 1.5,
    }

    with pytest.raises(ValueError):
        should_retrain(
            create_monitoring_summary(),
            invalid_policy,
        )


def test_unknown_policy_key_is_rejected():
    invalid_policy = {
        "minimum_auprc": 0.75,
        "drift_threshold": 0.20,
    }

    with pytest.raises(ValueError, match="Unsupported"):
        should_retrain(
            create_monitoring_summary(),
            invalid_policy,
        )


def test_non_dictionary_summary_is_rejected():
    with pytest.raises(TypeError):
        should_retrain([], DEFAULT_POLICY)


def test_non_dictionary_policy_is_rejected():
    with pytest.raises(TypeError):
        should_retrain(create_monitoring_summary(), [])


def test_empty_policy_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        should_retrain(create_monitoring_summary(), {})


def test_boolean_metric_is_rejected():
    summary = create_monitoring_summary(auroc=True)

    with pytest.raises(ValueError, match="auroc"):
        should_retrain(summary, DEFAULT_POLICY)
def test_default_retraining_policy_loads_successfully():
    from src.orchestration.retraining import load_retraining_policy

    policy = load_retraining_policy()

    assert policy["enabled"] is True
    assert policy["require_matured_labels"] is True
    assert policy["automatic_retraining"] is False
    assert policy["automatic_promotion"] is False
    assert policy["human_approval_required"] is True


def test_default_policy_has_performance_thresholds():
    from src.orchestration.retraining import load_retraining_policy

    policy = load_retraining_policy()

    thresholds = policy["performance_thresholds"]

    assert thresholds["minimum_auroc"] == 0.75
    assert thresholds["minimum_auprc"] == 0.75
    assert thresholds["minimum_recall"] == 0.60
    assert thresholds["maximum_brier_score"] == 0.25


def test_missing_policy_file_is_rejected(tmp_path):
    from src.orchestration.retraining import load_retraining_policy

    missing_file = tmp_path / "missing_policy.yaml"

    with pytest.raises(FileNotFoundError):
        load_retraining_policy(missing_file)


def test_unsafe_automatic_retraining_policy_is_rejected(tmp_path):
    from src.orchestration.retraining import load_retraining_policy

    policy_file = tmp_path / "unsafe_policy.yaml"

    policy_file.write_text(
        """
enabled: true
require_matured_labels: true
performance_thresholds:
  minimum_auroc: 0.75
  minimum_auprc: 0.75
  minimum_recall: 0.60
  maximum_brier_score: 0.25
drift_is_not_performance_failure: true
automatic_retraining: true
automatic_promotion: false
human_approval_required: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_retraining_policy(policy_file)
