
"""Unit tests for Champion-Challenger promotion policy."""

import yaml
import pytest

from src.evaluation.promotion_policy import (
    load_evaluation_policy,
    evaluate_promotion,
)


@pytest.fixture
def valid_policy():
    return {
        "minimum_auprc": 0.80,
        "minimum_recall": 0.75,
        "maximum_brier_score": 0.20,
        "minimum_auprc_improvement": 0.02,
        "maximum_recall_drop": 0.05,
    }


@pytest.fixture
def champion_metrics():
    return {
        "auprc": 0.82,
        "recall": 0.85,
        "brier_score": 0.15,
    }


@pytest.fixture
def challenger_metrics():
    return {
        "auprc": 0.87,
        "recall": 0.83,
        "brier_score": 0.12,
    }


# ---------------------------------------------------------
# Tests for load_evaluation_policy
# ---------------------------------------------------------


def test_loads_valid_yaml_policy(tmp_path, valid_policy):
    config_path = tmp_path / "evaluation_policy.yaml"

    config_path.write_text(
        yaml.safe_dump(valid_policy),
        encoding="utf-8",
    )

    result = load_evaluation_policy(config_path)

    assert result == valid_policy


def test_returns_dictionary(tmp_path, valid_policy):
    config_path = tmp_path / "evaluation_policy.yaml"

    config_path.write_text(
        yaml.safe_dump(valid_policy),
        encoding="utf-8",
    )

    result = load_evaluation_policy(config_path)

    assert isinstance(result, dict)


def test_loads_all_required_policy_settings(
    tmp_path,
    valid_policy,
):
    config_path = tmp_path / "evaluation_policy.yaml"

    config_path.write_text(
        yaml.safe_dump(valid_policy),
        encoding="utf-8",
    )

    result = load_evaluation_policy(config_path)

    expected_keys = {
        "minimum_auprc",
        "minimum_recall",
        "maximum_brier_score",
        "minimum_auprc_improvement",
        "maximum_recall_drop",
    }

    assert set(result.keys()) == expected_keys


def test_raises_error_when_config_file_missing(tmp_path):
    config_path = tmp_path / "missing_policy.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation policy not found",
    ):
        load_evaluation_policy(config_path)


@pytest.mark.parametrize(
    "yaml_content",
    [
        "- item1\n- item2\n",
        '"just a string"\n',
        "123\n",
        "null\n",
    ],
)
def test_rejects_non_mapping_yaml(tmp_path, yaml_content):
    config_path = tmp_path / "evaluation_policy.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Evaluation policy must be a YAML mapping",
    ):
        load_evaluation_policy(config_path)


def test_rejects_empty_yaml_file(tmp_path):
    config_path = tmp_path / "evaluation_policy.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Evaluation policy must be a YAML mapping",
    ):
        load_evaluation_policy(config_path)


def test_rejects_invalid_yaml_syntax(tmp_path):
    config_path = tmp_path / "evaluation_policy.yaml"

    config_path.write_text(
        "minimum_auprc: [0.8\n",
        encoding="utf-8",
    )

    with pytest.raises(yaml.YAMLError):
        load_evaluation_policy(config_path)


@pytest.mark.parametrize(
    "missing_key",
    [
        "minimum_auprc",
        "minimum_recall",
        "maximum_brier_score",
        "minimum_auprc_improvement",
        "maximum_recall_drop",
    ],
)
def test_rejects_missing_policy_setting(
    tmp_path,
    valid_policy,
    missing_key,
):
    invalid_policy = valid_policy.copy()
    invalid_policy.pop(missing_key)

    config_path = tmp_path / "evaluation_policy.yaml"

    config_path.write_text(
        yaml.safe_dump(invalid_policy),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing policy settings",
    ):
        load_evaluation_policy(config_path)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.1,
        "0.8",
        None,
        True,
        False,
        [],
    ],
)
@pytest.mark.parametrize(
    "policy_key",
    [
        "minimum_auprc",
        "minimum_recall",
        "maximum_brier_score",
        "minimum_auprc_improvement",
        "maximum_recall_drop",
    ],
)
def test_rejects_invalid_policy_values(
    tmp_path,
    valid_policy,
    policy_key,
    invalid_value,
):
    invalid_policy = valid_policy.copy()
    invalid_policy[policy_key] = invalid_value

    config_path = tmp_path / "evaluation_policy.yaml"

    config_path.write_text(
        yaml.safe_dump(invalid_policy),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=f"{policy_key} must be a numeric value",
    ):
        load_evaluation_policy(config_path)


@pytest.mark.parametrize(
    "valid_value",
    [
        0,
        1,
        0.0,
        1.0,
    ],
)
def test_accepts_boundary_policy_values(
    tmp_path,
    valid_policy,
    valid_value,
):
    valid_policy["minimum_auprc"] = valid_value

    config_path = tmp_path / "evaluation_policy.yaml"

    config_path.write_text(
        yaml.safe_dump(valid_policy),
        encoding="utf-8",
    )

    result = load_evaluation_policy(config_path)

    assert result["minimum_auprc"] == valid_value


def test_accepts_integer_policy_values(tmp_path):
    policy = {
        "minimum_auprc": 1,
        "minimum_recall": 1,
        "maximum_brier_score": 0,
        "minimum_auprc_improvement": 0,
        "maximum_recall_drop": 0,
    }

    config_path = tmp_path / "evaluation_policy.yaml"

    config_path.write_text(
        yaml.safe_dump(policy),
        encoding="utf-8",
    )

    result = load_evaluation_policy(config_path)

    assert result == policy


# ---------------------------------------------------------
# Tests for evaluate_promotion
# ---------------------------------------------------------


def test_returns_dictionary(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert isinstance(result, dict)


def test_returns_expected_result_keys(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    expected_keys = {
        "promotion_eligible",
        "promotion_approved",
        "human_approval_required",
        "human_approved",
        "current_champion",
        "challenger",
        "auprc_improvement",
        "recall_drop",
        "reasons",
        "decision",
    }

    assert set(result.keys()) == expected_keys


def test_eligible_challenger_passes_all_policy_checks(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_eligible"] is True
    assert result["reasons"] == []
    assert result["decision"] == "ELIGIBLE - HUMAN APPROVAL REQUIRED"


def test_promotion_is_never_automatically_approved(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_approved"] is False


def test_human_approval_is_always_required(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["human_approval_required"] is True
    assert result["human_approved"] is False


def test_returns_expected_champion_and_challenger_names(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["current_champion"] == "Logistic Regression"
    assert result["challenger"] == "Random Forest"


def test_calculates_auprc_improvement_correctly(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["auprc_improvement"] == pytest.approx(0.05)


def test_calculates_recall_drop_correctly(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["recall_drop"] == pytest.approx(0.02)


def test_rejects_challenger_below_minimum_auprc(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["auprc"] = 0.79

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_eligible"] is False
    assert "Challenger AUPRC is below the minimum." in result["reasons"]


def test_rejects_challenger_below_minimum_recall(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["recall"] = 0.70

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_eligible"] is False
    assert "Challenger recall is below the minimum." in result["reasons"]


def test_rejects_challenger_above_maximum_brier_score(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["brier_score"] = 0.25

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_eligible"] is False
    assert "Challenger Brier score is too high." in result["reasons"]


def test_rejects_insufficient_auprc_improvement(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["auprc"] = 0.83

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_eligible"] is False

    assert (
        "Required AUPRC improvement was not achieved."
        in result["reasons"]
    )


def test_rejects_excessive_recall_drop(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["recall"] = 0.79

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_eligible"] is False

    assert (
        "Challenger recall dropped beyond the allowed limit."
        in result["reasons"]
    )


def test_returns_all_failure_reasons(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics.update(
        {
            "auprc": 0.70,
            "recall": 0.60,
            "brier_score": 0.30,
        }
    )

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_eligible"] is False
    assert len(result["reasons"]) == 5


def test_rejected_challenger_keeps_current_champion(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["auprc"] = 0.70

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["decision"] == "KEEP CURRENT CHAMPION"
    assert result["promotion_approved"] is False


def test_accepts_metrics_exactly_at_policy_boundaries(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    # Use binary-exact fractions to avoid floating-point
    # representation issues at the policy boundaries.
    valid_policy.update(
        {
            "minimum_auprc": 0.50,
            "minimum_recall": 0.50,
            "maximum_brier_score": 0.25,
            "minimum_auprc_improvement": 0.25,
            "maximum_recall_drop": 0.25,
        }
    )

    champion_metrics.update(
        {
            "auprc": 0.50,
            "recall": 0.75,
            "brier_score": 0.15,
        }
    )

    challenger_metrics.update(
        {
            "auprc": 0.75,
            "recall": 0.50,
            "brier_score": 0.25,
        }
    )

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["promotion_eligible"] is True
    assert result["reasons"] == []


def test_accepts_zero_recall_drop(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["recall"] = champion_metrics["recall"]

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["recall_drop"] == 0
    assert result["promotion_eligible"] is True


def test_accepts_negative_recall_drop(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["recall"] = 0.95

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["recall_drop"] < 0
    assert result["promotion_eligible"] is True


def test_accepts_negative_auprc_improvement_value(
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    challenger_metrics["auprc"] = 0.70

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
        valid_policy,
    )

    assert result["auprc_improvement"] < 0
    assert result["promotion_eligible"] is False


# ---------------------------------------------------------
# Input validation tests for evaluate_promotion
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_metrics",
    [
        None,
        [],
        "metrics",
        123,
    ],
)
def test_rejects_invalid_champion_metrics(
    invalid_metrics,
    valid_policy,
    challenger_metrics,
):
    with pytest.raises(
        ValueError,
        match="Champion metrics must be a dictionary",
    ):
        evaluate_promotion(
            invalid_metrics,
            challenger_metrics,
            valid_policy,
        )


@pytest.mark.parametrize(
    "invalid_metrics",
    [
        None,
        [],
        "metrics",
        123,
    ],
)
def test_rejects_invalid_challenger_metrics(
    invalid_metrics,
    valid_policy,
    champion_metrics,
):
    with pytest.raises(
        ValueError,
        match="Challenger metrics must be a dictionary",
    ):
        evaluate_promotion(
            champion_metrics,
            invalid_metrics,
            valid_policy,
        )


@pytest.mark.parametrize(
    "missing_metric",
    [
        "auprc",
        "recall",
        "brier_score",
    ],
)
def test_rejects_missing_champion_metric(
    missing_metric,
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    invalid_metrics = champion_metrics.copy()
    invalid_metrics.pop(missing_metric)

    with pytest.raises(
        ValueError,
        match="Champion metrics missing",
    ):
        evaluate_promotion(
            invalid_metrics,
            challenger_metrics,
            valid_policy,
        )


@pytest.mark.parametrize(
    "missing_metric",
    [
        "auprc",
        "recall",
        "brier_score",
    ],
)
def test_rejects_missing_challenger_metric(
    missing_metric,
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    invalid_metrics = challenger_metrics.copy()
    invalid_metrics.pop(missing_metric)

    with pytest.raises(
        ValueError,
        match="Challenger metrics missing",
    ):
        evaluate_promotion(
            champion_metrics,
            invalid_metrics,
            valid_policy,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.1,
        "0.8",
        None,
        True,
        False,
        [],
    ],
)
@pytest.mark.parametrize(
    "metric",
    [
        "auprc",
        "recall",
        "brier_score",
    ],
)
def test_rejects_invalid_champion_metric_values(
    metric,
    invalid_value,
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    invalid_metrics = champion_metrics.copy()
    invalid_metrics[metric] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"Champion {metric} must be numeric",
    ):
        evaluate_promotion(
            invalid_metrics,
            challenger_metrics,
            valid_policy,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.1,
        "0.8",
        None,
        True,
        False,
        [],
    ],
)
@pytest.mark.parametrize(
    "metric",
    [
        "auprc",
        "recall",
        "brier_score",
    ],
)
def test_rejects_invalid_challenger_metric_values(
    metric,
    invalid_value,
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    invalid_metrics = challenger_metrics.copy()
    invalid_metrics[metric] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"Challenger {metric} must be numeric",
    ):
        evaluate_promotion(
            champion_metrics,
            invalid_metrics,
            valid_policy,
        )


@pytest.mark.parametrize(
    "invalid_policy",
    [
        None,
        [],
        "policy",
        123,
    ],
)
def test_rejects_invalid_explicit_policy(
    invalid_policy,
    champion_metrics,
    challenger_metrics,
):
    if invalid_policy is None:
        return

    with pytest.raises(
        ValueError,
        match="Policy must be a dictionary",
    ):
        evaluate_promotion(
            champion_metrics,
            challenger_metrics,
            invalid_policy,
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "minimum_auprc",
        "minimum_recall",
        "maximum_brier_score",
        "minimum_auprc_improvement",
        "maximum_recall_drop",
    ],
)
def test_rejects_missing_explicit_policy_setting(
    missing_key,
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    invalid_policy = valid_policy.copy()
    invalid_policy.pop(missing_key)

    with pytest.raises(
        ValueError,
        match=f"Invalid policy setting: {missing_key}",
    ):
        evaluate_promotion(
            champion_metrics,
            challenger_metrics,
            invalid_policy,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.1,
        "0.8",
        None,
        True,
        False,
        [],
    ],
)
@pytest.mark.parametrize(
    "policy_key",
    [
        "minimum_auprc",
        "minimum_recall",
        "maximum_brier_score",
        "minimum_auprc_improvement",
        "maximum_recall_drop",
    ],
)
def test_rejects_invalid_explicit_policy_values(
    policy_key,
    invalid_value,
    valid_policy,
    champion_metrics,
    challenger_metrics,
):
    invalid_policy = valid_policy.copy()
    invalid_policy[policy_key] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"Invalid policy setting: {policy_key}",
    ):
        evaluate_promotion(
            champion_metrics,
            challenger_metrics,
            invalid_policy,
        )


# ---------------------------------------------------------
# Tests for default policy loading
# ---------------------------------------------------------


def test_loads_default_policy_when_policy_is_none(
    monkeypatch,
    champion_metrics,
    challenger_metrics,
    valid_policy,
):
    import src.evaluation.promotion_policy as promotion_module

    monkeypatch.setattr(
        promotion_module,
        "load_evaluation_policy",
        lambda: valid_policy,
    )

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
    )

    assert result["promotion_eligible"] is True


def test_default_policy_is_used_for_eligibility(
    monkeypatch,
    champion_metrics,
    challenger_metrics,
    valid_policy,
):
    import src.evaluation.promotion_policy as promotion_module

    strict_policy = valid_policy.copy()
    strict_policy["minimum_auprc"] = 0.95

    monkeypatch.setattr(
        promotion_module,
        "load_evaluation_policy",
        lambda: strict_policy,
    )

    result = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
    )

    assert result["promotion_eligible"] is False
