
import pandas as pd
import pytest

from src.evaluation.three_model_promotion import (
    evaluate_three_model_promotion,
)


@pytest.fixture
def policy():
    return {
        "minimum_auprc": 0.75,
        "minimum_recall": 0.60,
        "maximum_brier_score": 0.25,
        "minimum_auprc_improvement": 0.05,
        "maximum_recall_drop": 0.05,
    }


@pytest.fixture
def model_metrics():
    return {
        "Logistic Regression": {
            "auprc": 0.80,
            "recall": 0.70,
            "brier_score": 0.20,
        },
        "Random Forest": {
            "auprc": 0.87,
            "recall": 0.68,
            "brier_score": 0.18,
        },
        "XGBoost": {
            "auprc": 0.90,
            "recall": 0.70,
            "brier_score": 0.20,
        },
    }


def test_selects_highest_auprc_eligible_challenger(
    model_metrics, policy
):
    result = evaluate_three_model_promotion(
        model_metrics,
        policy=policy,
    )

    assert result["selected_challenger"] == "XGBoost"
    assert result["promotion_eligible"] is True


def test_eligible_challenger_requires_human_approval(
    model_metrics, policy
):
    result = evaluate_three_model_promotion(
        model_metrics,
        policy=policy,
    )

    assert result["promotion_approved"] is False
    assert result["human_approval_required"] is True
    assert result["human_approved"] is False
    assert result["decision"] == (
        "ELIGIBLE - HUMAN APPROVAL REQUIRED"
    )


def test_decision_table_marks_eligible_models(
    model_metrics, policy
):
    result = evaluate_three_model_promotion(
        model_metrics,
        policy=policy,
    )

    decisions = result["decisions"]

    assert isinstance(decisions, pd.DataFrame)
    assert decisions["eligible"].all()
    assert decisions["human_approval_required"].all()
    assert not decisions["human_approved"].any()
    assert (
        decisions["decision"]
        == "ELIGIBLE - HUMAN APPROVAL REQUIRED"
    ).all()


def test_rejects_challengers_that_fail_policy(
    model_metrics, policy
):
    model_metrics["Random Forest"]["auprc"] = 0.82
    model_metrics["Random Forest"]["recall"] = 0.50
    model_metrics["XGBoost"]["brier_score"] = 0.30

    result = evaluate_three_model_promotion(
        model_metrics,
        policy=policy,
    )

    decisions = result["decisions"].set_index("model")

    assert not decisions.loc["Random Forest", "eligible"]
    assert not decisions.loc["XGBoost", "eligible"]
    assert result["selected_challenger"] is None
    assert result["promotion_eligible"] is False
    assert result["promotion_approved"] is False
    assert result["human_approval_required"] is False
    assert result["decision"] == "KEEP CURRENT CHAMPION"


def test_records_all_failed_checks(policy):
    metrics = {
        "Logistic Regression": {
            "auprc": 0.80,
            "recall": 0.70,
            "brier_score": 0.20,
        },
        "Random Forest": {
            "auprc": 0.60,
            "recall": 0.40,
            "brier_score": 0.40,
        },
    }

    result = evaluate_three_model_promotion(
        metrics,
        policy=policy,
    )

    decisions = result["decisions"]

    assert decisions.iloc[0]["failed_checks"] == [
        "minimum_auprc",
        "minimum_recall",
        "maximum_brier_score",
        "minimum_auprc_improvement",
        "maximum_recall_drop",
    ]


def test_uses_custom_current_champion(policy):
    metrics = {
        "Random Forest": {
            "auprc": 0.80,
            "recall": 0.70,
            "brier_score": 0.20,
        },
        "XGBoost": {
            "auprc": 0.90,
            "recall": 0.70,
            "brier_score": 0.20,
        },
    }

    result = evaluate_three_model_promotion(
        metrics,
        current_champion="Random Forest",
        policy=policy,
    )

    assert result["current_champion"] == "Random Forest"
    assert result["selected_challenger"] == "XGBoost"


def test_rejects_non_dictionary_model_metrics(policy):
    with pytest.raises(TypeError):
        evaluate_three_model_promotion(
            [],
            policy=policy,
        )


def test_rejects_missing_champion(policy):
    metrics = {
        "Random Forest": {
            "auprc": 0.90,
            "recall": 0.70,
            "brier_score": 0.20,
        },
        "XGBoost": {
            "auprc": 0.88,
            "recall": 0.70,
            "brier_score": 0.20,
        },
    }

    with pytest.raises(ValueError, match="Current champion"):
        evaluate_three_model_promotion(
            metrics,
            policy=policy,
        )


def test_rejects_only_one_model(policy):
    metrics = {
        "Logistic Regression": {
            "auprc": 0.80,
            "recall": 0.70,
            "brier_score": 0.20,
        },
    }

    with pytest.raises(ValueError, match="At least two models"):
        evaluate_three_model_promotion(
            metrics,
            policy=policy,
        )


def test_rejects_missing_required_metric(policy):
    metrics = {
        "Logistic Regression": {
            "auprc": 0.80,
            "recall": 0.70,
            "brier_score": 0.20,
        },
        "Random Forest": {
            "auprc": 0.90,
            "recall": 0.70,
        },
    }

    with pytest.raises(ValueError, match="brier_score"):
        evaluate_three_model_promotion(
            metrics,
            policy=policy,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [True, -0.1, 1.1, "0.8", None],
)
def test_rejects_invalid_metric_values(policy, invalid_value):
    metrics = {
        "Logistic Regression": {
            "auprc": 0.80,
            "recall": 0.70,
            "brier_score": 0.20,
        },
        "Random Forest": {
            "auprc": invalid_value,
            "recall": 0.70,
            "brier_score": 0.20,
        },
    }

    with pytest.raises(ValueError):
        evaluate_three_model_promotion(
            metrics,
            policy=policy,
        )


def test_rejects_missing_policy_key(model_metrics):
    invalid_policy = {
        "minimum_auprc": 0.75,
        "minimum_recall": 0.60,
    }

    with pytest.raises(ValueError, match="Policy is missing"):
        evaluate_three_model_promotion(
            model_metrics,
            policy=invalid_policy,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [True, -0.1, 1.1, "0.8", None],
)
def test_rejects_invalid_policy_values(
    model_metrics, policy, invalid_value
):
    policy["minimum_auprc"] = invalid_value

    with pytest.raises(ValueError):
        evaluate_three_model_promotion(
            model_metrics,
            policy=policy,
        )
