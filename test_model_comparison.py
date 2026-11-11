
"""Unit tests for Champion and Challenger model comparison."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.compare import compare_models


class DummyModel:
    """Lightweight model for testing without training."""

    def __init__(self, predictions, probabilities):
        self.predictions = np.asarray(predictions)
        self.probabilities = np.asarray(probabilities)
        self.received_features = None

    def predict(self, X):
        self.received_features = X.copy()
        return self.predictions

    def predict_proba(self, X):
        return np.column_stack([
            1 - self.probabilities,
            self.probabilities,
        ])


class ModelWithoutProbability:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


@pytest.fixture
def test_data():
    return pd.DataFrame({
        "subject_id": [1, 2, 3, 4, 5, 6],
        "heart_rate": [70, 80, 75, 110, 120, 115],
        "temperature": [36.5, 36.7, 36.6, 38.5, 39.0, 38.8],
        "deterioration_event": [0, 0, 0, 1, 1, 1],
    })


@pytest.fixture
def models():
    champion = DummyModel(
        predictions=[0, 0, 0, 1, 1, 1],
        probabilities=[0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    )

    challenger = DummyModel(
        predictions=[0, 0, 1, 1, 1, 1],
        probabilities=[0.2, 0.3, 0.6, 0.8, 0.9, 0.95],
    )

    return champion, challenger


@pytest.fixture
def feature_columns():
    return ["heart_rate", "temperature"]


def test_returns_champion_and_challenger_results(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    assert set(result.keys()) == {"champion", "challenger"}


def test_returns_expected_metrics(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    expected_metrics = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    }

    assert set(result["champion"].keys()) == expected_metrics
    assert set(result["challenger"].keys()) == expected_metrics


def test_champion_model_name_is_correct(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    assert result["champion"]["model"] == "Logistic Regression"


def test_challenger_model_name_is_correct(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    assert result["challenger"]["model"] == "Random Forest"


def test_perfect_champion_predictions_have_perfect_metrics(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    metrics = result["champion"]

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0


def test_metrics_are_numeric(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    for role in ["champion", "challenger"]:
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auroc",
            "auprc",
            "brier_score",
        ]:
            assert isinstance(result[role][metric], float)


def test_metrics_are_within_valid_ranges(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    for role in ["champion", "challenger"]:
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auroc",
            "auprc",
            "brier_score",
        ]:
            assert 0.0 <= result[role][metric] <= 1.0


def test_brier_score_is_non_negative(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    assert result["champion"]["brier_score"] >= 0
    assert result["challenger"]["brier_score"] >= 0


def test_both_models_receive_same_test_features(
    test_data, models, feature_columns
):
    champion, challenger = models

    compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    pd.testing.assert_frame_equal(
        champion.received_features,
        challenger.received_features,
    )


def test_accepts_different_feature_order_when_sets_match(
    test_data, models
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        ["heart_rate", "temperature"],
        ["temperature", "heart_rate"],
    )

    assert "champion" in result
    assert "challenger" in result


def test_rejects_non_dataframe_test_data(models, feature_columns):
    champion, challenger = models

    with pytest.raises(TypeError, match="DataFrame"):
        compare_models(
            champion,
            challenger,
            [1, 2, 3],
            feature_columns,
            feature_columns,
        )


def test_rejects_empty_test_data(models, feature_columns):
    champion, challenger = models
    empty_data = pd.DataFrame(
        columns=[
            "heart_rate",
            "temperature",
            "deterioration_event",
        ]
    )

    with pytest.raises(ValueError, match="empty"):
        compare_models(
            champion,
            challenger,
            empty_data,
            feature_columns,
            feature_columns,
        )


def test_rejects_missing_target_column(
    models, feature_columns
):
    champion, challenger = models
    data = pd.DataFrame({
        "heart_rate": [70, 80],
        "temperature": [36.5, 37.0],
    })

    with pytest.raises(ValueError, match="Target column"):
        compare_models(
            champion,
            challenger,
            data,
            feature_columns,
            feature_columns,
        )


def test_rejects_missing_target_values(
    test_data, models, feature_columns
):
    champion, challenger = models
    data = test_data.copy()
    data.loc[0, "deterioration_event"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        compare_models(
            champion,
            challenger,
            data,
            feature_columns,
            feature_columns,
        )


def test_rejects_single_target_class(
    test_data, models, feature_columns
):
    champion, challenger = models
    data = test_data.copy()
    data["deterioration_event"] = 0

    with pytest.raises(ValueError, match="both target classes"):
        compare_models(
            champion,
            challenger,
            data,
            feature_columns,
            feature_columns,
        )


def test_rejects_non_binary_target_values(
    test_data, models, feature_columns
):
    champion, challenger = models
    data = test_data.copy()
    data["deterioration_event"] = [0, 2, 0, 2, 0, 2]

    with pytest.raises(ValueError, match="binary"):
        compare_models(
            champion,
            challenger,
            data,
            feature_columns,
            feature_columns,
        )


def test_rejects_different_feature_sets(test_data, models):
    champion, challenger = models

    with pytest.raises(ValueError, match="same feature columns"):
        compare_models(
            champion,
            challenger,
            test_data,
            ["heart_rate"],
            ["temperature"],
        )


def test_rejects_missing_feature_columns(test_data, models):
    champion, challenger = models

    with pytest.raises(ValueError, match="missing features"):
        compare_models(
            champion,
            challenger,
            test_data,
            ["heart_rate", "missing_feature"],
            ["heart_rate", "missing_feature"],
        )


def test_rejects_model_without_predict_proba(
    test_data, models, feature_columns
):
    champion, challenger = models
    invalid_model = ModelWithoutProbability()

    with pytest.raises(TypeError, match="predict_proba"):
        compare_models(
            invalid_model,
            challenger,
            test_data,
            feature_columns,
            feature_columns,
        )


def test_rejects_incorrect_prediction_count(
    test_data, models, feature_columns
):
    _, challenger = models

    invalid_champion = DummyModel(
        predictions=[0, 1],
        probabilities=[0.2, 0.8],
    )

    with pytest.raises(ValueError, match="incorrect number"):
        compare_models(
            invalid_champion,
            challenger,
            test_data,
            feature_columns,
            feature_columns,
        )


def test_returns_results_for_both_models_independently(
    test_data, models, feature_columns
):
    champion, challenger = models

    result = compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    assert result["champion"]["accuracy"] == 1.0
    assert result["challenger"]["accuracy"] < 1.0


def test_does_not_modify_original_test_data(
    test_data, models, feature_columns
):
    champion, challenger = models
    original_data = test_data.copy(deep=True)

    compare_models(
        champion,
        challenger,
        test_data,
        feature_columns,
        feature_columns,
    )

    pd.testing.assert_frame_equal(test_data, original_data)
