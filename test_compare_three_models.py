
"""Unit tests for three-model comparison."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.compare_three_models import (
    evaluate_single_model,
    compare_three_models,
)


class DummyModel:
    """Lightweight model for testing."""

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


@pytest.fixture
def test_data():
    return pd.DataFrame({
        "subject_id": [1, 2, 3, 4, 5, 6],
        "heart_rate": [70, 80, 75, 110, 120, 115],
        "temperature": [36.5, 36.7, 36.6, 38.5, 39.0, 38.8],
        "deterioration_event": [0, 0, 0, 1, 1, 1],
    })


@pytest.fixture
def feature_columns():
    return ["heart_rate", "temperature"]


@pytest.fixture
def models():
    logistic = DummyModel(
        [0, 0, 0, 1, 1, 1],
        [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    )

    random_forest = DummyModel(
        [0, 0, 1, 1, 1, 1],
        [0.2, 0.3, 0.6, 0.8, 0.9, 0.95],
    )

    xgboost = DummyModel(
        [0, 1, 0, 1, 1, 1],
        [0.15, 0.55, 0.25, 0.75, 0.85, 0.95],
    )

    return logistic, random_forest, xgboost


# --------------------------------------------------
# evaluate_single_model tests
# --------------------------------------------------

def test_evaluate_single_model_returns_expected_metrics(
    test_data, models, feature_columns
):
    logistic, _, _ = models

    result = evaluate_single_model(
        logistic,
        test_data,
        feature_columns,
        "Logistic Regression",
    )

    expected_keys = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    }

    assert set(result.keys()) == expected_keys


def test_evaluate_single_model_returns_correct_name(
    test_data, models, feature_columns
):
    logistic, _, _ = models

    result = evaluate_single_model(
        logistic,
        test_data,
        feature_columns,
        "Logistic Regression",
    )

    assert result["model"] == "Logistic Regression"


def test_evaluate_single_model_perfect_predictions(
    test_data, models, feature_columns
):
    logistic, _, _ = models

    result = evaluate_single_model(
        logistic,
        test_data,
        feature_columns,
        "Logistic Regression",
    )

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["brier_score"] >= 0.0


def test_evaluate_single_model_metrics_are_floats(
    test_data, models, feature_columns
):
    logistic, _, _ = models

    result = evaluate_single_model(
        logistic,
        test_data,
        feature_columns,
        "Logistic Regression",
    )

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    ]:
        assert isinstance(result[key], float)


def test_evaluate_single_model_metrics_are_in_valid_ranges(
    test_data, models, feature_columns
):
    logistic, _, _ = models

    result = evaluate_single_model(
        logistic,
        test_data,
        feature_columns,
        "Logistic Regression",
    )

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    ]:
        assert 0.0 <= result[key] <= 1.0


def test_evaluate_single_model_uses_requested_features(
    test_data, models
):
    logistic, _, _ = models

    evaluate_single_model(
        logistic,
        test_data,
        ["heart_rate"],
        "Logistic Regression",
    )

    assert list(logistic.received_features.columns) == ["heart_rate"]


def test_evaluate_single_model_rejects_non_dataframe(
    models, feature_columns
):
    logistic, _, _ = models

    with pytest.raises(TypeError, match="DataFrame"):
        evaluate_single_model(
            logistic,
            [1, 2, 3],
            feature_columns,
            "Logistic Regression",
        )


def test_evaluate_single_model_rejects_empty_dataframe(
    models, feature_columns
):
    logistic, _, _ = models

    empty_data = pd.DataFrame(
        columns=feature_columns + ["deterioration_event"]
    )

    with pytest.raises(ValueError, match="empty"):
        evaluate_single_model(
            logistic,
            empty_data,
            feature_columns,
            "Logistic Regression",
        )


def test_evaluate_single_model_rejects_missing_target(
    models, feature_columns
):
    logistic, _, _ = models

    data = pd.DataFrame({
        "heart_rate": [70, 80],
        "temperature": [36.5, 37.0],
    })

    with pytest.raises(ValueError, match="Missing target"):
        evaluate_single_model(
            logistic,
            data,
            feature_columns,
            "Logistic Regression",
        )


def test_evaluate_single_model_rejects_missing_target_values(
    test_data, models, feature_columns
):
    logistic, _, _ = models

    data = test_data.copy()
    data.loc[0, "deterioration_event"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        evaluate_single_model(
            logistic,
            data,
            feature_columns,
            "Logistic Regression",
        )


def test_evaluate_single_model_rejects_missing_features(
    test_data, models
):
    logistic, _, _ = models

    with pytest.raises(ValueError, match="Missing features"):
        evaluate_single_model(
            logistic,
            test_data,
            ["missing_feature"],
            "Logistic Regression",
        )


def test_evaluate_single_model_rejects_non_binary_target(
    test_data, models, feature_columns
):
    logistic, _, _ = models

    data = test_data.copy()
    data["deterioration_event"] = [0, 2, 0, 2, 0, 2]

    with pytest.raises(ValueError, match="only 0 and 1"):
        evaluate_single_model(
            logistic,
            data,
            feature_columns,
            "Logistic Regression",
        )


def test_evaluate_single_model_rejects_single_class(
    test_data, models, feature_columns
):
    logistic, _, _ = models

    data = test_data.copy()
    data["deterioration_event"] = 0

    with pytest.raises(ValueError, match="both target classes"):
        evaluate_single_model(
            logistic,
            data,
            feature_columns,
            "Logistic Regression",
        )


# --------------------------------------------------
# compare_three_models tests
# --------------------------------------------------

def test_comparison_returns_expected_top_level_keys(
    test_data, models, feature_columns
):
    logistic, random_forest, xgboost = models

    result = compare_three_models(
        logistic,
        random_forest,
        xgboost,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
    )

    assert set(result.keys()) == {
        "comparison",
        "ranking",
        "best_model_by_auprc",
    }


def test_comparison_contains_three_models(
    test_data, models, feature_columns
):
    logistic, random_forest, xgboost = models

    result = compare_three_models(
        logistic,
        random_forest,
        xgboost,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
    )

    assert len(result["comparison"]) == 3


def test_comparison_contains_all_model_names(
    test_data, models, feature_columns
):
    logistic, random_forest, xgboost = models

    result = compare_three_models(
        logistic,
        random_forest,
        xgboost,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
    )

    names = result["comparison"]["model"].tolist()

    assert names == [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]


def test_comparison_contains_expected_metric_columns(
    test_data, models, feature_columns
):
    logistic, random_forest, xgboost = models

    result = compare_three_models(
        logistic,
        random_forest,
        xgboost,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
    )

    expected_columns = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    }

    assert set(result["comparison"].columns) == expected_columns


def test_ranking_is_descending_by_auprc(
    test_data, models, feature_columns
):
    logistic, random_forest, xgboost = models

    result = compare_three_models(
        logistic,
        random_forest,
        xgboost,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
    )

    scores = result["ranking"]["auprc"].tolist()

    assert scores == sorted(scores, reverse=True)


def test_best_model_matches_first_ranked_model(
    test_data, models, feature_columns
):
    logistic, random_forest, xgboost = models

    result = compare_three_models(
        logistic,
        random_forest,
        xgboost,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
    )

    assert (
        result["best_model_by_auprc"]
        == result["ranking"].iloc[0]["model"]
    )


def test_comparison_preserves_each_models_features(
    test_data, models
):
    logistic, random_forest, xgboost = models

    compare_three_models(
        logistic,
        random_forest,
        xgboost,
        test_data,
        ["heart_rate"],
        ["temperature"],
        ["heart_rate", "temperature"],
    )

    assert list(logistic.received_features.columns) == ["heart_rate"]
    assert list(random_forest.received_features.columns) == ["temperature"]
    assert list(xgboost.received_features.columns) == [
        "heart_rate",
        "temperature",
    ]


def test_comparison_does_not_modify_original_data(
    test_data, models, feature_columns
):
    logistic, random_forest, xgboost = models
    original_data = test_data.copy(deep=True)

    compare_three_models(
        logistic,
        random_forest,
        xgboost,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
    )

    pd.testing.assert_frame_equal(test_data, original_data)


def test_comparison_rejects_invalid_test_data(
    models, feature_columns
):
    logistic, random_forest, xgboost = models

    with pytest.raises(TypeError, match="DataFrame"):
        compare_three_models(
            logistic,
            random_forest,
            xgboost,
            [1, 2, 3],
            feature_columns,
            feature_columns,
            feature_columns,
        )


def test_comparison_rejects_empty_test_data(
    models, feature_columns
):
    logistic, random_forest, xgboost = models

    empty_data = pd.DataFrame(
        columns=feature_columns + ["deterioration_event"]
    )

    with pytest.raises(ValueError, match="empty"):
        compare_three_models(
            logistic,
            random_forest,
            xgboost,
            empty_data,
            feature_columns,
            feature_columns,
            feature_columns,
        )


def test_comparison_rejects_missing_target(
    models, feature_columns
):
    logistic, random_forest, xgboost = models

    data = pd.DataFrame({
        "heart_rate": [70, 80],
        "temperature": [36.5, 37.0],
    })

    with pytest.raises(ValueError, match="Missing target"):
        compare_three_models(
            logistic,
            random_forest,
            xgboost,
            data,
            feature_columns,
            feature_columns,
            feature_columns,
        )


def test_comparison_rejects_single_target_class(
    test_data, models, feature_columns
):
    logistic, random_forest, xgboost = models

    data = test_data.copy()
    data["deterioration_event"] = 0

    with pytest.raises(ValueError, match="both target classes"):
        compare_three_models(
            logistic,
            random_forest,
            xgboost,
            data,
            feature_columns,
            feature_columns,
            feature_columns,
        )
