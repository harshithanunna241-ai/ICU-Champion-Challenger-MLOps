
"""Unit tests for integrated calibrated evaluation of three models."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_three_models_calibrated as evaluation


# --------------------------------------------------
# Fake model
# --------------------------------------------------

class FakeModel:
    """Lightweight model used for unit testing."""

    def predict_proba(self, X):
        probabilities = np.full(len(X), 0.6)

        return np.column_stack((
            1 - probabilities,
            probabilities,
        ))

    def predict(self, X):
        return np.ones(len(X), dtype=int)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def train_data():
    return pd.DataFrame({
        "heart_rate": [70, 80, 75, 110, 120, 115, 72, 118, 85, 105],
        "temperature": [
            36.5, 36.7, 36.6, 38.5, 39.0,
            38.8, 36.4, 38.9, 37.0, 38.2,
        ],
        "deterioration_event": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def threshold_data():
    return pd.DataFrame({
        "heart_rate": [68, 82, 76, 112, 125, 118],
        "temperature": [36.4, 36.8, 36.6, 38.4, 39.1, 38.7],
        "deterioration_event": [0, 0, 0, 1, 1, 1],
    })


@pytest.fixture
def evaluation_data():
    return pd.DataFrame({
        "heart_rate": [69, 81, 74, 111, 123, 116],
        "temperature": [36.5, 36.7, 36.5, 38.6, 39.0, 38.8],
        "deterioration_event": [0, 0, 0, 1, 1, 1],
    })


@pytest.fixture
def fake_models():
    return {
        "logistic": FakeModel(),
        "random_forest": FakeModel(),
        "xgboost": FakeModel(),
    }


@pytest.fixture
def configure_mocks(monkeypatch, fake_models):
    """Configure training, calibration and evaluation dependencies."""

    monkeypatch.setattr(
        evaluation,
        "train_logistic_model",
        Mock(return_value=(
            fake_models["logistic"],
            ["heart_rate", "temperature"],
        )),
    )

    monkeypatch.setattr(
        evaluation,
        "train_random_forest_model",
        Mock(return_value=(
            fake_models["random_forest"],
            ["heart_rate", "temperature"],
        )),
    )

    monkeypatch.setattr(
        evaluation,
        "train_xgboost_model",
        Mock(return_value=(
            fake_models["xgboost"],
            ["heart_rate", "temperature"],
        )),
    )

    monkeypatch.setattr(
        evaluation,
        "calibrate_model",
        Mock(side_effect=lambda model, **kwargs: model),
    )

    monkeypatch.setattr(
        evaluation,
        "select_classification_threshold",
        Mock(return_value={
            "threshold": 0.45,
            "precision": 0.9,
            "recall": 0.8,
            "f1_score": 0.85,
            "objective": "f1",
        }),
    )

    monkeypatch.setattr(
        evaluation,
        "evaluate_calibration",
        Mock(return_value={
            "brier_score": 0.12,
            "log_loss": 0.35,
            "total_records": 6,
        }),
    )

    monkeypatch.setattr(
        evaluation,
        "evaluate_at_threshold",
        Mock(return_value={
            "threshold": 0.45,
            "accuracy": 0.9,
            "precision": 0.9,
            "recall": 0.8,
            "specificity": 1.0,
            "f1_score": 0.85,
        }),
    )

    return evaluation


# --------------------------------------------------
# Integrated evaluation tests
# --------------------------------------------------

def test_returns_dataframe(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    result = evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    assert isinstance(result, pd.DataFrame)


def test_evaluates_all_three_models(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    result = evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    assert len(result) == 3

    assert result["model"].tolist() == [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]


def test_returns_expected_result_columns(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    result = evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    expected_columns = {
        "model",
        "brier_score",
        "log_loss",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    }

    assert set(result.columns) == expected_columns


def test_trains_all_three_models(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    evaluation.train_logistic_model.assert_called_once_with(train_data)
    evaluation.train_random_forest_model.assert_called_once_with(train_data)
    evaluation.train_xgboost_model.assert_called_once_with(train_data)


def test_calibrates_each_model(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    assert evaluation.calibrate_model.call_count == 3


def test_calibration_uses_training_data(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    for call in evaluation.calibrate_model.call_args_list:
        assert call.kwargs["train_data"] is train_data


def test_uses_requested_calibration_method_and_cv(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
        calibration_method="sigmoid",
        cv=3,
    )

    for call in evaluation.calibrate_model.call_args_list:
        assert call.kwargs["method"] == "sigmoid"
        assert call.kwargs["cv"] == 3


def test_selects_threshold_for_each_model(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    assert evaluation.select_classification_threshold.call_count == 3


def test_threshold_selection_uses_f1_objective(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    for call in evaluation.select_classification_threshold.call_args_list:
        assert call.kwargs["objective"] == "f1"


def test_evaluates_calibration_for_each_model(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    assert evaluation.evaluate_calibration.call_count == 3


def test_calibration_evaluation_uses_independent_evaluation_data(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    for call in evaluation.evaluate_calibration.call_args_list:
        assert call.kwargs["validation_data"] is evaluation_data


def test_evaluates_threshold_metrics_for_each_model(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    assert evaluation.evaluate_at_threshold.call_count == 3


def test_passes_selected_threshold_to_evaluation(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    for call in evaluation.evaluate_at_threshold.call_args_list:
        assert call.kwargs["threshold"] == 0.45


def test_includes_calibration_metrics_in_results(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    result = evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    assert (result["brier_score"] == 0.12).all()
    assert (result["log_loss"] == 0.35).all()


def test_accepts_numeric_threshold_result(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation.select_classification_threshold.return_value = 0.6

    result = evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
    )

    assert (result["threshold"] == 0.6).all()


def test_accepts_custom_target_column(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    train_data = train_data.rename(
        columns={"deterioration_event": "target"}
    )

    threshold_data = threshold_data.rename(
        columns={"deterioration_event": "target"}
    )

    evaluation_data = evaluation_data.rename(
        columns={"deterioration_event": "target"}
    )

    result = evaluation.evaluate_three_models_calibrated(
        train_data,
        threshold_data,
        evaluation_data,
        target_column="target",
    )

    assert len(result) == 3


# --------------------------------------------------
# Input validation tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "argument",
    ["train_data", "threshold_data", "evaluation_data"],
)
def test_rejects_non_dataframe_inputs(
    train_data,
    threshold_data,
    evaluation_data,
    argument,
):
    inputs = {
        "train_data": train_data,
        "threshold_data": threshold_data,
        "evaluation_data": evaluation_data,
    }

    inputs[argument] = [1, 2, 3]

    with pytest.raises(TypeError, match="DataFrame"):
        evaluation.evaluate_three_models_calibrated(
            inputs["train_data"],
            inputs["threshold_data"],
            inputs["evaluation_data"],
        )


@pytest.mark.parametrize(
    "argument",
    ["train_data", "threshold_data", "evaluation_data"],
)
def test_rejects_empty_inputs(
    train_data,
    threshold_data,
    evaluation_data,
    argument,
):
    inputs = {
        "train_data": train_data,
        "threshold_data": threshold_data,
        "evaluation_data": evaluation_data,
    }

    inputs[argument] = inputs[argument].iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        evaluation.evaluate_three_models_calibrated(
            inputs["train_data"],
            inputs["threshold_data"],
            inputs["evaluation_data"],
        )


@pytest.mark.parametrize(
    "argument",
    ["train_data", "threshold_data", "evaluation_data"],
)
def test_rejects_missing_target_column(
    train_data,
    threshold_data,
    evaluation_data,
    argument,
):
    inputs = {
        "train_data": train_data,
        "threshold_data": threshold_data,
        "evaluation_data": evaluation_data,
    }

    inputs[argument] = inputs[argument].drop(
        columns=["deterioration_event"]
    )

    with pytest.raises(ValueError, match="must contain"):
        evaluation.evaluate_three_models_calibrated(
            inputs["train_data"],
            inputs["threshold_data"],
            inputs["evaluation_data"],
        )


@pytest.mark.parametrize(
    "argument",
    ["train_data", "threshold_data", "evaluation_data"],
)
def test_rejects_single_target_class(
    train_data,
    threshold_data,
    evaluation_data,
    argument,
):
    inputs = {
        "train_data": train_data,
        "threshold_data": threshold_data,
        "evaluation_data": evaluation_data,
    }

    inputs[argument] = inputs[argument].copy()
    inputs[argument]["deterioration_event"] = 0

    with pytest.raises(ValueError, match="both binary target classes"):
        evaluation.evaluate_three_models_calibrated(
            inputs["train_data"],
            inputs["threshold_data"],
            inputs["evaluation_data"],
        )


def test_rejects_missing_threshold_features(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    threshold_data = threshold_data.drop(columns=["heart_rate"])

    with pytest.raises(ValueError, match="threshold_data is missing"):
        evaluation.evaluate_three_models_calibrated(
            train_data,
            threshold_data,
            evaluation_data,
        )


def test_rejects_missing_evaluation_features(
    train_data,
    threshold_data,
    evaluation_data,
    configure_mocks,
):
    evaluation_data = evaluation_data.drop(columns=["temperature"])

    with pytest.raises(ValueError, match="evaluation_data is missing"):
        evaluation.evaluate_three_models_calibrated(
            train_data,
            threshold_data,
            evaluation_data,
        )


# --------------------------------------------------
# Calibrated evaluation report tests
# --------------------------------------------------

@pytest.fixture
def comparison_data():
    return pd.DataFrame({
        "model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        ],
        "threshold": [0.5, 0.4, 0.6],
        "brier_score": [0.15, 0.12, 0.10],
        "log_loss": [0.40, 0.35, 0.30],
        "accuracy": [0.80, 0.85, 0.90],
        "precision": [0.75, 0.80, 0.90],
        "recall": [0.70, 0.85, 0.88],
        "specificity": [0.90, 0.85, 0.92],
        "f1_score": [0.72, 0.82, 0.89],
        "auroc": [0.85, 0.90, 0.95],
    })


def test_report_returns_dataframe(comparison_data):
    result = evaluation.create_calibrated_evaluation_report(
        comparison_data
    )

    assert isinstance(result, pd.DataFrame)


def test_report_contains_only_required_columns(comparison_data):
    result = evaluation.create_calibrated_evaluation_report(
        comparison_data
    )

    expected_columns = [
        "model",
        "threshold",
        "brier_score",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
    ]

    assert result.columns.tolist() == expected_columns


def test_report_sorts_by_f1_descending(comparison_data):
    result = evaluation.create_calibrated_evaluation_report(
        comparison_data
    )

    scores = result["f1_score"].tolist()

    assert scores == sorted(scores, reverse=True)


def test_report_places_best_model_first(comparison_data):
    result = evaluation.create_calibrated_evaluation_report(
        comparison_data
    )

    assert result.iloc[0]["model"] == "XGBoost"


def test_report_resets_index(comparison_data):
    result = evaluation.create_calibrated_evaluation_report(
        comparison_data
    )

    assert result.index.tolist() == [0, 1, 2]


def test_report_does_not_modify_original_dataframe(comparison_data):
    original = comparison_data.copy(deep=True)

    evaluation.create_calibrated_evaluation_report(
        comparison_data
    )

    pd.testing.assert_frame_equal(comparison_data, original)


def test_report_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        evaluation.create_calibrated_evaluation_report(
            [1, 2, 3]
        )


def test_report_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        evaluation.create_calibrated_evaluation_report(
            pd.DataFrame()
        )


def test_report_rejects_missing_required_columns():
    data = pd.DataFrame({
        "model": ["Logistic Regression"],
        "accuracy": [0.9],
    })

    with pytest.raises(ValueError, match="missing columns"):
        evaluation.create_calibrated_evaluation_report(data)
