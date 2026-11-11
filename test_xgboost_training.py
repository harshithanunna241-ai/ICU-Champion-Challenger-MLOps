
import numpy as np
import pandas as pd
import pytest

from src.models.train_xgboost import (
    train_xgboost_model,
    evaluate_xgboost_model,
)


@pytest.fixture
def training_data():
    return pd.DataFrame({
        "subject_id": range(1, 21),
        "stay_id": range(101, 121),
        "prediction_time": pd.date_range(
            "2026-01-01", periods=20, freq="h"
        ),
        "heart_rate": [
            65, 70, 72, 75, 80,
            85, 90, 95, 100, 105,
            110, 115, 120, 125, 130,
            135, 140, 145, 150, 155,
        ],
        "spo2": [
            99, 98, 98, 97, 97,
            96, 96, 95, 95, 94,
            94, 93, 93, 92, 92,
            91, 91, 90, 90, 89,
        ],
        "temperature": [
            36.5, 36.6, 36.7, 36.8, 36.9,
            37.0, 37.1, 37.2, 37.3, 37.4,
            37.5, 37.6, 37.7, 37.8, 37.9,
            38.0, 38.1, 38.2, 38.3, 38.4,
        ],
        "deterioration_event": [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ],
    })


@pytest.fixture
def test_data(training_data):
    return training_data.iloc[[0, 2, 4, 6, 11, 13, 15, 19]].copy()


@pytest.fixture
def trained_model(training_data):
    return train_xgboost_model(training_data)


def test_training_returns_model_and_feature_columns(training_data):
    model, feature_columns = train_xgboost_model(training_data)

    assert model is not None
    assert isinstance(feature_columns, list)
    assert len(feature_columns) > 0


def test_training_returns_pipeline(training_data):
    from sklearn.pipeline import Pipeline

    model, _ = train_xgboost_model(training_data)

    assert isinstance(model, Pipeline)


def test_training_excludes_identifier_columns(training_data):
    _, feature_columns = train_xgboost_model(training_data)

    assert "subject_id" not in feature_columns
    assert "stay_id" not in feature_columns
    assert "prediction_time" not in feature_columns


def test_training_excludes_target_column(training_data):
    _, feature_columns = train_xgboost_model(training_data)

    assert "deterioration_event" not in feature_columns


def test_training_uses_numeric_features_only(training_data):
    training_data["patient_name"] = [
        f"Patient_{i}" for i in range(len(training_data))
    ]

    _, feature_columns = train_xgboost_model(training_data)

    assert "patient_name" not in feature_columns
    assert "heart_rate" in feature_columns


def test_training_handles_missing_feature_values(training_data):
    training_data.loc[0, "heart_rate"] = np.nan
    training_data.loc[1, "spo2"] = np.nan

    model, feature_columns = train_xgboost_model(training_data)

    assert model is not None
    assert "heart_rate" in feature_columns
    assert "spo2" in feature_columns


def test_training_rejects_non_dataframe():
    with pytest.raises((TypeError, ValueError)):
        train_xgboost_model([1, 2, 3])


def test_training_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        train_xgboost_model(pd.DataFrame())


def test_training_rejects_missing_target(training_data):
    training_data = training_data.drop(
        columns=["deterioration_event"]
    )

    with pytest.raises(ValueError):
        train_xgboost_model(training_data)


def test_training_rejects_missing_target_values(training_data):
    training_data.loc[0, "deterioration_event"] = np.nan

    with pytest.raises(ValueError):
        train_xgboost_model(training_data)


def test_training_rejects_single_class(training_data):
    training_data["deterioration_event"] = 0

    with pytest.raises(ValueError):
        train_xgboost_model(training_data)


def test_training_rejects_non_binary_target(training_data):
    training_data.loc[0, "deterioration_event"] = 2

    with pytest.raises(ValueError):
        train_xgboost_model(training_data)


def test_training_rejects_no_numeric_features(training_data):
    training_data = training_data.drop(
        columns=[
            "heart_rate",
            "spo2",
            "temperature",
        ]
    )

    training_data["patient_name"] = [
        f"Patient_{i}" for i in range(len(training_data))
    ]

    with pytest.raises(ValueError):
        train_xgboost_model(training_data)


def test_training_ignores_all_missing_features(training_data):
    training_data["all_missing"] = np.nan

    _, feature_columns = train_xgboost_model(training_data)

    assert "all_missing" not in feature_columns


def test_evaluation_returns_expected_metrics(
    trained_model,
    test_data,
):
    model, feature_columns = trained_model

    metrics = evaluate_xgboost_model(
        model,
        test_data,
        feature_columns,
    )

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    }

    assert expected_metrics.issubset(metrics.keys())


def test_evaluation_metrics_are_numeric(
    trained_model,
    test_data,
):
    model, feature_columns = trained_model

    metrics = evaluate_xgboost_model(
        model,
        test_data,
        feature_columns,
    )

    for metric in metrics.values():
        assert isinstance(metric, (int, float, np.number))
        assert np.isfinite(metric)


def test_evaluation_metrics_are_within_valid_ranges(
    trained_model,
    test_data,
):
    model, feature_columns = trained_model

    metrics = evaluate_xgboost_model(
        model,
        test_data,
        feature_columns,
    )

    for metric_name, value in metrics.items():
        assert 0 <= value <= 1, (
            f"{metric_name} is outside the expected range: {value}"
        )


def test_evaluation_rejects_non_dataframe(
    trained_model,
):
    model, feature_columns = trained_model

    with pytest.raises((TypeError, ValueError)):
        evaluate_xgboost_model(
            model,
            [1, 2, 3],
            feature_columns,
        )


def test_evaluation_rejects_empty_dataframe(
    trained_model,
):
    model, feature_columns = trained_model

    with pytest.raises(ValueError):
        evaluate_xgboost_model(
            model,
            pd.DataFrame(),
            feature_columns,
        )


def test_evaluation_rejects_missing_target(
    trained_model,
    test_data,
):
    model, feature_columns = trained_model

    test_data = test_data.drop(
        columns=["deterioration_event"]
    )

    with pytest.raises(ValueError):
        evaluate_xgboost_model(
            model,
            test_data,
            feature_columns,
        )


def test_evaluation_rejects_missing_feature(
    trained_model,
    test_data,
):
    model, feature_columns = trained_model

    test_data = test_data.drop(
        columns=[feature_columns[0]]
    )

    with pytest.raises(ValueError):
        evaluate_xgboost_model(
            model,
            test_data,
            feature_columns,
        )


def test_evaluation_rejects_missing_target_values(
    trained_model,
    test_data,
):
    model, feature_columns = trained_model

    test_data.loc[
        test_data.index[0],
        "deterioration_event",
    ] = np.nan

    with pytest.raises(ValueError):
        evaluate_xgboost_model(
            model,
            test_data,
            feature_columns,
        )


def test_evaluation_rejects_single_class(
    trained_model,
    test_data,
):
    model, feature_columns = trained_model

    test_data["deterioration_event"] = 0

    with pytest.raises(ValueError):
        evaluate_xgboost_model(
            model,
            test_data,
            feature_columns,
        )
