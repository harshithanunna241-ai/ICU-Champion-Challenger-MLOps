
import pandas as pd
import pytest

from src.models.train_random_forest import (
    train_random_forest_model,
    evaluate_random_forest_model,
)


@pytest.fixture
def training_data():
    """Synthetic ICU training records for testing only."""

    return pd.DataFrame({
        "subject_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "stay_id": [201, 202, 203, 204, 205, 206, 207, 208],
        "prediction_time": pd.date_range(
            "2026-01-01", periods=8, freq="h"
        ),
        "heart_rate": [70, 75, 80, 85, 110, 115, 120, 125],
        "systolic_bp": [120, 118, 115, 110, 90, 85, 80, 75],
        "temperature": [36.5, 36.6, 36.7, 36.8, 38.0, 38.2, 38.4, 38.6],
        "deterioration_event": [0, 0, 0, 0, 1, 1, 1, 1],
    })


@pytest.fixture
def test_data():
    """Synthetic ICU testing records."""

    return pd.DataFrame({
        "subject_id": [301, 302, 303, 304],
        "stay_id": [401, 402, 403, 404],
        "prediction_time": pd.date_range(
            "2026-02-01", periods=4, freq="h"
        ),
        "heart_rate": [72, 82, 112, 122],
        "systolic_bp": [119, 114, 88, 78],
        "temperature": [36.6, 36.8, 38.1, 38.5],
        "deterioration_event": [0, 0, 1, 1],
    })


def test_train_random_forest_returns_model_and_features(training_data):
    model, feature_columns = train_random_forest_model(training_data)

    assert model is not None
    assert isinstance(feature_columns, list)
    assert len(feature_columns) > 0


def test_training_excludes_identifier_and_target_columns(training_data):
    _, feature_columns = train_random_forest_model(training_data)

    assert "subject_id" not in feature_columns
    assert "stay_id" not in feature_columns
    assert "prediction_time" not in feature_columns
    assert "deterioration_event" not in feature_columns


def test_training_uses_numeric_features_only(training_data):
    training_data["category"] = ["A", "B"] * 4

    _, feature_columns = train_random_forest_model(training_data)

    assert "category" not in feature_columns
    assert "heart_rate" in feature_columns


def test_training_handles_missing_feature_values(training_data):
    training_data.loc[0, "heart_rate"] = float("nan")

    model, feature_columns = train_random_forest_model(training_data)

    assert model is not None
    assert "heart_rate" in feature_columns


def test_training_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="DataFrame"):
        train_random_forest_model([1, 2, 3])


def test_training_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        train_random_forest_model(pd.DataFrame())


def test_training_rejects_missing_target_column(training_data):
    training_data = training_data.drop(
        columns=["deterioration_event"]
    )

    with pytest.raises(ValueError, match="Target column"):
        train_random_forest_model(training_data)


def test_training_rejects_single_target_class(training_data):
    training_data["deterioration_event"] = 0

    with pytest.raises(ValueError, match="both target classes"):
        train_random_forest_model(training_data)


def test_training_rejects_no_usable_numeric_features(training_data):
    training_data = training_data.drop(
        columns=["heart_rate", "systolic_bp", "temperature"]
    )
    training_data["category"] = ["A", "B"] * 4

    with pytest.raises(ValueError, match="No usable numerical features"):
        train_random_forest_model(training_data)


def test_training_rejects_all_missing_features(training_data):
    training_data["heart_rate"] = float("nan")
    training_data["systolic_bp"] = float("nan")
    training_data["temperature"] = float("nan")

    with pytest.raises(ValueError, match="No usable numerical features"):
        train_random_forest_model(training_data)


def test_evaluate_random_forest_returns_all_metrics(
    training_data,
    test_data,
):
    model, feature_columns = train_random_forest_model(training_data)

    metrics = evaluate_random_forest_model(
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

    assert set(metrics.keys()) == expected_metrics

    for value in metrics.values():
        assert 0 <= value <= 1


def test_evaluation_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="DataFrame"):
        evaluate_random_forest_model(
            None,
            [1, 2, 3],
            ["heart_rate"],
        )


def test_evaluation_rejects_empty_test_data(training_data):
    model, feature_columns = train_random_forest_model(training_data)

    with pytest.raises(ValueError, match="empty"):
        evaluate_random_forest_model(
            model,
            pd.DataFrame(),
            feature_columns,
        )


def test_evaluation_rejects_missing_test_columns(
    training_data,
    test_data,
):
    model, feature_columns = train_random_forest_model(training_data)

    test_data = test_data.drop(columns=["heart_rate"])

    with pytest.raises(ValueError, match="missing columns"):
        evaluate_random_forest_model(
            model,
            test_data,
            feature_columns,
        )


def test_evaluation_rejects_single_target_class(
    training_data,
    test_data,
):
    model, feature_columns = train_random_forest_model(training_data)

    test_data["deterioration_event"] = 0

    with pytest.raises(ValueError, match="both target classes"):
        evaluate_random_forest_model(
            model,
            test_data,
            feature_columns,
        )
