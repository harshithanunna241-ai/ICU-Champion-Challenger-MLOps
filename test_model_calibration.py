
import numpy as np
import pandas as pd
import pytest

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.calibration import (
    calibrate_model,
    select_classification_threshold,
    evaluate_calibration,
)


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    n = 60

    heart_rate = rng.normal(90, 20, n)
    spo2 = rng.normal(95, 3, n)

    target = (
        (heart_rate > 90) & (spo2 < 95)
    ).astype(int)

    # Ensure both classes have sufficient samples.
    target[:20] = 0
    target[20:40] = 1

    return pd.DataFrame({
        "subject_id": range(1, n + 1),
        "stay_id": range(101, n + 101),
        "prediction_time": pd.date_range(
            "2026-01-01",
            periods=n,
            freq="h",
        ),
        "heart_rate": heart_rate,
        "spo2": spo2,
        "deterioration_event": target,
    })


@pytest.fixture
def feature_columns():
    return ["heart_rate", "spo2"]


@pytest.fixture
def base_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])


@pytest.fixture
def calibrated_model(
    base_model,
    training_data,
    feature_columns,
):
    return calibrate_model(
        model=base_model,
        train_data=training_data,
        feature_columns=feature_columns,
        method="sigmoid",
        cv=3,
    )


@pytest.fixture
def validation_data(training_data):
    return training_data.copy()


# --------------------------------------------------
# Calibration training tests
# --------------------------------------------------

def test_calibrate_model_returns_calibrated_estimator(
    calibrated_model,
):
    assert calibrated_model is not None
    assert hasattr(calibrated_model, "predict_proba")


def test_calibrate_model_predicts_probabilities(
    calibrated_model,
    training_data,
    feature_columns,
):
    probabilities = calibrated_model.predict_proba(
        training_data[feature_columns]
    )

    assert probabilities.shape == (
        len(training_data),
        2,
    )


def test_calibrate_model_probabilities_are_valid(
    calibrated_model,
    training_data,
    feature_columns,
):
    probabilities = calibrated_model.predict_proba(
        training_data[feature_columns]
    )

    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_calibrate_model_rejects_non_dataframe(
    base_model,
    feature_columns,
):
    with pytest.raises((TypeError, ValueError)):
        calibrate_model(
            base_model,
            [1, 2, 3],
            feature_columns,
        )


def test_calibrate_model_rejects_empty_dataframe(
    base_model,
    feature_columns,
):
    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            pd.DataFrame(),
            feature_columns,
        )


def test_calibrate_model_rejects_missing_target(
    base_model,
    training_data,
    feature_columns,
):
    data = training_data.drop(
        columns=["deterioration_event"]
    )

    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            data,
            feature_columns,
        )


def test_calibrate_model_rejects_missing_features(
    base_model,
    training_data,
):
    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            training_data,
            ["heart_rate", "missing_feature"],
        )


def test_calibrate_model_rejects_empty_feature_list(
    base_model,
    training_data,
):
    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            training_data,
            [],
        )


def test_calibrate_model_rejects_missing_target_values(
    base_model,
    training_data,
    feature_columns,
):
    data = training_data.copy()
    data.loc[0, "deterioration_event"] = np.nan

    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            data,
            feature_columns,
        )


def test_calibrate_model_rejects_invalid_method(
    base_model,
    training_data,
    feature_columns,
):
    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            training_data,
            feature_columns,
            method="invalid",
        )


def test_calibrate_model_rejects_invalid_cv(
    base_model,
    training_data,
    feature_columns,
):
    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            training_data,
            feature_columns,
            cv=1,
        )


def test_calibrate_model_rejects_boolean_cv(
    base_model,
    training_data,
    feature_columns,
):
    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            training_data,
            feature_columns,
            cv=True,
        )


def test_calibrate_model_rejects_single_class(
    base_model,
    training_data,
    feature_columns,
):
    data = training_data.copy()
    data["deterioration_event"] = 0

    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            data,
            feature_columns,
        )


def test_calibrate_model_rejects_insufficient_class_samples(
    base_model,
    training_data,
    feature_columns,
):
    data = training_data.copy()

    data["deterioration_event"] = 0
    data.loc[data.index[0], "deterioration_event"] = 1

    with pytest.raises(ValueError):
        calibrate_model(
            base_model,
            data,
            feature_columns,
            cv=5,
        )


# --------------------------------------------------
# Classification threshold tests
# --------------------------------------------------

def test_select_threshold_returns_expected_keys():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

    result = select_classification_threshold(
        y_true,
        y_prob,
    )

    expected_keys = {
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "objective",
    }

    assert expected_keys.issubset(result.keys())


def test_select_threshold_returns_valid_values():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

    result = select_classification_threshold(
        y_true,
        y_prob,
    )

    assert 0 <= result["threshold"] <= 1
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["f1_score"] <= 1
    assert result["objective"] == "f1"


def test_select_threshold_rejects_different_array_lengths():
    with pytest.raises(ValueError):
        select_classification_threshold(
            [0, 1, 0],
            [0.2, 0.8],
        )


def test_select_threshold_rejects_empty_arrays():
    with pytest.raises(ValueError):
        select_classification_threshold(
            [],
            [],
        )


def test_select_threshold_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        select_classification_threshold(
            [0, 1, 2],
            [0.1, 0.8, 0.9],
        )


def test_select_threshold_rejects_single_class():
    with pytest.raises(ValueError):
        select_classification_threshold(
            [0, 0, 0, 0],
            [0.1, 0.2, 0.3, 0.4],
        )


def test_select_threshold_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        select_classification_threshold(
            [0, 0, 1, 1],
            [0.1, 0.4, 1.2, 0.9],
        )


def test_select_threshold_rejects_nan_probabilities():
    with pytest.raises(ValueError):
        select_classification_threshold(
            [0, 0, 1, 1],
            [0.1, 0.4, np.nan, 0.9],
        )


def test_select_threshold_rejects_invalid_objective():
    with pytest.raises(ValueError):
        select_classification_threshold(
            [0, 0, 1, 1],
            [0.1, 0.3, 0.7, 0.9],
            objective="accuracy",
        )


# --------------------------------------------------
# Calibration evaluation tests
# --------------------------------------------------

def test_evaluate_calibration_returns_expected_metrics(
    calibrated_model,
    validation_data,
    feature_columns,
):
    result = evaluate_calibration(
        calibrated_model,
        validation_data,
        feature_columns,
    )

    assert "brier_score" in result
    assert "log_loss" in result
    assert "total_records" in result


def test_evaluate_calibration_returns_valid_metrics(
    calibrated_model,
    validation_data,
    feature_columns,
):
    result = evaluate_calibration(
        calibrated_model,
        validation_data,
        feature_columns,
    )

    assert 0 <= result["brier_score"] <= 1
    assert result["log_loss"] >= 0
    assert result["total_records"] == len(validation_data)


def test_evaluate_calibration_rejects_non_dataframe(
    calibrated_model,
    feature_columns,
):
    with pytest.raises((TypeError, ValueError)):
        evaluate_calibration(
            calibrated_model,
            [1, 2, 3],
            feature_columns,
        )


def test_evaluate_calibration_rejects_empty_data(
    calibrated_model,
    feature_columns,
):
    with pytest.raises(ValueError):
        evaluate_calibration(
            calibrated_model,
            pd.DataFrame(),
            feature_columns,
        )


def test_evaluate_calibration_rejects_missing_features(
    calibrated_model,
    validation_data,
):
    data = validation_data.drop(
        columns=["heart_rate"]
    )

    with pytest.raises(ValueError):
        evaluate_calibration(
            calibrated_model,
            data,
            ["heart_rate", "spo2"],
        )


def test_evaluate_calibration_rejects_missing_target(
    calibrated_model,
    validation_data,
    feature_columns,
):
    data = validation_data.drop(
        columns=["deterioration_event"]
    )

    with pytest.raises(ValueError):
        evaluate_calibration(
            calibrated_model,
            data,
            feature_columns,
        )


def test_evaluate_calibration_rejects_single_class(
    calibrated_model,
    validation_data,
    feature_columns,
):
    data = validation_data.copy()
    data["deterioration_event"] = 0

    with pytest.raises(ValueError):
        evaluate_calibration(
            calibrated_model,
            data,
            feature_columns,
        )


def test_evaluate_calibration_rejects_missing_target_values(
    calibrated_model,
    validation_data,
    feature_columns,
):
    data = validation_data.copy()
    data.loc[data.index[0], "deterioration_event"] = np.nan

    with pytest.raises(ValueError):
        evaluate_calibration(
            calibrated_model,
            data,
            feature_columns,
        )
