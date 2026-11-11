"""Unit tests for threshold-based classification evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.threshold_evaluation import (
    evaluate_at_threshold,
    evaluate_multiple_thresholds,
)


@pytest.fixture
def sample_data():
    """Provide sample binary classification data."""

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.7, 0.4])

    return y_true, y_prob


# ---------------------------------------------------------
# Tests for evaluate_at_threshold
# ---------------------------------------------------------


def test_returns_dictionary(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert isinstance(result, dict)


def test_returns_all_expected_keys(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    expected_keys = {
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1_score",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "total_records",
    }

    assert set(result.keys()) == expected_keys


def test_default_threshold_is_point_five(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["threshold"] == 0.5


def test_calculates_accuracy_correctly(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["accuracy"] == pytest.approx(0.5)


def test_calculates_precision_correctly(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["precision"] == pytest.approx(0.5)


def test_calculates_recall_correctly(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["recall"] == pytest.approx(0.5)


def test_sensitivity_equals_recall(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["sensitivity"] == result["recall"]


def test_calculates_specificity_correctly(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["specificity"] == pytest.approx(0.5)


def test_calculates_f1_score_correctly(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["f1_score"] == pytest.approx(0.5)


def test_calculates_confusion_matrix_counts(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_negatives"] == 1


def test_calculates_total_records(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["total_records"] == 4


def test_uses_greater_than_or_equal_threshold():
    y_true = np.array([0, 1])
    y_prob = np.array([0.5, 0.5])

    result = evaluate_at_threshold(
        y_true,
        y_prob,
        threshold=0.5,
    )

    assert result["false_positives"] == 1
    assert result["true_positives"] == 1


def test_threshold_zero_predicts_all_positive():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    result = evaluate_at_threshold(
        y_true,
        y_prob,
        threshold=0,
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 2
    assert result["true_negatives"] == 0
    assert result["false_negatives"] == 0


def test_threshold_one_predicts_only_probability_one_as_positive():
    y_true = np.array([0, 1, 1])
    y_prob = np.array([0.2, 1.0, 0.8])

    result = evaluate_at_threshold(
        y_true,
        y_prob,
        threshold=1,
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["true_negatives"] == 1
    assert result["false_negatives"] == 1


def test_accepts_python_lists():
    result = evaluate_at_threshold(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )

    assert isinstance(result, dict)
    assert result["accuracy"] == pytest.approx(1.0)


def test_accepts_pandas_series():
    y_true = pd.Series([0, 0, 1, 1])
    y_prob = pd.Series([0.1, 0.2, 0.8, 0.9])

    result = evaluate_at_threshold(y_true, y_prob)

    assert result["accuracy"] == pytest.approx(1.0)


def test_returns_float_metric_values(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    metric_keys = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1_score",
    ]

    for key in metric_keys:
        assert isinstance(result[key], float)


def test_returns_integer_confusion_counts(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_at_threshold(y_true, y_prob)

    count_keys = [
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "total_records",
    ]

    for key in count_keys:
        assert isinstance(result[key], int)


def test_handles_all_negative_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    result = evaluate_at_threshold(
        y_true,
        y_prob,
        threshold=0.9,
    )

    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["true_negatives"] == 2
    assert result["false_negatives"] == 2
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0


def test_handles_all_positive_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.8, 0.9, 0.7, 0.6])

    result = evaluate_at_threshold(
        y_true,
        y_prob,
        threshold=0.5,
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 2
    assert result["true_negatives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------
# Input validation tests for evaluate_at_threshold
# ---------------------------------------------------------


def test_raises_error_for_empty_y_true():
    with pytest.raises(ValueError, match="y_true cannot be empty"):
        evaluate_at_threshold([], [])


def test_raises_error_for_unequal_lengths():
    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        evaluate_at_threshold(
            [0, 1, 0],
            [0.2, 0.8],
        )


@pytest.mark.parametrize(
    "y_true",
    [
        [0, 1, 2],
        [0, -1, 1],
        [0, 0.5, 1],
    ],
)
def test_raises_error_for_non_binary_labels(y_true):
    with pytest.raises(
        ValueError,
        match="binary labels",
    ):
        evaluate_at_threshold(
            y_true,
            [0.1, 0.5, 0.9],
        )


@pytest.mark.parametrize(
    "y_prob",
    [
        [0.1, np.nan],
        [0.1, np.inf],
        [0.1, -np.inf],
    ],
)
def test_raises_error_for_non_finite_probabilities(y_prob):
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        evaluate_at_threshold(
            [0, 1],
            y_prob,
        )


@pytest.mark.parametrize(
    "y_prob",
    [
        [-0.1, 0.5],
        [0.5, 1.1],
    ],
)
def test_raises_error_for_probabilities_outside_range(y_prob):
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluate_at_threshold(
            [0, 1],
            y_prob,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
        np.nan,
    ],
)
def test_raises_error_for_invalid_threshold(threshold):
    with pytest.raises(
        ValueError,
        match="Threshold must be between 0 and 1",
    ):
        evaluate_at_threshold(
            [0, 1],
            [0.2, 0.8],
            threshold=threshold,
        )


# ---------------------------------------------------------
# Tests for evaluate_multiple_thresholds
# ---------------------------------------------------------


def test_multiple_thresholds_returns_dataframe(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_multiple_thresholds(y_true, y_prob)

    assert isinstance(result, pd.DataFrame)


def test_multiple_thresholds_uses_default_values(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_multiple_thresholds(y_true, y_prob)

    assert result["threshold"].tolist() == [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ]


def test_multiple_thresholds_returns_five_rows(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_multiple_thresholds(y_true, y_prob)

    assert len(result) == 5


def test_multiple_thresholds_accepts_custom_values(sample_data):
    y_true, y_prob = sample_data

    thresholds = [0.2, 0.5, 0.8]

    result = evaluate_multiple_thresholds(
        y_true,
        y_prob,
        thresholds=thresholds,
    )

    assert result["threshold"].tolist() == thresholds


def test_multiple_thresholds_returns_expected_columns(sample_data):
    y_true, y_prob = sample_data

    result = evaluate_multiple_thresholds(y_true, y_prob)

    expected_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1_score",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "total_records",
    ]

    assert result.columns.tolist() == expected_columns


def test_multiple_thresholds_calculates_each_threshold(sample_data):
    y_true, y_prob = sample_data

    thresholds = [0.3, 0.5, 0.7]

    result = evaluate_multiple_thresholds(
        y_true,
        y_prob,
        thresholds=thresholds,
    )

    for index, threshold in enumerate(thresholds):
        expected = evaluate_at_threshold(
            y_true,
            y_prob,
            threshold=threshold,
        )

        assert result.iloc[index]["accuracy"] == pytest.approx(
            expected["accuracy"]
        )


def test_multiple_thresholds_accepts_numpy_array(sample_data):
    y_true, y_prob = sample_data

    thresholds = np.array([0.2, 0.5, 0.8])

    result = evaluate_multiple_thresholds(
        y_true,
        y_prob,
        thresholds=thresholds,
    )

    assert result["threshold"].tolist() == [0.2, 0.5, 0.8]


def test_multiple_thresholds_accepts_generator(sample_data):
    y_true, y_prob = sample_data

    thresholds = (value for value in [0.2, 0.5, 0.8])

    result = evaluate_multiple_thresholds(
        y_true,
        y_prob,
        thresholds=thresholds,
    )

    assert result["threshold"].tolist() == [0.2, 0.5, 0.8]


def test_multiple_thresholds_with_empty_list_returns_empty_dataframe(
    sample_data,
):
    y_true, y_prob = sample_data

    result = evaluate_multiple_thresholds(
        y_true,
        y_prob,
        thresholds=[],
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_multiple_thresholds_rejects_invalid_threshold(sample_data):
    y_true, y_prob = sample_data

    with pytest.raises(
        ValueError,
        match="Threshold must be between 0 and 1",
    ):
        evaluate_multiple_thresholds(
            y_true,
            y_prob,
            thresholds=[0.3, 1.5],
        )


def test_multiple_thresholds_preserves_threshold_order(sample_data):
    y_true, y_prob = sample_data

    thresholds = [0.8, 0.3, 0.6]

    result = evaluate_multiple_thresholds(
        y_true,
        y_prob,
        thresholds=thresholds,
    )

    assert result["threshold"].tolist() == thresholds