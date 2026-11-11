
import numpy as np
import pytest

from src.evaluation.metrics import calculate_metrics


@pytest.fixture
def sample_data():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    threshold = 0.5

    return y_true, probabilities, threshold


def test_calculate_metrics_returns_expected_keys(sample_data):
    y_true, probabilities, threshold = sample_data

    result = calculate_metrics(
        y_true,
        probabilities,
        threshold,
    )

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
        "log_loss",
        "threshold",
    }

    assert expected_keys == set(result.keys())


def test_calculate_metrics_returns_numeric_values(sample_data):
    result = calculate_metrics(*sample_data)

    for value in result.values():
        assert isinstance(value, (int, float, np.number))
        assert np.isfinite(value)


def test_calculate_metrics_returns_valid_metric_ranges(sample_data):
    result = calculate_metrics(*sample_data)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    ]:
        assert 0 <= result[metric] <= 1


def test_calculate_metrics_returns_correct_threshold(sample_data):
    result = calculate_metrics(*sample_data)

    assert result["threshold"] == 0.5


def test_calculate_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]

    result = calculate_metrics(
        y_true,
        probabilities,
        0.5,
    )

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["brier_score"] < 0.1


def test_calculate_metrics_handles_no_positive_predictions():
    y_true = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.3, 0.4]

    result = calculate_metrics(
        y_true,
        probabilities,
        0.5,
    )

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0


def test_calculate_metrics_accepts_numpy_threshold():
    result = calculate_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        np.float64(0.5),
    )

    assert result["threshold"] == 0.5


def test_calculate_metrics_rejects_empty_inputs():
    with pytest.raises(ValueError):
        calculate_metrics([], [], 0.5)


def test_calculate_metrics_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        calculate_metrics(
            [0, 1, 0],
            [0.1, 0.9],
            0.5,
        )


def test_calculate_metrics_rejects_multidimensional_labels():
    with pytest.raises(ValueError):
        calculate_metrics(
            [[0], [1]],
            [0.1, 0.9],
            0.5,
        )


def test_calculate_metrics_rejects_multidimensional_probabilities():
    with pytest.raises(ValueError):
        calculate_metrics(
            [0, 1],
            [[0.1], [0.9]],
            0.5,
        )


def test_calculate_metrics_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        calculate_metrics(
            [0, 1, 2],
            [0.1, 0.8, 0.9],
            0.5,
        )


def test_calculate_metrics_rejects_single_class():
    with pytest.raises(ValueError):
        calculate_metrics(
            [0, 0, 0, 0],
            [0.1, 0.2, 0.3, 0.4],
            0.5,
        )


def test_calculate_metrics_rejects_missing_labels():
    with pytest.raises(ValueError):
        calculate_metrics(
            [0, np.nan, 1],
            [0.1, 0.5, 0.9],
            0.5,
        )


def test_calculate_metrics_rejects_missing_probabilities():
    with pytest.raises(ValueError):
        calculate_metrics(
            [0, 0, 1, 1],
            [0.1, np.nan, 0.8, 0.9],
            0.5,
        )


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1],
)
def test_calculate_metrics_rejects_out_of_range_probabilities(
    probability,
):
    with pytest.raises(ValueError):
        calculate_metrics(
            [0, 0, 1, 1],
            [0.1, probability, 0.8, 0.9],
            0.5,
        )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, None, "0.5", True, np.nan],
)
def test_calculate_metrics_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        calculate_metrics(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            threshold,
        )


def test_calculate_metrics_accepts_boundary_thresholds():
    for threshold in [0.0, 1.0]:
        result = calculate_metrics(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            threshold,
        )

        assert result["threshold"] == threshold
