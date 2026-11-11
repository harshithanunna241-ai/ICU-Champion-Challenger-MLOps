
import numpy as np
import pytest

from src.evaluation.confidence_intervals import (
    calculate_confidence_intervals,
)


@pytest.fixture
def sample_data():
    y_true = np.array([0] * 50 + [1] * 50)

    y_prob = np.concatenate([
        np.linspace(0.05, 0.45, 50),
        np.linspace(0.55, 0.95, 50),
    ])

    return y_true, y_prob


def test_returns_expected_top_level_keys(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true, y_prob, n_bootstraps=200
    )

    assert set(result.keys()) == {
        "confidence_level",
        "threshold",
        "n_bootstraps",
        "metrics",
    }


def test_returns_all_five_metrics(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true, y_prob, n_bootstraps=200
    )

    assert set(result["metrics"].keys()) == {
        "auroc",
        "auprc",
        "precision",
        "recall",
        "f1_score",
    }


def test_each_metric_contains_expected_fields(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true, y_prob, n_bootstraps=200
    )

    for metric in result["metrics"].values():
        assert set(metric.keys()) == {
            "estimate",
            "lower_bound",
            "upper_bound",
            "valid_bootstrap_samples",
        }


def test_confidence_intervals_have_valid_bounds(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true, y_prob, n_bootstraps=200
    )

    for metric in result["metrics"].values():
        assert 0 <= metric["lower_bound"] <= 1
        assert 0 <= metric["upper_bound"] <= 1
        assert metric["lower_bound"] <= metric["upper_bound"]


def test_estimates_are_inside_confidence_intervals(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true, y_prob, n_bootstraps=200
    )

    for metric in result["metrics"].values():
        assert (
            metric["lower_bound"]
            <= metric["estimate"]
            <= metric["upper_bound"]
        )


def test_perfect_predictions_have_perfect_discrimination():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([
        np.linspace(0.01, 0.40, 50),
        np.linspace(0.60, 0.99, 50),
    ])

    result = calculate_confidence_intervals(
        y_true,
        y_prob,
        n_bootstraps=200,
    )

    assert result["metrics"]["auroc"]["estimate"] == 1.0
    assert result["metrics"]["auprc"]["estimate"] == 1.0
    assert result["metrics"]["precision"]["estimate"] == 1.0
    assert result["metrics"]["recall"]["estimate"] == 1.0
    assert result["metrics"]["f1_score"]["estimate"] == 1.0


def test_returns_requested_confidence_level(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true,
        y_prob,
        n_bootstraps=200,
        confidence_level=0.90,
    )

    assert result["confidence_level"] == 0.90


def test_returns_requested_threshold(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true,
        y_prob,
        threshold=0.6,
        n_bootstraps=200,
    )

    assert result["threshold"] == 0.6


def test_returns_requested_bootstrap_count(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true,
        y_prob,
        n_bootstraps=200,
    )

    assert result["n_bootstraps"] == 200


def test_reproducible_with_same_random_state(sample_data):
    y_true, y_prob = sample_data

    result1 = calculate_confidence_intervals(
        y_true,
        y_prob,
        n_bootstraps=200,
        random_state=42,
    )

    result2 = calculate_confidence_intervals(
        y_true,
        y_prob,
        n_bootstraps=200,
        random_state=42,
    )

    assert result1 == result2


def test_different_random_states_can_change_intervals(sample_data):
    y_true, y_prob = sample_data

    result1 = calculate_confidence_intervals(
        y_true,
        y_prob,
        n_bootstraps=200,
        random_state=42,
    )

    result2 = calculate_confidence_intervals(
        y_true,
        y_prob,
        n_bootstraps=200,
        random_state=99,
    )

    assert result1["metrics"] != result2["metrics"]


def test_valid_bootstrap_samples_are_at_least_100(sample_data):
    y_true, y_prob = sample_data

    result = calculate_confidence_intervals(
        y_true,
        y_prob,
        n_bootstraps=200,
    )

    for metric in result["metrics"].values():
        assert metric["valid_bootstrap_samples"] >= 100


def test_rejects_multidimensional_labels(sample_data):
    y_true, y_prob = sample_data

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true.reshape(-1, 1),
            y_prob,
            n_bootstraps=200,
        )


def test_rejects_multidimensional_probabilities(sample_data):
    y_true, y_prob = sample_data

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true,
            y_prob.reshape(-1, 1),
            n_bootstraps=200,
        )


def test_rejects_unequal_array_lengths():
    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            [0, 1, 0],
            [0.1, 0.9],
            n_bootstraps=200,
        )


def test_rejects_empty_arrays():
    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            [],
            [],
            n_bootstraps=200,
        )


def test_rejects_missing_labels(sample_data):
    y_true, y_prob = sample_data
    y_true = y_true.astype(float)
    y_true[0] = np.nan

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true,
            y_prob,
            n_bootstraps=200,
        )


def test_rejects_missing_probabilities(sample_data):
    y_true, y_prob = sample_data
    y_prob = y_prob.copy()
    y_prob[0] = np.nan

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true,
            y_prob,
            n_bootstraps=200,
        )


def test_rejects_infinite_probabilities(sample_data):
    y_true, y_prob = sample_data
    y_prob = y_prob.copy()
    y_prob[0] = np.inf

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true,
            y_prob,
            n_bootstraps=200,
        )


@pytest.mark.parametrize("invalid_probability", [-0.1, 1.1])
def test_rejects_out_of_range_probabilities(
    sample_data,
    invalid_probability,
):
    y_true, y_prob = sample_data
    y_prob = y_prob.copy()
    y_prob[0] = invalid_probability

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true,
            y_prob,
            n_bootstraps=200,
        )


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            [0, 1, 2, 0],
            [0.1, 0.8, 0.9, 0.2],
            n_bootstraps=200,
        )


def test_rejects_single_class_labels():
    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            [0, 0, 0, 0],
            [0.1, 0.2, 0.3, 0.4],
            n_bootstraps=200,
        )


@pytest.mark.parametrize("threshold", [0, 1, -0.1, 1.1])
def test_rejects_invalid_threshold(sample_data, threshold):
    y_true, y_prob = sample_data

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true,
            y_prob,
            threshold=threshold,
            n_bootstraps=200,
        )


@pytest.mark.parametrize("n_bootstraps", [0, 50, 99])
def test_rejects_insufficient_bootstrap_count(
    sample_data,
    n_bootstraps,
):
    y_true, y_prob = sample_data

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true,
            y_prob,
            n_bootstraps=n_bootstraps,
        )


@pytest.mark.parametrize(
    "confidence_level",
    [0, 1, -0.1, 1.1],
)
def test_rejects_invalid_confidence_level(
    sample_data,
    confidence_level,
):
    y_true, y_prob = sample_data

    with pytest.raises(ValueError):
        calculate_confidence_intervals(
            y_true,
            y_prob,
            n_bootstraps=200,
            confidence_level=confidence_level,
        )


def test_rejects_when_too_few_valid_bootstrap_samples_remain():
    # A highly imbalanced sample makes one-class bootstrap
    # samples overwhelmingly likely.
    y_true = np.array([0] * 999 + [1])
    y_prob = np.linspace(0.1, 0.9, 1000)

    with pytest.raises(ValueError, match="Too few valid"):
        calculate_confidence_intervals(
            y_true,
            y_prob,
            n_bootstraps=100,
            random_state=42,
        )
