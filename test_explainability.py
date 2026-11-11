"""Unit tests for SHAP-based model explainability."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import explainability


class FakeModel:
    """Simple model for testing explainability."""

    def predict_proba(self, data):
        probabilities = np.asarray(data.iloc[:, 0], dtype=float)
        probabilities = 1 / (1 + np.exp(-probabilities))

        return np.column_stack([
            1 - probabilities,
            probabilities,
        ])


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "heart_rate": [70.0, 80.0, 90.0, 100.0],
        "temperature": [36.5, 37.0, 37.5, 38.0],
    })


@pytest.fixture
def sample_shap_values():
    return np.array([
        [0.10, -0.20],
        [0.30, 0.10],
        [-0.40, 0.20],
        [0.20, -0.10],
    ])


@pytest.fixture
def sample_explanation(sample_data, sample_shap_values):
    return {
        "explainer": None,
        "shap_values": sample_shap_values,
        "feature_data": sample_data,
        "feature_columns": ["heart_rate", "temperature"],
    }


# ---------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------


def test_validate_inputs_accepts_valid_data(sample_data):
    result = explainability._validate_inputs(
        FakeModel(),
        sample_data,
        ["heart_rate", "temperature"],
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "heart_rate",
        "temperature",
    ]


def test_validate_inputs_rejects_none_model(sample_data):
    with pytest.raises(ValueError, match="Model cannot be None"):
        explainability._validate_inputs(
            None,
            sample_data,
            ["heart_rate"],
        )


def test_validate_inputs_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        explainability._validate_inputs(
            FakeModel(),
            [[70, 36.5]],
            ["heart_rate"],
        )


def test_validate_inputs_rejects_empty_dataframe():
    data = pd.DataFrame(columns=["heart_rate"])

    with pytest.raises(ValueError, match="cannot be empty"):
        explainability._validate_inputs(
            FakeModel(),
            data,
            ["heart_rate"],
        )


def test_validate_inputs_rejects_empty_feature_columns(sample_data):
    with pytest.raises(ValueError, match="feature_columns cannot be empty"):
        explainability._validate_inputs(
            FakeModel(),
            sample_data,
            [],
        )


def test_validate_inputs_rejects_missing_features(sample_data):
    with pytest.raises(ValueError, match="Missing feature columns"):
        explainability._validate_inputs(
            FakeModel(),
            sample_data,
            ["missing_feature"],
        )


def test_validate_inputs_rejects_non_numeric_features():
    data = pd.DataFrame({
        "heart_rate": [70, 80],
        "category": ["low", "high"],
    })

    with pytest.raises(ValueError, match="must be numeric"):
        explainability._validate_inputs(
            FakeModel(),
            data,
            ["category"],
        )


def test_validate_inputs_rejects_nan_values():
    data = pd.DataFrame({
        "heart_rate": [70.0, np.nan],
    })

    with pytest.raises(ValueError, match="finite numeric values"):
        explainability._validate_inputs(
            FakeModel(),
            data,
            ["heart_rate"],
        )


def test_validate_inputs_rejects_infinite_values():
    data = pd.DataFrame({
        "heart_rate": [70.0, np.inf],
    })

    with pytest.raises(ValueError, match="finite numeric values"):
        explainability._validate_inputs(
            FakeModel(),
            data,
            ["heart_rate"],
        )


# ---------------------------------------------------------
# SHAP output extraction tests
# ---------------------------------------------------------


def test_extract_positive_class_from_two_dimensional_array(
    sample_shap_values,
):
    result = explainability._extract_positive_class_values(
        sample_shap_values,
        2,
    )

    np.testing.assert_array_equal(
        result,
        sample_shap_values,
    )


def test_extract_positive_class_from_list():
    class_values = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([[0.5, 0.6], [0.7, 0.8]]),
    ]

    result = explainability._extract_positive_class_values(
        class_values,
        2,
    )

    np.testing.assert_array_equal(
        result,
        class_values[1],
    )


def test_extract_first_class_from_single_item_list():
    class_values = [
        np.array([[0.1, 0.2], [0.3, 0.4]])
    ]

    result = explainability._extract_positive_class_values(
        class_values,
        2,
    )

    np.testing.assert_array_equal(
        result,
        class_values[0],
    )


def test_extract_positive_class_from_three_dimensional_array():
    values = np.array([
        [[0.1, 0.5], [0.2, 0.6]],
        [[0.3, 0.7], [0.4, 0.8]],
    ])

    result = explainability._extract_positive_class_values(
        values,
        2,
    )

    np.testing.assert_array_equal(
        result,
        values[:, :, 1],
    )


def test_extract_first_class_when_three_dimensional_output_has_one_class():
    values = np.array([
        [[0.1], [0.2]],
        [[0.3], [0.4]],
    ])

    result = explainability._extract_positive_class_values(
        values,
        2,
    )

    np.testing.assert_array_equal(
        result,
        values[:, :, 0],
    )


def test_extract_rejects_unexpected_dimensions():
    values = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="Unexpected SHAP output shape"):
        explainability._extract_positive_class_values(
            values,
            3,
        )


def test_extract_rejects_wrong_feature_count():
    values = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ])

    with pytest.raises(
        ValueError,
        match="feature count does not match",
    ):
        explainability._extract_positive_class_values(
            values,
            2,
        )


# ---------------------------------------------------------
# Model explanation tests
# ---------------------------------------------------------


def test_explain_model_returns_expected_structure(
    monkeypatch,
    sample_data,
):
    expected_values = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ])

    class FakeExplainer:
        def __init__(self, prediction_function, background):
            self.prediction_function = prediction_function
            self.background = background

        def shap_values(self, data, nsamples):
            assert nsamples == "auto"
            return expected_values

    monkeypatch.setattr(
        explainability.shap,
        "KernelExplainer",
        FakeExplainer,
    )

    result = explainability.explain_model(
        FakeModel(),
        sample_data,
        ["heart_rate", "temperature"],
    )

    assert set(result.keys()) == {
        "explainer",
        "shap_values",
        "feature_data",
        "feature_columns",
    }

    np.testing.assert_array_equal(
        result["shap_values"],
        expected_values,
    )

    assert list(result["feature_columns"]) == [
        "heart_rate",
        "temperature",
    ]

    assert isinstance(result["feature_data"], pd.DataFrame)


def test_explain_model_limits_background_samples(
    monkeypatch,
    sample_data,
):
    captured = {}

    class FakeExplainer:
        def __init__(self, prediction_function, background):
            captured["background"] = background

        def shap_values(self, data, nsamples):
            return np.zeros((len(data), data.shape[1]))

    monkeypatch.setattr(
        explainability.shap,
        "KernelExplainer",
        FakeExplainer,
    )

    explainability.explain_model(
        FakeModel(),
        sample_data,
        ["heart_rate", "temperature"],
        max_background_samples=2,
    )

    assert captured["background"].shape == (2, 2)


def test_explain_model_rejects_zero_background_samples(sample_data):
    with pytest.raises(
        ValueError,
        match="max_background_samples must be positive",
    ):
        explainability.explain_model(
            FakeModel(),
            sample_data,
            ["heart_rate"],
            max_background_samples=0,
        )


def test_explain_model_rejects_negative_background_samples(
    sample_data,
):
    with pytest.raises(
        ValueError,
        match="max_background_samples must be positive",
    ):
        explainability.explain_model(
            FakeModel(),
            sample_data,
            ["heart_rate"],
            max_background_samples=-1,
        )


# ---------------------------------------------------------
# Global feature importance tests
# ---------------------------------------------------------


def test_calculate_global_feature_importance_returns_dataframe(
    sample_explanation,
):
    result = explainability.calculate_global_feature_importance(
        sample_explanation
    )

    assert isinstance(result, pd.DataFrame)

    assert list(result.columns) == [
        "feature",
        "mean_absolute_shap",
    ]

    assert len(result) == 2


def test_global_feature_importance_is_sorted_descending(
    sample_explanation,
):
    result = explainability.calculate_global_feature_importance(
        sample_explanation
    )

    values = result["mean_absolute_shap"].to_numpy()

    assert np.all(values[:-1] >= values[1:])


def test_global_feature_importance_calculates_mean_absolute_values(
    sample_explanation,
):
    result = explainability.calculate_global_feature_importance(
        sample_explanation
    )

    importance = dict(
        zip(
            result["feature"],
            result["mean_absolute_shap"],
        )
    )

    assert importance["heart_rate"] == pytest.approx(0.25)
    assert importance["temperature"] == pytest.approx(0.15)


def test_global_feature_importance_rejects_non_dictionary():
    with pytest.raises(TypeError, match="dictionary"):
        explainability.calculate_global_feature_importance(
            []
        )


def test_global_feature_importance_rejects_wrong_dimensions(
    sample_explanation,
):
    sample_explanation["shap_values"] = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="two-dimensional"):
        explainability.calculate_global_feature_importance(
            sample_explanation
        )


# ---------------------------------------------------------
# Individual prediction explanation tests
# ---------------------------------------------------------


def test_explain_individual_prediction_returns_dataframe(
    sample_explanation,
):
    result = explainability.explain_individual_prediction(
        sample_explanation,
        row_index=0,
    )

    assert isinstance(result, pd.DataFrame)

    assert list(result.columns) == [
        "feature",
        "feature_value",
        "shap_value",
        "absolute_shap",
    ]

    assert len(result) == 2


def test_individual_prediction_is_sorted_by_absolute_shap(
    sample_explanation,
):
    result = explainability.explain_individual_prediction(
        sample_explanation,
        row_index=0,
    )

    values = result["absolute_shap"].to_numpy()

    assert np.all(values[:-1] >= values[1:])


def test_individual_prediction_uses_requested_row(
    sample_explanation,
):
    result = explainability.explain_individual_prediction(
        sample_explanation,
        row_index=1,
    )

    feature_values = dict(
        zip(
            result["feature"],
            result["feature_value"],
        )
    )

    assert feature_values["heart_rate"] == 80.0
    assert feature_values["temperature"] == 37.0


def test_individual_prediction_rejects_negative_row_index(
    sample_explanation,
):
    with pytest.raises(IndexError, match="outside the available data"):
        explainability.explain_individual_prediction(
            sample_explanation,
            row_index=-1,
        )


def test_individual_prediction_rejects_out_of_range_row_index(
    sample_explanation,
):
    with pytest.raises(IndexError, match="outside the available data"):
        explainability.explain_individual_prediction(
            sample_explanation,
            row_index=10,
        )


# ---------------------------------------------------------
# Model feature importance comparison tests
# ---------------------------------------------------------


def test_compare_model_feature_importance_returns_dataframe(
    sample_explanation,
):
    explanations = {
        "Logistic Regression": sample_explanation,
        "Random Forest": sample_explanation,
    }

    result = explainability.compare_model_feature_importance(
        explanations
    )

    assert isinstance(result, pd.DataFrame)

    assert set(result.columns) == {
        "feature",
        "Logistic Regression",
        "Random Forest",
    }

    assert len(result) == 2


def test_compare_model_feature_importance_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one model"):
        explainability.compare_model_feature_importance({})


def test_compare_model_feature_importance_rejects_non_dictionary():
    with pytest.raises(ValueError, match="at least one model"):
        explainability.compare_model_feature_importance([])


def test_compare_model_feature_importance_handles_different_features():
    first = {
        "shap_values": np.array([[0.2, 0.4], [0.3, 0.5]]),
        "feature_columns": ["heart_rate", "temperature"],
    }

    second = {
        "shap_values": np.array([[0.1, 0.6], [0.2, 0.7]]),
        "feature_columns": ["heart_rate", "oxygen"],
    }

    result = explainability.compare_model_feature_importance({
        "Model A": first,
        "Model B": second,
    })

    assert set(result["feature"]) == {
        "heart_rate",
        "temperature",
        "oxygen",
    }

    assert not result.isna().any().any()