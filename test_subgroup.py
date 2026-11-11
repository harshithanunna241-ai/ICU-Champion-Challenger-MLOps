"""Unit tests for ICU model subgroup evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import subgroup


class FakeModel:
    """Fake binary classification model."""

    def predict_proba(self, data):
        probabilities = np.asarray(
            data["heart_rate"],
            dtype=float,
        ) / 200

        return np.column_stack([
            1 - probabilities,
            probabilities,
        ])


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "heart_rate": [40, 60, 80, 100, 50, 70, 90, 110],
        "temperature": [36.5, 37.0, 37.2, 37.5, 36.8, 37.1, 37.4, 38.0],
        "deterioration_event": [0, 0, 1, 1, 0, 1, 0, 1],
        "age_group": [
            "Young", "Young", "Young", "Young",
            "Older", "Older", "Older", "Older",
        ],
    })


@pytest.fixture
def sample_models():
    return {
        "Logistic Regression": FakeModel(),
        "Random Forest": FakeModel(),
        "XGBoost": FakeModel(),
    }


# ---------------------------------------------------------
# evaluate_subgroup tests
# ---------------------------------------------------------


def test_evaluate_subgroup_returns_expected_metrics(sample_data):
    result = subgroup.evaluate_subgroup(
        FakeModel(),
        sample_data,
        ["heart_rate", "temperature"],
    )

    expected_keys = {
        "sample_size",
        "positive_cases",
        "negative_cases",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
    }

    assert set(result.keys()) == expected_keys


def test_evaluate_subgroup_returns_correct_sample_counts(sample_data):
    result = subgroup.evaluate_subgroup(
        FakeModel(),
        sample_data,
        ["heart_rate", "temperature"],
    )

    assert result["sample_size"] == 8
    assert result["positive_cases"] == 4
    assert result["negative_cases"] == 4


def test_evaluate_subgroup_returns_metrics_between_zero_and_one(
    sample_data,
):
    result = subgroup.evaluate_subgroup(
        FakeModel(),
        sample_data,
        ["heart_rate", "temperature"],
    )

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
    ]:
        assert 0 <= result[metric] <= 1


def test_evaluate_subgroup_rejects_non_dataframe():
    with pytest.raises(TypeError, match="must be a DataFrame"):
        subgroup.evaluate_subgroup(
            FakeModel(),
            [[70, 37.0, 1]],
            ["heart_rate"],
        )


def test_evaluate_subgroup_rejects_empty_dataframe():
    data = pd.DataFrame(
        columns=["heart_rate", "deterioration_event"]
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        subgroup.evaluate_subgroup(
            FakeModel(),
            data,
            ["heart_rate"],
        )


def test_evaluate_subgroup_rejects_missing_target_column(sample_data):
    data = sample_data.drop(columns=["deterioration_event"])

    with pytest.raises(ValueError, match="Missing target column"):
        subgroup.evaluate_subgroup(
            FakeModel(),
            data,
            ["heart_rate"],
        )


def test_evaluate_subgroup_rejects_missing_features(sample_data):
    with pytest.raises(ValueError, match="Missing features"):
        subgroup.evaluate_subgroup(
            FakeModel(),
            sample_data,
            ["missing_feature"],
        )


def test_evaluate_subgroup_rejects_missing_target_values(sample_data):
    data = sample_data.copy()
    data.loc[0, "deterioration_event"] = np.nan

    with pytest.raises(ValueError, match="Target contains missing values"):
        subgroup.evaluate_subgroup(
            FakeModel(),
            data,
            ["heart_rate"],
        )


@pytest.mark.parametrize(
    "invalid_target",
    [
        [0, 1, 2, 1, 0, 1, 0, 1],
        [0, 1, -1, 1, 0, 1, 0, 1],
        [0, 1, 0.5, 1, 0, 1, 0, 1],
    ],
)
def test_evaluate_subgroup_rejects_non_binary_target(
    sample_data,
    invalid_target,
):
    data = sample_data.copy()
    data["deterioration_event"] = invalid_target

    with pytest.raises(ValueError, match="binary labels"):
        subgroup.evaluate_subgroup(
            FakeModel(),
            data,
            ["heart_rate"],
        )


def test_evaluate_subgroup_handles_only_positive_cases(sample_data):
    data = sample_data.copy()
    data["deterioration_event"] = 1

    result = subgroup.evaluate_subgroup(
        FakeModel(),
        data,
        ["heart_rate"],
    )

    assert result["positive_cases"] == 8
    assert result["negative_cases"] == 0
    assert np.isnan(result["auroc"])
    assert np.isnan(result["auprc"])


def test_evaluate_subgroup_handles_only_negative_cases(sample_data):
    data = sample_data.copy()
    data["deterioration_event"] = 0

    result = subgroup.evaluate_subgroup(
        FakeModel(),
        data,
        ["heart_rate"],
    )

    assert result["positive_cases"] == 0
    assert result["negative_cases"] == 8
    assert np.isnan(result["auroc"])
    assert np.isnan(result["auprc"])


def test_evaluate_subgroup_rejects_non_finite_probabilities(sample_data):
    class NonFiniteModel:
        def predict_proba(self, data):
            probabilities = np.full(len(data), 0.5)
            probabilities[0] = np.nan

            return np.column_stack([
                1 - probabilities,
                probabilities,
            ])

    with pytest.raises(
        ValueError,
        match="non-finite probabilities",
    ):
        subgroup.evaluate_subgroup(
            NonFiniteModel(),
            sample_data,
            ["heart_rate"],
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [-0.1, 1.1],
)
def test_evaluate_subgroup_rejects_probabilities_outside_range(
    sample_data,
    invalid_probability,
):
    class InvalidProbabilityModel:
        def predict_proba(self, data):
            probabilities = np.full(len(data), 0.5)
            probabilities[0] = invalid_probability

            return np.column_stack([
                1 - probabilities,
                probabilities,
            ])

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        subgroup.evaluate_subgroup(
            InvalidProbabilityModel(),
            sample_data,
            ["heart_rate"],
        )


# ---------------------------------------------------------
# evaluate_model_subgroups tests
# ---------------------------------------------------------


def test_evaluate_model_subgroups_returns_dataframe(sample_data):
    result = subgroup.evaluate_model_subgroups(
        model=FakeModel(),
        test_data=sample_data,
        feature_columns=["heart_rate", "temperature"],
        group_column="age_group",
        model_name="Logistic Regression",
        minimum_group_size=2,
    )

    assert isinstance(result, pd.DataFrame)


def test_evaluate_model_subgroups_returns_expected_columns(sample_data):
    result = subgroup.evaluate_model_subgroups(
        model=FakeModel(),
        test_data=sample_data,
        feature_columns=["heart_rate"],
        group_column="age_group",
        model_name="Logistic Regression",
        minimum_group_size=2,
    )

    expected_columns = {
        "model",
        "group_column",
        "group",
        "sample_size",
        "positive_cases",
        "negative_cases",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auroc",
        "auprc",
    }

    assert set(result.columns) == expected_columns


def test_evaluate_model_subgroups_evaluates_each_group(sample_data):
    result = subgroup.evaluate_model_subgroups(
        model=FakeModel(),
        test_data=sample_data,
        feature_columns=["heart_rate"],
        group_column="age_group",
        model_name="Logistic Regression",
        minimum_group_size=2,
    )

    assert len(result) == 2
    assert set(result["group"]) == {"Young", "Older"}


def test_evaluate_model_subgroups_includes_model_name(sample_data):
    result = subgroup.evaluate_model_subgroups(
        model=FakeModel(),
        test_data=sample_data,
        feature_columns=["heart_rate"],
        group_column="age_group",
        model_name="Random Forest",
        minimum_group_size=2,
    )

    assert set(result["model"]) == {"Random Forest"}


def test_evaluate_model_subgroups_includes_group_column_name(
    sample_data,
):
    result = subgroup.evaluate_model_subgroups(
        model=FakeModel(),
        test_data=sample_data,
        feature_columns=["heart_rate"],
        group_column="age_group",
        model_name="XGBoost",
        minimum_group_size=2,
    )

    assert set(result["group_column"]) == {"age_group"}


def test_evaluate_model_subgroups_skips_small_groups(sample_data):
    data = sample_data.copy()
    data.loc[0:3, "age_group"] = "Large"
    data.loc[4:5, "age_group"] = "Small A"
    data.loc[6:7, "age_group"] = "Small B"

    result = subgroup.evaluate_model_subgroups(
        model=FakeModel(),
        test_data=data,
        feature_columns=["heart_rate"],
        group_column="age_group",
        model_name="Logistic Regression",
        minimum_group_size=3,
    )

    assert len(result) == 1
    assert set(result["group"]) == {"Large"}


def test_evaluate_model_subgroups_returns_empty_dataframe_when_all_groups_are_small(
    sample_data,
):
    result = subgroup.evaluate_model_subgroups(
        model=FakeModel(),
        test_data=sample_data,
        feature_columns=["heart_rate"],
        group_column="age_group",
        model_name="Logistic Regression",
        minimum_group_size=10,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_evaluate_model_subgroups_rejects_non_dataframe():
    with pytest.raises(TypeError, match="must be a DataFrame"):
        subgroup.evaluate_model_subgroups(
            model=FakeModel(),
            test_data=[[1, 2]],
            feature_columns=["heart_rate"],
            group_column="age_group",
            model_name="Test Model",
        )


def test_evaluate_model_subgroups_rejects_missing_group_column(
    sample_data,
):
    with pytest.raises(ValueError, match="not found"):
        subgroup.evaluate_model_subgroups(
            model=FakeModel(),
            test_data=sample_data,
            feature_columns=["heart_rate"],
            group_column="missing_group",
            model_name="Test Model",
        )


@pytest.mark.parametrize(
    "minimum_group_size",
    [0, -1],
)
def test_evaluate_model_subgroups_rejects_invalid_minimum_group_size(
    sample_data,
    minimum_group_size,
):
    with pytest.raises(
        ValueError,
        match="minimum_group_size must be positive",
    ):
        subgroup.evaluate_model_subgroups(
            model=FakeModel(),
            test_data=sample_data,
            feature_columns=["heart_rate"],
            group_column="age_group",
            model_name="Test Model",
            minimum_group_size=minimum_group_size,
        )


def test_evaluate_model_subgroups_rejects_missing_group_values(
    sample_data,
):
    data = sample_data.copy()
    data.loc[0, "age_group"] = np.nan

    with pytest.raises(ValueError, match="contains missing values"):
        subgroup.evaluate_model_subgroups(
            model=FakeModel(),
            test_data=data,
            feature_columns=["heart_rate"],
            group_column="age_group",
            model_name="Test Model",
            minimum_group_size=2,
        )


def test_evaluate_model_subgroups_uses_sorted_group_order(sample_data):
    result = subgroup.evaluate_model_subgroups(
        model=FakeModel(),
        test_data=sample_data,
        feature_columns=["heart_rate"],
        group_column="age_group",
        model_name="Test Model",
        minimum_group_size=2,
    )

    assert result["group"].tolist() == ["Older", "Young"]


# ---------------------------------------------------------
# evaluate_three_model_subgroups tests
# ---------------------------------------------------------


def test_evaluate_three_model_subgroups_rejects_non_list_groups(
    sample_data,
):
    with pytest.raises(ValueError, match="at least one subgroup column"):
        subgroup.evaluate_three_model_subgroups(
            FakeModel(),
            FakeModel(),
            FakeModel(),
            sample_data,
            ["heart_rate"],
            ["heart_rate"],
            ["heart_rate"],
            group_columns="age_group",
        )


def test_evaluate_three_model_subgroups_rejects_empty_groups(
    sample_data,
):
    with pytest.raises(ValueError, match="at least one subgroup column"):
        subgroup.evaluate_three_model_subgroups(
            FakeModel(),
            FakeModel(),
            FakeModel(),
            sample_data,
            ["heart_rate"],
            ["heart_rate"],
            ["heart_rate"],
            group_columns=[],
        )


def test_evaluate_three_model_subgroups_combines_all_models(
    sample_data,
):
    result = subgroup.evaluate_three_model_subgroups(
        logistic_model=FakeModel(),
        random_forest_model=FakeModel(),
        xgboost_model=FakeModel(),
        test_data=sample_data,
        logistic_features=["heart_rate"],
        random_forest_features=["heart_rate"],
        xgboost_features=["heart_rate"],
        group_columns=["age_group"],
        minimum_group_size=2,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 6

    assert set(result["model"]) == {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }


def test_evaluate_three_model_subgroups_handles_multiple_group_columns(
    sample_data,
):
    data = sample_data.copy()
    data["risk_group"] = [
        "Low", "Low", "High", "High",
        "Low", "Low", "High", "High",
    ]

    result = subgroup.evaluate_three_model_subgroups(
        FakeModel(),
        FakeModel(),
        FakeModel(),
        data,
        ["heart_rate"],
        ["heart_rate"],
        ["heart_rate"],
        group_columns=["age_group", "risk_group"],
        minimum_group_size=2,
    )

    assert set(result["group_column"]) == {
        "age_group",
        "risk_group",
    }


def test_evaluate_three_model_subgroups_returns_empty_when_no_groups_qualify(
    sample_data,
):
    result = subgroup.evaluate_three_model_subgroups(
        FakeModel(),
        FakeModel(),
        FakeModel(),
        sample_data,
        ["heart_rate"],
        ["heart_rate"],
        ["heart_rate"],
        group_columns=["age_group"],
        minimum_group_size=10,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_evaluate_three_model_subgroups_passes_minimum_group_size(
    monkeypatch,
    sample_data,
):
    captured = []

    def fake_evaluate_model_subgroups(**kwargs):
        captured.append(kwargs["minimum_group_size"])
        return pd.DataFrame()

    monkeypatch.setattr(
        subgroup,
        "evaluate_model_subgroups",
        fake_evaluate_model_subgroups,
    )

    subgroup.evaluate_three_model_subgroups(
        FakeModel(),
        FakeModel(),
        FakeModel(),
        sample_data,
        ["heart_rate"],
        ["heart_rate"],
        ["heart_rate"],
        group_columns=["age_group"],
        minimum_group_size=5,
    )

    assert captured == [5, 5, 5]