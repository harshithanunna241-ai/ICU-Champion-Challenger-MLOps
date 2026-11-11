
import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate_three_models_with_ci import (
    evaluate_model_with_ci,
    evaluate_three_models_with_ci,
    create_confidence_interval_table,
)


class FakeModel:
    """Fake classification model for unit testing."""

    def predict_proba(self, X):
        probabilities = np.where(
            X["feature"] > 0,
            0.8,
            0.2,
        )

        return np.column_stack(
            (1 - probabilities, probabilities)
        )


@pytest.fixture
def test_data():
    return pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8],
        "deterioration_event": [0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def feature_columns():
    return ["feature"]


@pytest.fixture
def confidence_results():
    return {
        "confidence_level": 0.95,
        "threshold": 0.5,
        "n_bootstraps": 1000,
        "metrics": {
            "auroc": {
                "estimate": 0.90,
                "lower_bound": 0.80,
                "upper_bound": 0.98,
                "valid_bootstrap_samples": 1000,
            },
            "auprc": {
                "estimate": 0.85,
                "lower_bound": 0.75,
                "upper_bound": 0.95,
                "valid_bootstrap_samples": 1000,
            },
        },
    }


# --------------------------------------------------
# evaluate_model_with_ci tests
# --------------------------------------------------

def test_evaluate_model_with_ci_returns_dictionary(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_model_with_ci(
        fake_model,
        test_data,
        feature_columns,
        "Test Model",
        n_bootstraps=100,
    )

    assert isinstance(result, dict)


def test_evaluate_model_with_ci_returns_model_name(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_model_with_ci(
        fake_model,
        test_data,
        feature_columns,
        "Test Model",
        n_bootstraps=100,
    )

    assert result["model_name"] == "Test Model"


def test_evaluate_model_with_ci_returns_record_count(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_model_with_ci(
        fake_model,
        test_data,
        feature_columns,
        "Test Model",
        n_bootstraps=100,
    )

    assert result["total_test_records"] == len(test_data)


def test_evaluate_model_with_ci_returns_confidence_intervals(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_model_with_ci(
        fake_model,
        test_data,
        feature_columns,
        "Test Model",
        n_bootstraps=100,
    )

    assert "confidence_intervals" in result
    assert "metrics" in result["confidence_intervals"]


def test_evaluate_model_with_ci_rejects_non_dataframe(
    fake_model,
    feature_columns,
):
    with pytest.raises(TypeError):
        evaluate_model_with_ci(
            fake_model,
            [1, 2, 3],
            feature_columns,
            "Test Model",
        )


def test_evaluate_model_with_ci_rejects_empty_dataframe(
    fake_model,
    feature_columns,
):
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_model_with_ci(
            fake_model,
            pd.DataFrame(),
            feature_columns,
            "Test Model",
        )


def test_evaluate_model_with_ci_rejects_missing_target(
    fake_model,
    feature_columns,
):
    data = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing target column"):
        evaluate_model_with_ci(
            fake_model,
            data,
            feature_columns,
            "Test Model",
        )


def test_evaluate_model_with_ci_rejects_empty_features(
    test_data,
    fake_model,
):
    with pytest.raises(
        ValueError,
        match="feature_columns cannot be empty",
    ):
        evaluate_model_with_ci(
            fake_model,
            test_data,
            [],
            "Test Model",
        )


def test_evaluate_model_with_ci_rejects_missing_features(
    test_data,
    fake_model,
):
    with pytest.raises(ValueError, match="Missing model features"):
        evaluate_model_with_ci(
            fake_model,
            test_data,
            ["missing_feature"],
            "Test Model",
        )


def test_evaluate_model_with_ci_rejects_missing_target_values(
    test_data,
    fake_model,
    feature_columns,
):
    data = test_data.copy()
    data.loc[0, "deterioration_event"] = np.nan

    with pytest.raises(
        ValueError,
        match="Target column contains missing",
    ):
        evaluate_model_with_ci(
            fake_model,
            data,
            feature_columns,
            "Test Model",
        )


@pytest.mark.parametrize(
    "labels",
    [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 2, 1],
    ],
)
def test_evaluate_model_with_ci_rejects_invalid_target(
    fake_model,
    labels,
):
    data = pd.DataFrame({
        "feature": range(len(labels)),
        "deterioration_event": labels,
    })

    with pytest.raises(
        ValueError,
        match="both binary classes",
    ):
        evaluate_model_with_ci(
            fake_model,
            data,
            ["feature"],
            "Test Model",
        )


def test_evaluate_model_with_ci_uses_custom_target_column(
    fake_model,
):
    data = pd.DataFrame({
        "feature": list(range(1, 9)),
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    result = evaluate_model_with_ci(
        fake_model,
        data,
        ["feature"],
        "Test Model",
        target_column="target",
        n_bootstraps=1000,
    )

    assert result["total_test_records"] == 8
    assert result["model_name"] == "Test Model"
    assert "confidence_intervals" in result


def test_evaluate_model_with_ci_uses_custom_threshold(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_model_with_ci(
        fake_model,
        test_data,
        feature_columns,
        "Test Model",
        threshold=0.6,
        n_bootstraps=100,
    )

    assert result["confidence_intervals"]["threshold"] == 0.6


def test_evaluate_model_with_ci_rejects_model_without_predict_proba(
    test_data,
    feature_columns,
):
    class InvalidModel:
        pass

    with pytest.raises(AttributeError):
        evaluate_model_with_ci(
            InvalidModel(),
            test_data,
            feature_columns,
            "Invalid Model",
        )


# --------------------------------------------------
# evaluate_three_models_with_ci tests
# --------------------------------------------------

def test_evaluate_three_models_with_ci_returns_dictionary(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_three_models_with_ci(
        fake_model,
        fake_model,
        fake_model,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
        n_bootstraps=100,
    )

    assert isinstance(result, dict)


def test_evaluate_three_models_with_ci_returns_all_models(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_three_models_with_ci(
        fake_model,
        fake_model,
        fake_model,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
        n_bootstraps=100,
    )

    assert set(result.keys()) == {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }


def test_evaluate_three_models_with_ci_returns_three_results(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_three_models_with_ci(
        fake_model,
        fake_model,
        fake_model,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
        n_bootstraps=100,
    )

    assert len(result) == 3


def test_evaluate_three_models_with_ci_uses_custom_bootstraps(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_three_models_with_ci(
        fake_model,
        fake_model,
        fake_model,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
        n_bootstraps=100,
    )

    for model_result in result.values():
        assert (
            model_result["confidence_intervals"]["n_bootstraps"]
            == 100
        )


def test_evaluate_three_models_with_ci_uses_custom_threshold(
    test_data,
    fake_model,
    feature_columns,
):
    result = evaluate_three_models_with_ci(
        fake_model,
        fake_model,
        fake_model,
        test_data,
        feature_columns,
        feature_columns,
        feature_columns,
        threshold=0.6,
        n_bootstraps=100,
    )

    for model_result in result.values():
        assert model_result["confidence_intervals"]["threshold"] == 0.6


def test_evaluate_three_models_with_ci_rejects_invalid_test_data(
    fake_model,
    feature_columns,
):
    with pytest.raises(TypeError):
        evaluate_three_models_with_ci(
            fake_model,
            fake_model,
            fake_model,
            [1, 2, 3],
            feature_columns,
            feature_columns,
            feature_columns,
        )


# --------------------------------------------------
# create_confidence_interval_table tests
# --------------------------------------------------

def test_create_confidence_interval_table_returns_dataframe(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        }
    }

    table = create_confidence_interval_table(results)

    assert isinstance(table, pd.DataFrame)


def test_create_confidence_interval_table_returns_expected_columns(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        }
    }

    table = create_confidence_interval_table(results)

    expected_columns = [
        "model",
        "metric",
        "estimate",
        "lower_bound",
        "upper_bound",
        "valid_bootstrap_samples",
    ]

    assert list(table.columns) == expected_columns


def test_create_confidence_interval_table_returns_metric_rows(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        }
    }

    table = create_confidence_interval_table(results)

    assert len(table) == 2


def test_create_confidence_interval_table_preserves_model_name(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        }
    }

    table = create_confidence_interval_table(results)

    assert set(table["model"]) == {"Logistic Regression"}


def test_create_confidence_interval_table_preserves_metric_names(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        }
    }

    table = create_confidence_interval_table(results)

    assert set(table["metric"]) == {"auroc", "auprc"}


def test_create_confidence_interval_table_preserves_estimates(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        }
    }

    table = create_confidence_interval_table(results)

    auroc_row = table[table["metric"] == "auroc"].iloc[0]

    assert auroc_row["estimate"] == 0.90


def test_create_confidence_interval_table_preserves_bounds(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        }
    }

    table = create_confidence_interval_table(results)

    auroc_row = table[table["metric"] == "auroc"].iloc[0]

    assert auroc_row["lower_bound"] == 0.80
    assert auroc_row["upper_bound"] == 0.98


def test_create_confidence_interval_table_preserves_valid_samples(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        }
    }

    table = create_confidence_interval_table(results)

    assert (table["valid_bootstrap_samples"] == 1000).all()


def test_create_confidence_interval_table_supports_three_models(
    confidence_results,
):
    results = {
        "Logistic Regression": {
            "confidence_intervals": confidence_results,
        },
        "Random Forest": {
            "confidence_intervals": confidence_results,
        },
        "XGBoost": {
            "confidence_intervals": confidence_results,
        },
    }

    table = create_confidence_interval_table(results)

    assert len(table) == 6
    assert table["model"].nunique() == 3
