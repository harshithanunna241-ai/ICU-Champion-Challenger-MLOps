"""Unit tests for integrated three-model SHAP explainability."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import three_model_explainability as module


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "heart_rate": [70.0, 80.0, 90.0],
        "temperature": [36.5, 37.0, 37.5],
    })


@pytest.fixture
def sample_models():
    return {
        "Logistic Regression": ("lr_model", ["heart_rate", "temperature"]),
        "Random Forest": ("rf_model", ["heart_rate", "temperature"]),
        "XGBoost": ("xgb_model", ["heart_rate", "temperature"]),
    }


@pytest.fixture
def sample_explanation():
    return {
        "explainer": None,
        "shap_values": np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]),
        "feature_data": pd.DataFrame({
            "heart_rate": [70.0, 80.0, 90.0],
            "temperature": [36.5, 37.0, 37.5],
        }),
        "feature_columns": ["heart_rate", "temperature"],
    }


@pytest.fixture
def sample_global_report():
    return pd.DataFrame({
        "feature": ["heart_rate", "temperature"],
        "mean_absolute_shap": [0.3, 0.2],
    })


@pytest.fixture
def sample_individual_report():
    return pd.DataFrame({
        "feature": ["heart_rate", "temperature"],
        "feature_value": [70.0, 36.5],
        "shap_value": [0.1, 0.2],
        "absolute_shap": [0.1, 0.2],
    })


@pytest.fixture
def sample_comparison():
    return pd.DataFrame({
        "feature": ["heart_rate", "temperature"],
        "Logistic Regression": [0.3, 0.2],
        "Random Forest": [0.4, 0.1],
    })


@pytest.fixture
def sample_results(
    sample_global_report,
    sample_individual_report,
    sample_comparison,
):
    return {
        "explanations": {},
        "global_reports": {
            "Logistic Regression": sample_global_report,
            "Random Forest": sample_global_report,
        },
        "individual_reports": {
            "Logistic Regression": sample_individual_report,
            "Random Forest": sample_individual_report,
        },
        "feature_importance_comparison": sample_comparison,
    }


# ---------------------------------------------------------
# explain_three_models: input validation
# ---------------------------------------------------------


def test_explain_three_models_rejects_non_dictionary_models(
    sample_data,
):
    with pytest.raises(ValueError, match="at least one trained model"):
        module.explain_three_models(
            models=[],
            explanation_data=sample_data,
        )


def test_explain_three_models_rejects_empty_models(
    sample_data,
):
    with pytest.raises(ValueError, match="at least one trained model"):
        module.explain_three_models(
            models={},
            explanation_data=sample_data,
        )


def test_explain_three_models_rejects_non_dataframe():
    with pytest.raises(TypeError, match="must be a DataFrame"):
        module.explain_three_models(
            models={"Model A": ("model", ["feature"])},
            explanation_data=[[1, 2]],
        )


def test_explain_three_models_rejects_empty_dataframe():
    data = pd.DataFrame(columns=["heart_rate"])

    with pytest.raises(ValueError, match="cannot be empty"):
        module.explain_three_models(
            models={"Model A": ("model", ["heart_rate"])},
            explanation_data=data,
        )


# ---------------------------------------------------------
# explain_three_models: explanation generation
# ---------------------------------------------------------


def test_explain_three_models_processes_all_models(
    monkeypatch,
    sample_models,
    sample_data,
    sample_explanation,
    sample_global_report,
    sample_individual_report,
    sample_comparison,
):
    explained_models = []

    def fake_explain_model(
        model,
        data,
        feature_columns,
        max_background_samples,
    ):
        explained_models.append(model)
        return sample_explanation

    monkeypatch.setattr(
        module,
        "explain_model",
        fake_explain_model,
    )

    monkeypatch.setattr(
        module,
        "calculate_global_feature_importance",
        lambda explanation: sample_global_report,
    )

    monkeypatch.setattr(
        module,
        "explain_individual_prediction",
        lambda explanation, row_index: sample_individual_report,
    )

    monkeypatch.setattr(
        module,
        "compare_model_feature_importance",
        lambda explanations: sample_comparison,
    )

    result = module.explain_three_models(
        sample_models,
        sample_data,
    )

    assert len(explained_models) == 3

    assert explained_models == [
        "lr_model",
        "rf_model",
        "xgb_model",
    ]

    assert set(result["explanations"]) == set(sample_models)
    assert set(result["global_reports"]) == set(sample_models)
    assert set(result["individual_reports"]) == set(sample_models)


def test_explain_three_models_returns_expected_keys(
    monkeypatch,
    sample_models,
    sample_data,
    sample_explanation,
    sample_global_report,
    sample_individual_report,
    sample_comparison,
):
    monkeypatch.setattr(
        module,
        "explain_model",
        lambda **kwargs: sample_explanation,
    )

    monkeypatch.setattr(
        module,
        "calculate_global_feature_importance",
        lambda explanation: sample_global_report,
    )

    monkeypatch.setattr(
        module,
        "explain_individual_prediction",
        lambda explanation, row_index: sample_individual_report,
    )

    monkeypatch.setattr(
        module,
        "compare_model_feature_importance",
        lambda explanations: sample_comparison,
    )

    result = module.explain_three_models(
        sample_models,
        sample_data,
    )

    assert set(result.keys()) == {
        "explanations",
        "global_reports",
        "individual_reports",
        "feature_importance_comparison",
    }


def test_explain_three_models_passes_background_sample_limit(
    monkeypatch,
    sample_data,
    sample_explanation,
    sample_global_report,
    sample_individual_report,
    sample_comparison,
):
    captured = {}

    def fake_explain_model(**kwargs):
        captured["max_background_samples"] = (
            kwargs["max_background_samples"]
        )
        return sample_explanation

    monkeypatch.setattr(
        module,
        "explain_model",
        fake_explain_model,
    )

    monkeypatch.setattr(
        module,
        "calculate_global_feature_importance",
        lambda explanation: sample_global_report,
    )

    monkeypatch.setattr(
        module,
        "explain_individual_prediction",
        lambda explanation, row_index: sample_individual_report,
    )

    monkeypatch.setattr(
        module,
        "compare_model_feature_importance",
        lambda explanations: sample_comparison,
    )

    module.explain_three_models(
        {"Model A": ("model_a", ["heart_rate"])},
        sample_data,
        max_background_samples=25,
    )

    assert captured["max_background_samples"] == 25


def test_explain_three_models_uses_first_row_for_individual_explanation(
    monkeypatch,
    sample_data,
    sample_explanation,
    sample_global_report,
    sample_comparison,
):
    captured = {}

    monkeypatch.setattr(
        module,
        "explain_model",
        lambda **kwargs: sample_explanation,
    )

    monkeypatch.setattr(
        module,
        "calculate_global_feature_importance",
        lambda explanation: sample_global_report,
    )

    def fake_individual_explanation(explanation, row_index):
        captured["row_index"] = row_index
        return pd.DataFrame()

    monkeypatch.setattr(
        module,
        "explain_individual_prediction",
        fake_individual_explanation,
    )

    monkeypatch.setattr(
        module,
        "compare_model_feature_importance",
        lambda explanations: sample_comparison,
    )

    module.explain_three_models(
        {"Model A": ("model_a", ["heart_rate"])},
        sample_data,
    )

    assert captured["row_index"] == 0


def test_explain_three_models_calls_comparison_with_all_explanations(
    monkeypatch,
    sample_models,
    sample_data,
    sample_explanation,
    sample_global_report,
    sample_individual_report,
    sample_comparison,
):
    captured = {}

    monkeypatch.setattr(
        module,
        "explain_model",
        lambda **kwargs: sample_explanation,
    )

    monkeypatch.setattr(
        module,
        "calculate_global_feature_importance",
        lambda explanation: sample_global_report,
    )

    monkeypatch.setattr(
        module,
        "explain_individual_prediction",
        lambda explanation, row_index: sample_individual_report,
    )

    def fake_comparison(explanations):
        captured["models"] = list(explanations.keys())
        return sample_comparison

    monkeypatch.setattr(
        module,
        "compare_model_feature_importance",
        fake_comparison,
    )

    module.explain_three_models(
        sample_models,
        sample_data,
    )

    assert captured["models"] == list(sample_models.keys())


def test_explain_three_models_supports_single_model(
    monkeypatch,
    sample_data,
    sample_explanation,
    sample_global_report,
    sample_individual_report,
    sample_comparison,
):
    monkeypatch.setattr(
        module,
        "explain_model",
        lambda **kwargs: sample_explanation,
    )

    monkeypatch.setattr(
        module,
        "calculate_global_feature_importance",
        lambda explanation: sample_global_report,
    )

    monkeypatch.setattr(
        module,
        "explain_individual_prediction",
        lambda explanation, row_index: sample_individual_report,
    )

    monkeypatch.setattr(
        module,
        "compare_model_feature_importance",
        lambda explanations: sample_comparison,
    )

    result = module.explain_three_models(
        {"Model A": ("model_a", ["heart_rate"])},
        sample_data,
    )

    assert list(result["explanations"]) == ["Model A"]


def test_explain_three_models_propagates_explain_model_error(
    monkeypatch,
    sample_models,
    sample_data,
):
    def fake_explain_model(**kwargs):
        raise RuntimeError("SHAP calculation failed")

    monkeypatch.setattr(
        module,
        "explain_model",
        fake_explain_model,
    )

    with pytest.raises(RuntimeError, match="SHAP calculation failed"):
        module.explain_three_models(
            sample_models,
            sample_data,
        )


def test_explain_three_models_propagates_global_report_error(
    monkeypatch,
    sample_data,
    sample_explanation,
):
    monkeypatch.setattr(
        module,
        "explain_model",
        lambda **kwargs: sample_explanation,
    )

    def fake_global_report(explanation):
        raise RuntimeError("Global report failed")

    monkeypatch.setattr(
        module,
        "calculate_global_feature_importance",
        fake_global_report,
    )

    with pytest.raises(RuntimeError, match="Global report failed"):
        module.explain_three_models(
            {"Model A": ("model_a", ["heart_rate"])},
            sample_data,
        )


# ---------------------------------------------------------
# create_explainability_report: input validation
# ---------------------------------------------------------


def test_create_report_rejects_non_dictionary():
    with pytest.raises(TypeError, match="must be a dictionary"):
        module.create_explainability_report([])


@pytest.mark.parametrize(
    "missing_key",
    [
        "global_reports",
        "individual_reports",
        "feature_importance_comparison",
    ],
)
def test_create_report_rejects_missing_components(
    sample_results,
    missing_key,
):
    results = sample_results.copy()
    del results[missing_key]

    with pytest.raises(
        ValueError,
        match=f"Missing report component: {missing_key}",
    ):
        module.create_explainability_report(results)


def test_create_report_rejects_non_dataframe_comparison(
    sample_results,
):
    sample_results["feature_importance_comparison"] = []

    with pytest.raises(
        TypeError,
        match="Feature comparison must be a DataFrame",
    ):
        module.create_explainability_report(sample_results)


def test_create_report_rejects_empty_comparison(
    sample_results,
):
    sample_results["feature_importance_comparison"] = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Feature comparison cannot be empty",
    ):
        module.create_explainability_report(sample_results)


# ---------------------------------------------------------
# create_explainability_report: output tests
# ---------------------------------------------------------


def test_create_report_returns_expected_keys(sample_results):
    result = module.create_explainability_report(sample_results)

    assert set(result.keys()) == {
        "model_count",
        "global_feature_importance",
        "individual_prediction_explanations",
        "model_feature_comparison",
    }


def test_create_report_returns_correct_model_count(sample_results):
    result = module.create_explainability_report(sample_results)

    assert result["model_count"] == 2


def test_create_report_preserves_global_reports(sample_results):
    result = module.create_explainability_report(sample_results)

    assert result["global_feature_importance"] is (
        sample_results["global_reports"]
    )


def test_create_report_preserves_individual_reports(sample_results):
    result = module.create_explainability_report(sample_results)

    assert result["individual_prediction_explanations"] is (
        sample_results["individual_reports"]
    )


def test_create_report_preserves_comparison_dataframe(sample_results):
    result = module.create_explainability_report(sample_results)

    pd.testing.assert_frame_equal(
        result["model_feature_comparison"],
        sample_results["feature_importance_comparison"],
    )


def test_create_report_supports_one_model(
    sample_results,
    sample_global_report,
    sample_individual_report,
):
    sample_results["global_reports"] = {
        "Model A": sample_global_report,
    }

    sample_results["individual_reports"] = {
        "Model A": sample_individual_report,
    }

    result = module.create_explainability_report(sample_results)

    assert result["model_count"] == 1


def test_create_report_returns_zero_models_when_reports_are_empty(
    sample_results,
):
    sample_results["global_reports"] = {}
    sample_results["individual_reports"] = {}

    result = module.create_explainability_report(sample_results)

    assert result["model_count"] == 0