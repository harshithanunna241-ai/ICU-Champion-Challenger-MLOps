
"""Test ICU model performance monitoring."""

import numpy as np
import pandas as pd

from src.monitoring.performance import monitor_model_performance


def main():
    print("\n========== MODEL PERFORMANCE MONITORING TEST ==========")

    rng = np.random.default_rng(42)

    # Generate synthetic actual outcomes
    n = 500

    y_true = np.array([0, 1] * (n // 2))
    rng.shuffle(y_true)

    # Generate prediction probabilities correlated with outcomes
    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.60, 0.95, n),
        rng.uniform(0.05, 0.40, n),
    )

    evaluation_data = pd.DataFrame({
        "deterioration_event": y_true,
        "predicted_probability": y_prob,
    })

    print("Evaluation records:", len(evaluation_data))
    print("Positive outcomes:", int(y_true.sum()))
    print("Negative outcomes:", int((y_true == 0).sum()))

    # Run performance monitoring
    results = monitor_model_performance(
        evaluation_data=evaluation_data,
        target_column="deterioration_event",
        probability_column="predicted_probability",
        decision_threshold=0.50,
    )

    print("\n========== PERFORMANCE REPORT ==========")

    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Validate record counts
    assert results["total_records"] == 500
    assert results["positive_records"] == 250
    assert results["negative_records"] == 250

    # Validate classification metrics
    for metric in [
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "f1_score",
        "auroc",
        "auprc",
        "brier_score",
    ]:
        assert 0 <= results[metric] <= 1, (
            f"Invalid value for {metric}"
        )

    assert results["accuracy"] == 1.0
    assert results["auroc"] == 1.0
    assert results["recall"] == 1.0
    assert results["sensitivity"] == 1.0

    assert results["decision_threshold"] == 0.50

    # Test invalid input: missing column
    try:
        monitor_model_performance(
            evaluation_data=pd.DataFrame({
                "deterioration_event": [0, 1]
            })
        )
    except ValueError:
        print("\nMissing-column validation: PASSED")
    else:
        raise AssertionError("Missing-column validation failed.")

    print("\nAll model performance monitoring tests passed successfully.")


if __name__ == "__main__":
    main()
