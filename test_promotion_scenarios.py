
"""Test multiple Champion-Challenger promotion scenarios."""

from src.evaluation.promotion_policy import evaluate_promotion


def print_result(scenario, result):

    print(f"\n========== {scenario} ==========")

    print("Decision:", result["decision"])
    print("Promotion approved:", result["promotion_approved"])
    print("Current Champion:", result["current_champion"])
    print("Challenger:", result["challenger"])

    print("\nReasons:")

    for reason in result["reasons"]:
        print("-", reason)


def main():

    print("\n========== PROMOTION SCENARIO TESTS ==========")

    # Scenario 1: Challenger performs better.
    champion_metrics = {
        "auprc": 0.80,
        "recall": 0.70,
        "brier_score": 0.20,
    }

    challenger_metrics = {
        "auprc": 0.90,
        "recall": 0.68,
        "brier_score": 0.18,
    }

    result_1 = evaluate_promotion(
        champion_metrics,
        challenger_metrics,
    )

    print_result(
        "SCENARIO 1: PROMOTION APPROVED",
        result_1,
    )

    if not result_1["promotion_approved"]:
        raise AssertionError(
            "Scenario 1 should approve the Challenger."
        )

    # Scenario 2: Challenger fails minimum AUPRC.
    champion_metrics_2 = {
        "auprc": 0.80,
        "recall": 0.70,
        "brier_score": 0.20,
    }

    challenger_metrics_2 = {
        "auprc": 0.82,
        "recall": 0.68,
        "brier_score": 0.18,
    }

    result_2 = evaluate_promotion(
        champion_metrics_2,
        challenger_metrics_2,
    )

    print_result(
        "SCENARIO 2: PROMOTION REJECTED",
        result_2,
    )

    if result_2["promotion_approved"]:
        raise AssertionError(
            "Scenario 2 should reject the Challenger."
        )

    # Scenario 3: Challenger fails the Brier score limit.
    champion_metrics_3 = {
        "auprc": 0.80,
        "recall": 0.70,
        "brier_score": 0.20,
    }

    challenger_metrics_3 = {
        "auprc": 0.90,
        "recall": 0.68,
        "brier_score": 0.30,
    }

    result_3 = evaluate_promotion(
        champion_metrics_3,
        challenger_metrics_3,
    )

    print_result(
        "SCENARIO 3: PROMOTION REJECTED",
        result_3,
    )

    if result_3["promotion_approved"]:
        raise AssertionError(
            "Scenario 3 should reject the Challenger."
        )

    # Final report.
    print("\n========== FINAL TEST REPORT ==========")

    print("Scenario 1: PASSED")
    print("Scenario 2: PASSED")
    print("Scenario 3: PASSED")

    print("\nAll promotion scenarios passed successfully.")


if __name__ == "__main__":
    main()
