
"""Test model registration, promotion, and archiving."""

import tempfile

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.registry.model_registry import (
    register_model,
    list_registered_models,
    promote_model,
    archive_model,
)


def main():

    # Small synthetic dataset for testing only.
    X = [
        [80, 110],
        [90, 120],
        [100, 130],
        [110, 140],
        [120, 150],
        [70, 100],
    ]

    y = [0, 0, 1, 1, 1, 0]

    feature_columns = [
        "heart_rate_mean",
        "systolic_bp_mean",
    ]

    # Train two models.
    logistic_model = LogisticRegression()
    logistic_model.fit(X, y)

    random_forest_model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    random_forest_model.fit(X, y)

    # Use a temporary registry for this test.
    with tempfile.TemporaryDirectory() as temp_dir:

        # Register Logistic Regression as Champion.
        register_model(
            model=logistic_model,
            model_name="Logistic Regression",
            version="v1",
            metrics={
                "accuracy": 0.85,
                "auprc": 0.80,
                "recall": 0.70,
                "brier_score": 0.20,
            },
            feature_columns=feature_columns,
            stage="champion",
            registry_dir=temp_dir,
        )

        # Register Random Forest as Candidate.
        register_model(
            model=random_forest_model,
            model_name="Random Forest",
            version="v1",
            metrics={
                "accuracy": 0.90,
                "auprc": 0.90,
                "recall": 0.68,
                "brier_score": 0.18,
            },
            feature_columns=feature_columns,
            stage="candidate",
            registry_dir=temp_dir,
        )

        print("\n========== MODEL LIFECYCLE REPORT ==========")

        # Display stages before promotion.
        print("\nBefore promotion:")

        models = list_registered_models(
            registry_dir=temp_dir
        )

        for model in models:
            print(
                model["model_name"],
                ":",
                model["stage"],
            )

        # Promote Random Forest.
        promote_model(
            model_name="Random Forest",
            version="v1",
            registry_dir=temp_dir,
        )

        # Display stages after promotion.
        print("\nAfter promotion:")

        models = list_registered_models(
            registry_dir=temp_dir
        )

        for model in models:
            print(
                model["model_name"],
                ":",
                model["stage"],
            )

        champion_count = sum(
            1 for model in models
            if model["stage"] == "champion"
        )

        print("\nNumber of Champions:", champion_count)

        # Test protection against archiving the current Champion.
        print("\nAttempting to archive the current Champion:")

        try:
            archive_model(
                model_name="Random Forest",
                version="v1",
                registry_dir=temp_dir,
            )

        except ValueError as error:
            print("Expected protection:", error)

        # Verify that the Champion remains unchanged.
        print("\nFinal model stages:")

        final_models = list_registered_models(
            registry_dir=temp_dir
        )

        for model in final_models:
            print(
                model["model_name"],
                ":",
                model["stage"],
            )

        final_champions = sum(
            1 for model in final_models
            if model["stage"] == "champion"
        )

        assert final_champions == 1

        print("\nModel lifecycle test completed successfully.")
        print("============================================")


if __name__ == "__main__":
    main()
