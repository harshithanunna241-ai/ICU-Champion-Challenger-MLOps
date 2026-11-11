
"""Verify Logistic Regression and Random Forest models stored in MLflow."""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient


def main():

    print("\n========== MLFLOW MODEL VERIFICATION ==========")

    # --------------------------------------------------
    # 1. Configure MLflow tracking database
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parent

    tracking_dir = (
        project_root
        / "artifacts"
        / "mlflow_test"
    )

    database_path = tracking_dir / "mlflow_test.db"

    tracking_uri = (
        f"sqlite:///{database_path.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    print("\nTracking database:", database_path)

    # --------------------------------------------------
    # 2. Find the integration experiment
    # --------------------------------------------------

    experiment_name = "ICU Model Integration Test"

    experiment = client.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        raise RuntimeError(
            "MLflow experiment was not found."
        )

    print("\nExperiment:", experiment.name)
    print("Experiment ID:", experiment.experiment_id)

    # --------------------------------------------------
    # 3. Retrieve experiment runs
    # --------------------------------------------------

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        raise RuntimeError(
            "No model training runs were found."
        )

    print("\nTotal training runs:", len(runs))

    # --------------------------------------------------
    # 4. Define prediction input
    # --------------------------------------------------

    feature_columns = [
        "heart_rate_mean",
        "systolic_bp_mean",
        "diastolic_bp_mean",
        "temperature_mean",
        "spo2_mean",
        "respiratory_rate_mean",
    ]

    sample_data = pd.DataFrame(
        [
            [85, 120, 75, 37.0, 97, 18],
            [110, 105, 65, 38.5, 92, 28],
        ],
        columns=feature_columns,
    )

    print("\nSample prediction input:")
    print(sample_data)

    # --------------------------------------------------
    # 5. Verify both models
    # --------------------------------------------------

    expected_models = [
        "Logistic Regression",
        "Random Forest",
    ]

    verified_models = []

    for expected_name in expected_models:

        print(
            f"\n========== VERIFYING {expected_name.upper()} =========="
        )

        matching_runs = [
            run
            for run in runs
            if run.data.tags.get("model_name") == expected_name
        ]

        if not matching_runs:
            raise RuntimeError(
                f"No MLflow run found for {expected_name}."
            )

        run = matching_runs[0]

        print("Run ID:", run.info.run_id)
        print("Run status:", run.info.status)

        if run.info.status != "FINISHED":
            raise RuntimeError(
                f"{expected_name} training run did not finish."
            )

        # Display stored metrics.
        print("\nStored evaluation metrics:")

        for metric_name, metric_value in run.data.metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        # Find the Logged Model associated with this run.
        logged_models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"source_run_id = '{run.info.run_id}'"
            ),
            max_results=10,
        )

        if not logged_models:
            raise RuntimeError(
                f"No logged model found for {expected_name}."
            )

        logged_model = logged_models[0]

        print("\nLogged Model ID:", logged_model.model_id)
        print("Logged Model status:", logged_model.status)
        print("Model URI:", logged_model.model_uri)

        if logged_model.status != "READY":
            raise RuntimeError(
                f"{expected_name} model is not READY."
            )

        # Load the saved model directly from MLflow.
        loaded_model = mlflow.sklearn.load_model(
            logged_model.model_uri
        )

        print("\nModel loading: SUCCESS")

        # Generate predictions.
        predictions = loaded_model.predict(sample_data)

        print("\nPredictions:")

        for index, prediction in enumerate(predictions):

            print(
                f"Sample {index + 1}: "
                f"Prediction = {int(prediction)}"
            )

        if len(predictions) != len(sample_data):
            raise RuntimeError(
                "Prediction count does not match input count."
            )

        if not set(predictions).issubset({0, 1}):
            raise RuntimeError(
                "Unexpected prediction labels."
            )

        verified_models.append(expected_name)

        print(f"\n{expected_name} verification: SUCCESS")

    # --------------------------------------------------
    # 6. Final verification report
    # --------------------------------------------------

    print("\n========== FINAL VERIFICATION REPORT ==========")

    for model_name in verified_models:
        print(f"{model_name}: VERIFIED")

    print("\nTotal models verified:", len(verified_models))

    if len(verified_models) != 2:
        raise RuntimeError(
            "Not all expected models were verified."
        )

    print("\nMLflow model verification completed successfully.")
    print("Both models were loaded and generated predictions.")

    print("================================================")


if __name__ == "__main__":
    main()
