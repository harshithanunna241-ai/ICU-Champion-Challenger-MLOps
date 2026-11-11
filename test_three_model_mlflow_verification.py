
"""Verify all three models stored in MLflow."""

from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient


def main():

    print("\n========== THREE-MODEL MLflow VERIFICATION ==========")

    project_root = Path(__file__).resolve().parent

    tracking_database = (
        project_root
        / "artifacts"
        / "mlflow_test"
        / "mlflow_test.db"
    )

    if not tracking_database.exists():
        raise FileNotFoundError(
            f"MLflow database not found: {tracking_database}"
        )

    # IMPORTANT: Use SQLite URI, not file:// URI.
    tracking_uri = (
        "sqlite:///"
        + tracking_database.resolve().as_posix()
    )

    print(f"Tracking database: {tracking_database}")
    print(f"Tracking URI: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_name = "ICU Model Integration Test"

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment not found: {experiment_name}"
        )

    model_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]

    sample_data = pd.DataFrame({
        "heart_rate_mean": [85.0, 125.0],
        "systolic_bp_mean": [120.0, 85.0],
        "diastolic_bp_mean": [80.0, 55.0],
        "temperature_mean": [37.0, 39.0],
        "spo2_mean": [98.0, 88.0],
        "respiratory_rate_mean": [16.0, 28.0],
    })

    for model_name in model_names:

        print(f"\n========== VERIFYING {model_name.upper()} ==========")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_name = '{model_name}'",
            order_by=["attributes.start_time DESC"],
            max_results=100,
        )

        if not runs:
            raise AssertionError(
                f"No MLflow run found for {model_name}."
            )

        run = runs[0]

        print(f"Run ID: {run.info.run_id}")

        if not run.data.params:
            raise AssertionError(
                f"Parameters missing for {model_name}."
            )

        print("Parameters: Verified")

        if not run.data.metrics:
            raise AssertionError(
                f"Metrics missing for {model_name}."
            )

        print("Metrics: Verified")

        # Verify feature schema.
        schema_path = client.download_artifacts(
            run.info.run_id,
            "feature_schema.json",
        )

        with open(schema_path, "r", encoding="utf-8") as file:
            schema = json.load(file)

        if not schema.get("feature_columns"):
            raise AssertionError(
                f"Feature schema is empty for {model_name}."
            )

        print("Feature schema: Verified")

        # Find model associated with the run.
        logged_models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"source_run_id = '{run.info.run_id}'"
            ),
            max_results=10,
        )

        if not logged_models:
            raise AssertionError(
                f"Logged model not found for {model_name}."
            )

        logged_model = logged_models[0]

        if logged_model.status != "READY":
            raise AssertionError(
                f"Model status is not READY: {model_name}"
            )

        print(f"Model status: {logged_model.status}")

        # Load the model.
        loaded_model = mlflow.sklearn.load_model(
            logged_model.model_uri
        )

        print("Model loading: Successful")

        # Generate predictions using the model's own feature schema.
        feature_columns = schema["feature_columns"]

        missing_features = [
            column
            for column in feature_columns
            if column not in sample_data.columns
        ]

        if missing_features:
            raise AssertionError(
                f"Missing sample features: {missing_features}"
            )

        predictions = loaded_model.predict(
            sample_data[feature_columns]
        )

        if len(predictions) != 2:
            raise AssertionError(
                f"Incorrect prediction count for {model_name}."
            )

        print(f"Predictions: {predictions.tolist()}")
        print("Prediction test: Successful")

    print("\n==============================================")
    print("ALL THREE MODELS VERIFIED SUCCESSFULLY")
    print("==============================================")


if __name__ == "__main__":
    main()
