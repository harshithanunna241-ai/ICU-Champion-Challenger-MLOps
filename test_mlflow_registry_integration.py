
"""Integrate MLflow models with the local Champion-Challenger registry."""

import json
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient

from src.registry.model_registry import (
    register_model,
    list_registered_models,
    load_registered_model,
)


def main():

    print("\n========== MLFLOW REGISTRY INTEGRATION ==========")

    # --------------------------------------------------
    # 1. Configure MLflow
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parent

    tracking_database = (
        project_root
        / "artifacts"
        / "mlflow_test"
        / "mlflow_test.db"
    )

    tracking_uri = (
        f"sqlite:///{tracking_database.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_name = "ICU Model Integration Test"

    experiment = client.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        raise RuntimeError("MLflow experiment was not found.")

    print("\nMLflow experiment:", experiment.name)

    # --------------------------------------------------
    # 2. Prepare an isolated test registry
    # --------------------------------------------------

    registry_dir = (
        project_root
        / "artifacts"
        / "mlflow_registry_test"
    )

    # This removes only the dedicated test registry folder.
    # It does not modify the actual model_registry folder.
    if registry_dir.exists():
        shutil.rmtree(registry_dir)

    registry_dir.mkdir(parents=True, exist_ok=True)

    print("Test registry:", registry_dir)

    # --------------------------------------------------
    # 3. Find both MLflow training runs
    # --------------------------------------------------

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )

    expected_models = [
        {
            "name": "Logistic Regression",
            "version": "v1",
            "stage": "champion",
        },
        {
            "name": "Random Forest",
            "version": "v1",
            "stage": "candidate",
        },
    ]

    registered_names = []

    # --------------------------------------------------
    # 4. Retrieve and register each MLflow model
    # --------------------------------------------------

    for model_info in expected_models:

        model_name = model_info["name"]
        version = model_info["version"]
        stage = model_info["stage"]

        print(f"\n========== {model_name.upper()} ==========")

        matching_runs = [
            run
            for run in runs
            if run.data.tags.get("model_name") == model_name
        ]

        if not matching_runs:
            raise RuntimeError(
                f"No MLflow run found for {model_name}."
            )

        run = matching_runs[0]

        if run.info.status != "FINISHED":
            raise RuntimeError(
                f"{model_name} run did not finish successfully."
            )

        print("MLflow Run ID:", run.info.run_id)

        # Retrieve the Logged Model.
        logged_models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"source_run_id = '{run.info.run_id}'"
            ),
            max_results=10,
        )

        if not logged_models:
            raise RuntimeError(
                f"No Logged Model found for {model_name}."
            )

        logged_model = logged_models[0]

        if logged_model.status != "READY":
            raise RuntimeError(
                f"{model_name} Logged Model is not READY."
            )

        # Load the model from MLflow.
        model = mlflow.sklearn.load_model(
            logged_model.model_uri
        )

        print("MLflow model loaded successfully.")

        # Retrieve the feature schema saved during tracking.
        schema_path = client.download_artifacts(
            run.info.run_id,
            "feature_schema.json",
        )

        with open(schema_path, "r", encoding="utf-8") as file:
            schema = json.load(file)

        feature_columns = schema["feature_columns"]

        # Retrieve evaluation metrics from the MLflow run.
        metrics = dict(run.data.metrics)

        print("Feature columns:", feature_columns)
        print("Metrics retrieved:", len(metrics))

        # Register the model in the local registry.
        registration = register_model(
            model=model,
            model_name=model_name,
            version=version,
            metrics=metrics,
            feature_columns=feature_columns,
            stage=stage,
            registry_dir=registry_dir,
        )

        registered_names.append(model_name)

        print("Registry registration: SUCCESS")
        print("Model name:", model_name)
        print("Version:", version)
        print("Stage:", stage)

    # --------------------------------------------------
    # 5. Verify registry records
    # --------------------------------------------------

    print("\n========== REGISTRY RECORDS ==========")

    registry_records = list_registered_models(
        registry_dir=registry_dir
    )

    print("Total registered models:", len(registry_records))

    for record in registry_records:
        print("\nModel:", record["model_name"])
        print("Version:", record["version"])
        print("Stage:", record["stage"])

    if len(registry_records) != 2:
        raise RuntimeError(
            "Expected exactly two registered models."
        )

    # --------------------------------------------------
    # 6. Load both models from the local registry
    # --------------------------------------------------

    sample_data = pd.DataFrame(
        [
            [85, 120, 75, 37.0, 97, 18],
            [110, 105, 65, 38.5, 92, 28],
        ],
        columns=[
            "heart_rate_mean",
            "systolic_bp_mean",
            "diastolic_bp_mean",
            "temperature_mean",
            "spo2_mean",
            "respiratory_rate_mean",
        ],
    )

    print("\n========== REGISTRY MODEL LOADING ==========")

    for model_name in registered_names:

        loaded_model = load_registered_model(
            model_name=model_name,
            version="v1",
            registry_dir=registry_dir,
        )

        predictions = loaded_model.predict(sample_data)

        print(f"\nModel: {model_name}")
        print("Registry loading: SUCCESS")
        print("Predictions:", predictions.tolist())

        if len(predictions) != len(sample_data):
            raise RuntimeError(
                f"Prediction count mismatch for {model_name}."
            )

        if not set(predictions).issubset({0, 1}):
            raise RuntimeError(
                f"Unexpected prediction labels for {model_name}."
            )

    # --------------------------------------------------
    # 7. Final report
    # --------------------------------------------------

    print("\n========== FINAL INTEGRATION REPORT ==========")

    print("Logistic Regression: CHAMPION")
    print("Random Forest: CHALLENGER")

    print("Models registered:", len(registry_records))
    print("Models loaded successfully:", len(registered_names))

    print("\nMLflow to Model Registry integration: SUCCESS")

    print("================================================")


if __name__ == "__main__":
    main()
