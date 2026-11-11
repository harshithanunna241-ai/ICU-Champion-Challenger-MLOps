
"""Verify MLflow 3.x Logged Model and experiment artifacts."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def main():

    print("\n========== MLFLOW FINAL VERIFICATION ==========")

    print("MLflow version:", mlflow.__version__)

    # Step 1: Connect to the existing SQLite database.
    project_root = Path(__file__).resolve().parent

    database_path = (
        project_root
        / "artifacts"
        / "mlflow_test"
        / "mlflow_test.db"
    )

    if not database_path.exists():
        raise FileNotFoundError("MLflow database not found.")

    tracking_uri = (
        f"sqlite:///{database_path.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    print("\nSQLite database: FOUND")

    # Step 2: Find experiment.
    experiment = client.get_experiment_by_name(
        "ICU Tracking Test"
    )

    if experiment is None:
        raise RuntimeError("Experiment not found.")

    print("Experiment:", experiment.name)
    print("Experiment ID:", experiment.experiment_id)

    # Step 3: Retrieve latest run.
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No MLflow runs found.")

    run = runs[0]

    print("\nRun ID:", run.info.run_id)
    print("Run status:", run.info.status)

    if run.info.status != "FINISHED":
        raise AssertionError("Run did not finish successfully.")

    print("Run verification: PASS")

    # Step 4: Verify parameters.
    expected_params = {
        "n_estimators": "100",
        "max_depth": "5",
        "random_state": "42",
    }

    print("\n========== PARAMETER VERIFICATION ==========")

    for key, expected in expected_params.items():

        actual = run.data.params.get(key)

        print(f"{key}: {actual}")

        if actual != expected:
            raise AssertionError(
                f"Parameter mismatch: {key}"
            )

    print("Parameters: PASS")

    # Step 5: Verify metrics.
    expected_metrics = [
        "accuracy",
        "auroc",
        "auprc",
        "recall",
        "brier_score",
    ]

    print("\n========== METRIC VERIFICATION ==========")

    for metric in expected_metrics:

        if metric not in run.data.metrics:
            raise AssertionError(
                f"Metric missing: {metric}"
            )

        print(f"{metric}: {run.data.metrics[metric]}")

    print("Metrics: PASS")

    # Step 6: Verify feature schema.
    print("\n========== FEATURE SCHEMA VERIFICATION ==========")

    schema_path = client.download_artifacts(
        run.info.run_id,
        "feature_schema.json",
    )

    with open(schema_path, "r", encoding="utf-8") as file:
        schema = json.load(file)

    feature_columns = schema.get("feature_columns")

    expected_features = [
        "heart_rate_mean",
        "systolic_bp_mean",
    ]

    print("Saved features:", feature_columns)

    if feature_columns != expected_features:
        raise AssertionError("Feature schema mismatch.")

    print("Feature schema: PASS")

    # Step 7: Find the Logged Model.
    print("\n========== LOGGED MODEL VERIFICATION ==========")

    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"source_run_id = '{run.info.run_id}'"
        ),
        max_results=10,
    )

    if not logged_models:
        raise AssertionError("No Logged Model found.")

    logged_model = logged_models[0]

    model_status = getattr(
        logged_model.status,
        "value",
        str(logged_model.status),
    )

    print("Model ID:", logged_model.model_id)
    print("Model name:", logged_model.name)
    print("Model status:", model_status)
    print("Model URI:", logged_model.model_uri)

    if model_status != "READY":
        raise AssertionError("Logged Model is not READY.")

    print("Logged Model: PASS")

    # Step 8: Load the saved model.
    print("\n========== MODEL LOADING TEST ==========")

    loaded_model = mlflow.sklearn.load_model(
        logged_model.model_uri
    )

    print("Saved Random Forest model loaded successfully.")

    # Step 9: Test predictions using the loaded model.
    test_data = [
        [85, 115],
        [110, 140],
    ]

    predictions = loaded_model.predict(test_data)

    print("Test records:", len(test_data))
    print("Predictions:", predictions.tolist())

    if len(predictions) != len(test_data):
        raise AssertionError("Prediction count mismatch.")

    if not set(predictions.tolist()).issubset({0, 1}):
        raise AssertionError("Unexpected prediction labels.")

    print("Model prediction: PASS")

    # Step 10: Final report.
    print("\n========== FINAL VERIFICATION REPORT ==========")

    print("SQLite database: PASS")
    print("Experiment: PASS")
    print("Run status: PASS")
    print("Parameters: PASS")
    print("Metrics: PASS")
    print("Feature schema: PASS")
    print("Logged Model: PASS")
    print("Model loading: PASS")
    print("Model prediction: PASS")

    print("\nMLflow verification completed successfully.")
    print("================================================")


if __name__ == "__main__":
    main()
