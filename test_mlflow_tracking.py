
"""Test MLflow experiment tracking using synthetic ICU data."""

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from src.orchestration.mlflow_tracking import track_model_run


def main():

    print("\n========== MLFLOW TRACKING TEST ==========")

    # Step 1: Create synthetic data.
    X = [
        [80, 110],
        [90, 120],
        [100, 130],
        [110, 140],
        [120, 150],
        [70, 100],
        [95, 125],
        [105, 135],
    ]

    y = [0, 0, 1, 1, 1, 0, 0, 1]

    feature_columns = [
        "heart_rate_mean",
        "systolic_bp_mean",
    ]

    print("\nSynthetic dataset created.")
    print("Total records:", len(X))
    print("Total features:", len(feature_columns))

    # Step 2: Train Random Forest.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    model.fit(X, y)

    print("\nRandom Forest model training: Successful")

    # Step 3: Define parameters.
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    }

    # These are illustrative test values, not measured model metrics.
    metrics = {
        "accuracy": 0.85,
        "auroc": 0.90,
        "auprc": 0.88,
        "recall": 0.80,
        "brier_score": 0.15,
    }

    # Step 4: Create a persistent test database.
    project_root = Path(__file__).resolve().parent

    tracking_dir = project_root / "artifacts" / "mlflow_test"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    database_path = tracking_dir / "mlflow_test.db"

    tracking_uri = (
        f"sqlite:///{database_path.resolve().as_posix()}"
    )

    print("\nTracking backend: SQLite")
    print("Database:", database_path)

    # Step 5: Track the model.
    result = track_model_run(
        model=model,
        model_name="Random Forest",
        params=params,
        metrics=metrics,
        feature_columns=feature_columns,
        experiment_name="ICU Tracking Test",
        tracking_uri=tracking_uri,
    )

    # Step 6: Display results.
    print("\n========== MLFLOW TRACKING REPORT ==========")

    print("Model name:", result["model_name"])
    print("Run ID:", result["run_id"])
    print("Experiment ID:", result["experiment_id"])
    print("Tracking URI:", result["tracking_uri"])

    print("\nLogged parameters:")

    for key, value in params.items():
        print(f"{key}: {value}")

    print("\nLogged metrics:")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nLogged feature columns:")

    for feature in feature_columns:
        print(feature)

    print("\nMLflow tracking completed successfully.")

    print("============================================")


if __name__ == "__main__":
    main()
