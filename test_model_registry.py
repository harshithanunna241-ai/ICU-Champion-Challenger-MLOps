
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.registry.model_registry import (
    register_model,
    list_registered_models,
    load_registered_model,
)


# Create a small model for registry testing
X = np.array([
    [80, 110],
    [90, 120],
    [100, 130],
    [110, 140],
    [120, 150],
    [70, 100],
])

y = np.array([0, 0, 1, 1, 1, 0])

model = RandomForestClassifier(
    n_estimators=10,
    random_state=42,
)

model.fit(X, y)

metrics = {
    "accuracy": 0.85,
    "auprc": 0.90,
    "recall": 0.80,
    "brier_score": 0.15,
}

feature_columns = [
    "heart_rate_mean",
    "systolic_bp_mean",
]


# Use a temporary directory for this test
with tempfile.TemporaryDirectory() as temp_dir:

    # Register the model
    record = register_model(
        model=model,
        model_name="Random Forest",
        version="v1",
        metrics=metrics,
        feature_columns=feature_columns,
        stage="candidate",
        registry_dir=temp_dir,
    )

    print("\n========== MODEL REGISTRY REPORT ==========")

    print("\nRegistered model:")
    print("Model name:", record["model_name"])
    print("Version:", record["version"])
    print("Stage:", record["stage"])
    print("Metrics:", record["metrics"])
    print("Features:", record["feature_columns"])
    print("Artifact:", record["artifact_path"])

    # List registered models
    registered_models = list_registered_models(temp_dir)

    print("\nTotal registered models:", len(registered_models))

    # Load the saved model
    loaded_model = load_registered_model(
        model_name="Random Forest",
        version="v1",
        registry_dir=temp_dir,
    )

    predictions = loaded_model.predict(X)

    print("\nModel loaded successfully.")
    print("Predictions:", predictions.tolist())

    print("\nRegistry test completed successfully.")
    print("==========================================")
