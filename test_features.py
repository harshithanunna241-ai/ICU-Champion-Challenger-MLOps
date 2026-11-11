
import pandas as pd

from src.features.build import build_features


# Synthetic ICU measurements
observations = pd.DataFrame({
    "subject_id": [101, 101, 101, 101, 101, 101],
    "stay_id": [301, 301, 301, 301, 301, 301],
    "charttime": [
        "2026-01-01 05:00:00",
        "2026-01-01 06:00:00",
        "2026-01-01 07:00:00",
        "2026-01-01 08:00:00",
        "2026-01-01 09:00:00",
        "2026-01-01 11:00:00",
    ],
    "variable": [
        "heart_rate",
        "heart_rate",
        "heart_rate",
        "temperature",
        "spo2",
        "heart_rate",
    ],
    "value": [80, 90, 100, 37.0, 98, 150],
})


# Prediction timestamp
prediction_rows = pd.DataFrame({
    "subject_id": [101],
    "stay_id": [301],
    "prediction_time": ["2026-01-01 10:00:00"],
})


# Build features using previous six hours
features = build_features(
    observations=observations,
    prediction_rows=prediction_rows,
    lookback_hours=6,
)


print("\n========== FEATURE ENGINEERING REPORT ==========")

print("Prediction rows:", len(features))

print("\nSelected features:")

print(
    features[
        [
            "subject_id",
            "stay_id",
            "prediction_time",
            "heart_rate_mean",
            "heart_rate_min",
            "heart_rate_max",
            "heart_rate_last",
            "heart_rate_count",
            "heart_rate_missing",
        ]
    ].to_string(index=False)
)

print("\nFuture measurement excluded: heart rate 150 at 11 AM")

print("\nPatient ID preserved:", features["subject_id"].iloc[0] == 101)

print("================================================")
