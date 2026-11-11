
import pandas as pd

from src.features.dataset import create_model_dataset


# Synthetic engineered features
features = pd.DataFrame({
    "subject_id": [101, 102, 103, 104],
    "stay_id": [301, 302, 303, 304],
    "prediction_time": [
        "2026-01-01 10:00:00",
        "2026-01-02 10:00:00",
        "2026-01-03 10:00:00",
        "2026-01-04 10:00:00",
    ],
    "heart_rate_mean": [90.0, 85.0, 110.0, 75.0],
    "heart_rate_max": [100.0, 95.0, 125.0, 80.0],
    "heart_rate_count": [3, 4, 5, 2],
})


# Synthetic deterioration labels
labels = pd.DataFrame({
    "stay_id": [301, 302, 303, 305],
    "prediction_time": [
        "2026-01-01 10:00:00",
        "2026-01-02 10:00:00",
        "2026-01-03 10:00:00",
        "2026-01-05 10:00:00",
    ],
    "deterioration_event": [1, 0, 1, 0],
})


# Combine features and labels
model_dataset = create_model_dataset(features, labels)


print("\n========== MODEL DATASET REPORT ==========")

print("Feature rows:", len(features))
print("Label rows:", len(labels))
print("Matched model rows:", len(model_dataset))

print("\nCombined model dataset:")

print(
    model_dataset[
        [
            "subject_id",
            "stay_id",
            "prediction_time",
            "heart_rate_mean",
            "heart_rate_max",
            "heart_rate_count",
            "deterioration_event",
        ]
    ].to_string(index=False)
)

print("\nUnique patients:", model_dataset["subject_id"].nunique())

print("\nTarget distribution:")
print(model_dataset["deterioration_event"].value_counts().to_dict())

print("\nMissing values:")
print(model_dataset.isnull().sum().to_dict())

print("\nPatient ID preserved:", "subject_id" in model_dataset.columns)

print("==========================================")
