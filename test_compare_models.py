
import pandas as pd

from src.data.split import split_dataset

from src.models.train_logistic import (
    train_logistic_model,
)

from src.models.train_random_forest import (
    train_random_forest_model,
)

from src.evaluation.compare import compare_models


# Generate synthetic patient records
records = []

for i in range(20):
    records.append({
        "subject_id": 101 + i,
        "stay_id": 301 + i,
        "prediction_time": "2026-01-01 10:00:00",
        "heart_rate_mean": 80 + (i * 2),
        "heart_rate_max": 95 + (i * 2),
        "heart_rate_count": 3 + (i % 3),
        "systolic_bp_mean": None if i == 5 else 110 + i,
        "temperature_mean": None if i == 8 else 36.5 + (i * 0.05),
        "spo2_mean": 96 + (i % 4),
        "deterioration_event": i % 2,
    })

dataset = pd.DataFrame(records)


# Split data by patient
train_data, test_data = split_dataset(
    dataset=dataset,
    test_size=0.2,
    random_state=42,
)


# Train Logistic Regression
logistic_model, logistic_features = train_logistic_model(
    train_data
)


# Train Random Forest
random_forest_model, random_forest_features = (
    train_random_forest_model(train_data)
)


# Compare both models
comparison = compare_models(
    champion_model=logistic_model,
    challenger_model=random_forest_model,
    test_data=test_data,
    champion_features=logistic_features,
    challenger_features=random_forest_features,
)


print("\n========== CHAMPION–CHALLENGER REPORT ==========")

print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nModel comparison:")
print(comparison.to_string(index=False))

print("\nPatient overlap:")
print(
    set(train_data["subject_id"])
    & set(test_data["subject_id"])
)

print("\nChampion–Challenger comparison completed.")
print("================================================")
