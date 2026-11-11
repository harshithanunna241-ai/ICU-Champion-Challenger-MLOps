
import pandas as pd

from src.data.split import split_dataset
from src.models.train_random_forest import (
    train_random_forest_model,
    evaluate_random_forest_model,
)

# Create 20 synthetic patients
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

# Split patients into training and testing datasets
train_data, test_data = split_dataset(
    dataset=dataset,
    test_size=0.2,
    random_state=42,
)

# Train Random Forest
model, feature_columns = train_random_forest_model(train_data)

# Evaluate Random Forest
metrics = evaluate_random_forest_model(
    model=model,
    test_data=test_data,
    feature_columns=feature_columns,
)

print("\n========== RANDOM FOREST REPORT ==========")
print("Total records:", len(dataset))
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nFeatures used:")
print(feature_columns)

print("\nModel training: Successful")

print("\nEvaluation metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

# Check patient-level separation
overlap = (
    set(train_data["subject_id"])
    & set(test_data["subject_id"])
)

print("\nPatient overlap:")
print(overlap)

print("\nRandom Forest test completed successfully.")
print("==========================================")
