
import pandas as pd

from src.data.split import split_dataset
from src.models.train_logistic import (
    train_logistic_model,
    evaluate_logistic_model,
)


# --------------------------------------------------
# STEP 1: Create synthetic patient dataset
# --------------------------------------------------

records = []

for i in range(20):

    records.append({
        "subject_id": 101 + i,
        "stay_id": 301 + i,
        "prediction_time": "2026-01-01 10:00:00",

        "heart_rate_mean": 80 + (i * 2),
        "heart_rate_max": 95 + (i * 2),
        "heart_rate_count": 3 + (i % 3),

        "systolic_bp_mean": (
            None if i == 5 else 110 + i
        ),

        "temperature_mean": (
            None if i == 8 else 36.5 + (i * 0.05)
        ),

        "spo2_mean": 96 + (i % 4),

        "deterioration_event": i % 2,
    })


dataset = pd.DataFrame(records)


# --------------------------------------------------
# STEP 2: Split by patient
# --------------------------------------------------

train_data, test_data = split_dataset(
    dataset=dataset,
    test_size=0.2,
    random_state=42,
)


# --------------------------------------------------
# STEP 3: Train Logistic Regression
# --------------------------------------------------

model, feature_columns = train_logistic_model(train_data)


# --------------------------------------------------
# STEP 4: Evaluate model
# --------------------------------------------------

metrics = evaluate_logistic_model(
    model=model,
    test_data=test_data,
    feature_columns=feature_columns,
)


# --------------------------------------------------
# STEP 5: Display results
# --------------------------------------------------

print("\n========== LOGISTIC REGRESSION REPORT ==========")

print("Total records:", len(dataset))
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nFeatures used:")
print(feature_columns)

print("\nModel training: Successful")

print("\nEvaluation metrics:")

for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

print("\nPatient overlap:")

overlap = (
    set(train_data["subject_id"])
    & set(test_data["subject_id"])
)

print(overlap)

print("\nLogistic Regression test completed successfully.")

print("================================================")
