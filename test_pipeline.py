
import pandas as pd

from src.features.build import build_features
from src.features.dataset import create_model_dataset
from src.data.split import split_dataset


# --------------------------------------------------
# STEP 1: Generate synthetic ICU observations
# --------------------------------------------------

observation_records = []
prediction_records = []
label_records = []

for i in range(10):

    subject_id = 101 + i
    stay_id = 301 + i

    # Three measurements within the previous six hours
    measurements = [
        ("2026-01-01 05:00:00", 80 + i),
        ("2026-01-01 07:00:00", 90 + i),
        ("2026-01-01 09:00:00", 100 + i),

        # Future measurement: must not enter the features
        ("2026-01-01 11:00:00", 150 + i),
    ]

    for charttime, heart_rate in measurements:
        observation_records.append({
            "subject_id": subject_id,
            "stay_id": stay_id,
            "charttime": charttime,
            "variable": "heart_rate",
            "value": heart_rate,
        })

    prediction_records.append({
        "subject_id": subject_id,
        "stay_id": stay_id,
        "prediction_time": "2026-01-01 10:00:00",
    })

    label_records.append({
        "stay_id": stay_id,
        "prediction_time": "2026-01-01 10:00:00",
        "deterioration_event": i % 2,
    })


observations = pd.DataFrame(observation_records)
prediction_rows = pd.DataFrame(prediction_records)
labels = pd.DataFrame(label_records)


# --------------------------------------------------
# STEP 2: Build features
# --------------------------------------------------

features = build_features(
    observations=observations,
    prediction_rows=prediction_rows,
    lookback_hours=6,
)


# --------------------------------------------------
# STEP 3: Combine features and labels
# --------------------------------------------------

model_dataset = create_model_dataset(
    features=features,
    labels=labels,
)


# --------------------------------------------------
# STEP 4: Split data by patient
# --------------------------------------------------

train_data, test_data = split_dataset(
    dataset=model_dataset,
    test_size=0.2,
    random_state=42,
    group_column="subject_id",
    target_column="deterioration_event",
)


# --------------------------------------------------
# STEP 5: Verify patient separation
# --------------------------------------------------

train_patients = set(train_data["subject_id"])
test_patients = set(test_data["subject_id"])

overlap = train_patients.intersection(test_patients)


# --------------------------------------------------
# STEP 6: Display pipeline report
# --------------------------------------------------

print("\n========== COMPLETE DATA PIPELINE REPORT ==========")

print("Original observations:", len(observations))
print("Prediction windows:", len(prediction_rows))
print("Generated feature rows:", len(features))
print("Combined model rows:", len(model_dataset))

print("\nTraining records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining patients:", len(train_patients))
print("Testing patients:", len(test_patients))

print("\nPatients appearing in both sets:", overlap)

print("\nTraining target distribution:")
print(train_data["deterioration_event"].value_counts().to_dict())

print("\nTesting target distribution:")
print(test_data["deterioration_event"].value_counts().to_dict())

print("\nPatient-level separation successful:", len(overlap) == 0)

print("\nFinal model dataset preview:")
print(model_dataset.head().to_string(index=False))

print("====================================================")
