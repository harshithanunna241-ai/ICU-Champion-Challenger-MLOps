
import pandas as pd

from src.data.split import split_dataset


# Synthetic patient dataset for testing
dataset = pd.DataFrame({
    "subject_id": [
        101, 101, 102, 102, 103,
        104, 104, 105, 106, 106,
        107, 108, 109, 110, 110,
    ],
    "stay_id": [
        301, 302, 303, 304, 305,
        306, 307, 308, 309, 310,
        311, 312, 313, 314, 315,
    ],
    "deterioration_event": [
        0, 1, 0, 0, 1,
        0, 0, 1, 0, 1,
        0, 0, 1, 0, 0,
    ],
})


train_data, test_data = split_dataset(
    dataset,
    test_size=0.2,
    random_state=42,
)

print("\n========== PATIENT-LEVEL SPLIT REPORT ==========")

print("Original records:", len(dataset))
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nUnique training patients:", train_data["subject_id"].nunique())
print("Unique testing patients:", test_data["subject_id"].nunique())

overlap = (
    set(train_data["subject_id"])
    & set(test_data["subject_id"])
)

print("Patients appearing in both sets:", overlap)

print("\nTraining target counts:")
print(train_data["deterioration_event"].value_counts().to_dict())

print("\nTesting target counts:")
print(test_data["deterioration_event"].value_counts().to_dict())

print("\nPatient-level split successful:", len(overlap) == 0)

print("=================================================")
