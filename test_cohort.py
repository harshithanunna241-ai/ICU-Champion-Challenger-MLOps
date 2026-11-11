
import pandas as pd

from src.data.cohort import select_icu_cohort

# Create synthetic ICU admission records
data = {
    "subject_id": [101, 102, 103, 104, 105],
    "hadm_id": [201, 202, 203, 204, 205],
    "stay_id": [301, 302, 303, 304, 305],
    "age": [45, 65, 16, 55, 70],
    "intime": [
        "2026-01-01 08:00:00",
        "2026-01-02 10:00:00",
        "2026-01-03 09:00:00",
        "invalid_date",
        "2026-01-05 08:00:00"
    ],
    "outtime": [
        "2026-01-03 08:00:00",
        "2026-01-04 10:00:00",
        "2026-01-04 09:00:00",
        "2026-01-06 09:00:00",
        "2026-01-05 07:00:00"
    ]
}

stays = pd.DataFrame(data)

# Select eligible adult ICU stays
cohort = select_icu_cohort(
    stays,
    minimum_age=18
)

# Display cohort report
print("\n========== ICU COHORT SELECTION REPORT ==========")

print("Original ICU stays:", len(stays))
print("Eligible ICU stays:", len(cohort))
print("Excluded ICU stays:", len(stays) - len(cohort))

print("\nSelected patients:")
print(
    cohort[
        ["subject_id", "stay_id", "age", "icu_los_hours"]
    ]
)

print("=================================================")
