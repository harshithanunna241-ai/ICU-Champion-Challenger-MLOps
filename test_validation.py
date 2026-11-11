
import pandas as pd

from src.data.validate import validate_records

# Step 1: Create sample ICU patient data
data = {
    "patient_id": [101, 102, 103, 104],
    "heart_rate": [80, 95, None, 110],
    "temperature": [36.5, 37.0, 38.2, 37.5]
}

# Step 2: Convert the data into a DataFrame
df = pd.DataFrame(data)

# Step 3: Validate the dataset
result = validate_records(
    df,
    required_columns=[
        "patient_id",
        "heart_rate",
        "temperature"
    ]
)

# Step 4: Display the validation report
print("\n========== DATA VALIDATION REPORT ==========")

for key, value in result.items():
    print(f"{key}: {value}")

print("============================================")
