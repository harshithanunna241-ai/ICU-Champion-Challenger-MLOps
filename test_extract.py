
import pandas as pd
from pathlib import Path

from src.data.extract import extract_dataset

# Create a temporary synthetic dataset
sample_data = {
    "patient_id": [101, 102, 103, 104],
    "heart_rate": [80, 95, 110, 75],
    "temperature": [36.5, 37.0, 38.2, 37.5]
}

df = pd.DataFrame(sample_data)

# Save synthetic data as a CSV file
file_path = Path("sample_icu_data.csv")
df.to_csv(file_path, index=False)

# Extract the dataset
extracted_data = extract_dataset(file_path)

print("\n========== DATA EXTRACTION REPORT ==========")
print("Extraction successful!")
print("Total records:", len(extracted_data))
print("Total columns:", len(extracted_data.columns))
print("Column names:", list(extracted_data.columns))
print("\nExtracted data:")
print(extracted_data)
print("============================================")
