
import yaml
from pathlib import Path

config_path = Path("configs/evaluation_policy.yaml")

with config_path.open("r", encoding="utf-8") as file:
    policy = yaml.safe_load(file)

print("\n========== EVALUATION CONFIG REPORT ==========")

print("Configuration loaded successfully.")

for key, value in policy.items():
    print(f"{key}: {value}")

required_keys = [
    "minimum_auprc",
    "minimum_recall",
    "maximum_brier_score",
    "minimum_auprc_improvement",
    "maximum_recall_drop",
]

missing_keys = [
    key for key in required_keys if key not in policy
]

print("\nMissing configuration keys:", missing_keys)
print("Configuration valid:", len(missing_keys) == 0)

print("==============================================")
