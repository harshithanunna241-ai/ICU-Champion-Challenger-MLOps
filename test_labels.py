
import pandas as pd

from src.data.labels import create_deterioration_labels


# Synthetic prediction timestamps
prediction_data = {
    "stay_id": [301, 302, 303, 304],
    "prediction_time": [
        "2026-01-01 10:00:00",
        "2026-01-02 10:00:00",
        "2026-01-03 10:00:00",
        "2026-01-04 10:00:00",
    ],
    "observation_end": [
        "2026-01-01 18:00:00",
        "2026-01-02 18:00:00",
        "2026-01-03 18:00:00",
        "2026-01-04 15:00:00",
    ],
}

predictions = pd.DataFrame(prediction_data)


# Synthetic deterioration events
event_data = {
    "stay_id": [301, 302, 303, 304],
    "event_type": [
        "vasopressor_initiation",
        "invasive_mechanical_ventilation",
        "death",
        "death",
    ],
    "event_time": [
        "2026-01-01 13:00:00",  # Within six hours
        "2026-01-02 16:30:00",  # After six hours
        "2026-01-03 10:00:00",  # Exactly at prediction time
        "2026-01-04 14:00:00",  # Window is not fully observable
    ],
}

events = pd.DataFrame(event_data)


# Create deterioration labels
result = create_deterioration_labels(
    prediction_rows=predictions,
    events=events,
    horizon_hours=6,
)


# Display results
print("\n========== DETERIORATION LABEL REPORT ==========")

print("Original prediction windows:", len(predictions))
print("Fully observed windows:", len(result))
print("Positive labels:", result["deterioration_event"].sum())
print(
    "Negative labels:",
    (result["deterioration_event"] == 0).sum()
)

print("\nLabel details:")

print(
    result[
        [
            "stay_id",
            "prediction_time",
            "label_horizon_end",
            "deterioration_event",
            "label_event_type",
        ]
    ].to_string(index=False)
)

print("================================================")
