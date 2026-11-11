
"""Tests for ICU prediction API schemas."""

from pydantic import ValidationError

from src.api.schemas import PredictionRequest, PredictionResponse


def main():
    print("\n========== API SCHEMA TEST ==========\n")

    print("Testing valid prediction request...")

    request = PredictionRequest(
        features={
            "heart_rate_mean": 95.0,
            "spo2_mean": 97.0,
            "respiratory_rate_mean": 20.0
        }
    )

    assert request.model_name == "Champion"
    assert len(request.features) == 3

    print("Valid prediction request: SUCCESS")

    print("\nTesting custom model name...")

    custom_request = PredictionRequest(
        features={"heart_rate_mean": 90.0},
        model_name="Random Forest"
    )

    assert custom_request.model_name == "Random Forest"

    print("Custom model name: SUCCESS")

    print("\nTesting invalid prediction response...")

    try:
        PredictionResponse(
            prediction=2,
            deterioration_probability=0.8,
            model_name="Champion"
        )
        raise AssertionError("Invalid prediction was accepted.")

    except ValidationError:
        print("Invalid prediction rejected: SUCCESS")

    print("\nTesting valid prediction response...")

    response = PredictionResponse(
        prediction=1,
        deterioration_probability=0.82,
        model_name="Champion"
    )

    assert response.prediction == 1
    assert response.deterioration_probability == 0.82

    print("Valid prediction response: SUCCESS")

    print("\n========== API SCHEMA TEST PASSED ==========")


if __name__ == "__main__":
    main()
