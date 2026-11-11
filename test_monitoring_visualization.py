
"""Integration tests for monitoring visualization."""

import tempfile
from pathlib import Path

import pandas as pd

from src.monitoring.visualization import (
    plot_monitoring_trends,
    plot_alert_trends,
)


def create_test_trends():
    """Create synthetic monitoring history."""

    return pd.DataFrame({
        "timestamp": pd.date_range(
            start="2026-11-01",
            periods=5,
            freq="D",
        ),
        "auroc": [0.91, 0.90, 0.89, 0.87, 0.86],
        "auprc": [0.88, 0.87, 0.86, 0.84, 0.82],
        "brier_score": [0.12, 0.13, 0.14, 0.16, 0.18],
        "psi": [0.05, 0.08, 0.12, 0.19, 0.25],
        "feature_drift_count": [0, 0, 1, 2, 3],
        "alert_count": [0, 0, 1, 2, 4],
    })


def main():
    print("\n========== MONITORING VISUALIZATION TEST ==========\n")

    trends = create_test_trends()

    print("Synthetic monitoring history created.")
    print("Monitoring records:", len(trends))

    with tempfile.TemporaryDirectory() as temp_dir:

        output_dir = Path(temp_dir) / "plots"

        print("\nGenerating performance and drift charts...")

        chart_files = plot_monitoring_trends(
            trends=trends,
            output_dir=str(output_dir),
        )

        assert isinstance(chart_files, dict)
        assert "performance_chart" in chart_files
        assert "drift_chart" in chart_files

        for chart_name, chart_path in chart_files.items():
            assert Path(chart_path).exists()
            assert Path(chart_path).stat().st_size > 0

            print(f"{chart_name}: SUCCESS")

        print("\nGenerating alert history chart...")

        alert_chart = plot_alert_trends(
            trends=trends,
            output_dir=str(output_dir),
        )

        assert Path(alert_chart).exists()
        assert Path(alert_chart).stat().st_size > 0

        print("Alert chart: SUCCESS")

        generated_images = list(output_dir.glob("*.png"))

        assert len(generated_images) == 3

        print("\nGenerated chart files:")
        for image_path in generated_images:
            print(image_path.name)

    print("\n========== MONITORING VISUALIZATION TEST PASSED ==========")


if __name__ == "__main__":
    main()
