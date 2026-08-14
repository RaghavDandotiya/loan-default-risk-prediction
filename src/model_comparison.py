from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "processed"


def load_threshold_results(filename):
    """Load saved threshold evaluation results."""
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    return pd.read_csv(path)


def build_model_comparison():
    """Build a consolidated comparison of all trained models."""

    logistic_results = load_threshold_results(
        "threshold_results.csv"
    )

    random_forest_results = load_threshold_results(
        "random_forest_threshold_results.csv"
    )

    gradient_boosting_results = load_threshold_results(
        "gradient_boosting_threshold_results.csv"
    )

    model_results = []

    for model_name, results in [
        ("Logistic Regression", logistic_results),
        ("Random Forest", random_forest_results),
        ("Gradient Boosting", gradient_boosting_results),
    ]:
        best_row = results.loc[
            results["F1"].idxmax()
        ]

        model_results.append(
            {
                "Model": model_name,
                "Best Threshold": best_row["Threshold"],
                "Accuracy": best_row["Accuracy"],
                "Precision": best_row["Precision"],
                "Recall": best_row["Recall"],
                "F1": best_row["F1"],
            }
        )

    comparison = pd.DataFrame(model_results)

    comparison = comparison.sort_values(
        by="F1",
        ascending=False,
    )

    comparison_path = (
        RESULTS_DIR / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print("\nModel Comparison")
    print("=" * 80)
    print(
        comparison.to_string(
            index=False,
            float_format="{:.4f}".format,
        )
    )

    print(
        f"\nComparison saved to: {comparison_path}"
    )


if __name__ == "__main__":
    build_model_comparison()