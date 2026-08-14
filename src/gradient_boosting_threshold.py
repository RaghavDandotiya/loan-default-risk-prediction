from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from preprocessing import load_data, prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "loan_default_gradient_boosting.pkl"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "gradient_boosting_preprocessor.pkl"
)


def evaluate_thresholds():
    """Evaluate Gradient Boosting across multiple probability thresholds."""

    df = load_data()
    X, y = prepare_features(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X_test_processed = preprocessor.transform(X_test)

    # Predicted probability of default
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\nGradient Boosting Probability-Based Evaluation")
    print("=" * 60)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        results.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy_score(
                    y_test,
                    y_pred,
                ),
                "Precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
            }
        )

    threshold_results = pd.DataFrame(results)

    print("\nThreshold Comparison")
    print("=" * 80)
    print(
        threshold_results.to_string(
            index=False,
            float_format="{:.4f}".format,
        )
    )

    # Select threshold with the highest F1 score
    best_f1_row = threshold_results.loc[
        threshold_results["F1"].idxmax()
    ]

    print("\nBest Threshold by F1 Score")
    print("=" * 50)
    print(best_f1_row.to_string())

    best_threshold = best_f1_row["Threshold"]

    y_best_pred = (
        y_prob >= best_threshold
    ).astype(int)

    print("\nConfusion Matrix at Best F1 Threshold")
    print("=" * 50)
    print(
        confusion_matrix(
            y_test,
            y_best_pred,
        )
    )

    print("\nClassification Report at Best F1 Threshold")
    print("=" * 50)
    print(
        classification_report(
            y_test,
            y_best_pred,
            zero_division=0,
        )
    )

    # Save threshold results
    results_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    results_dir.mkdir(exist_ok=True)

    results_path = (
        results_dir
        / "gradient_boosting_threshold_results.csv"
    )

    threshold_results.to_csv(
        results_path,
        index=False,
    )

    print(
        f"\nThreshold results saved to: "
        f"{results_path}"
    )


if __name__ == "__main__":
    evaluate_thresholds()