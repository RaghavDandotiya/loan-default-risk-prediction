from pathlib import Path

import joblib
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


def evaluate_gradient_boosting():
    """Evaluate the saved Gradient Boosting model."""

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

    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\nGradient Boosting Evaluation")
    print("=" * 55)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix")
    print("=" * 50)
    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    print("\nClassification Report")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    evaluate_gradient_boosting()