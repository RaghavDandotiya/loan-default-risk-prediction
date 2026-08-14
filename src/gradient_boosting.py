from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from preprocessing import load_data, prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def train_gradient_boosting():
    """Train and save a HistGradientBoosting classification pipeline."""

    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # HistGradientBoosting requires numeric input,
    # so we first use the existing preprocessing pipeline.
    from train import build_preprocessor

    preprocessor = build_preprocessor(X_train)

    # Transform the data before fitting the gradient boosting model.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    print("Training Gradient Boosting...")

    model.fit(X_train_processed, y_train)

    model_path = MODEL_DIR / "loan_default_gradient_boosting.pkl"
    preprocessor_path = MODEL_DIR / "gradient_boosting_preprocessor.pkl"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print("Gradient Boosting training completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Model saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")


if __name__ == "__main__":
    train_gradient_boosting()