from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocessing import load_data, prepare_features
from train import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


def train_random_forest():
    """Train and save a Random Forest classification pipeline."""

    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model_pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training Random Forest...")

    model_pipeline.fit(X_train, y_train)

    model_path = MODEL_DIR / "loan_default_random_forest.pkl"

    joblib.dump(model_pipeline, model_path)

    print("Random Forest training completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_random_forest()