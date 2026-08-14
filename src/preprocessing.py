from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Loan_default.csv"


def load_data():
    """Load the raw loan default dataset."""
    return pd.read_csv(DATA_PATH)


def create_features(df):
    """Create engineered features used for analysis and modeling."""
    data = df.copy()

    # Loan amount relative to annual income
    data["LoanToIncomeRatio"] = data["LoanAmount"] / data["Income"]

    # Credit score risk bands
    credit_bins = [0, 499, 599, 699, 799, 900]
    credit_labels = [
        "<500",
        "500-599",
        "600-699",
        "700-799",
        "800+",
    ]

    data["CreditScoreBand"] = pd.cut(
        data["CreditScore"],
        bins=credit_bins,
        labels=credit_labels,
        right=True,
        include_lowest=True,
    )

    # Debt-to-income risk bands
    dti_bins = [0, 0.3, 0.5, 0.7, 1.0]
    dti_labels = [
        "<=30%",
        "31-50%",
        "51-70%",
        ">70%",
    ]

    data["DTIRiskBand"] = pd.cut(
        data["DTIRatio"],
        bins=dti_bins,
        labels=dti_labels,
        include_lowest=True,
    )

    return data


def prepare_features(df):
    """Create the feature matrix and target variable."""
    data = create_features(df)

    X = data.drop(columns=["LoanID", "Default"])
    y = data["Default"]

    return X, y


if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)

    print("Dataset shape:", df.shape)
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)