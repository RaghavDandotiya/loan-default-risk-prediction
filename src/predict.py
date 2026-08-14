from pathlib import Path

import joblib
import pandas as pd

try:
    from src.preprocessing import create_features
except ModuleNotFoundError:
    from preprocessing import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "loan_default_logistic_regression.pkl"
)


def load_model():
    """Load the trained Logistic Regression pipeline."""
    return joblib.load(MODEL_PATH)


def predict_default(customer_data, threshold=0.20):
    """
    Predict loan default risk for a single customer.

    Parameters
    ----------
    customer_data : dict
        Customer and loan information.

    threshold : float
        Probability threshold used to classify default.

    Returns
    -------
    dict
        Default probability and prediction.
    """

    model = load_model()

    input_df = pd.DataFrame([customer_data])

    input_df = create_features(input_df)

    probability = model.predict_proba(input_df)[0, 1]

    prediction = int(probability >= threshold)

    return {
        "default_probability": round(
            float(probability),
            4,
        ),
        "prediction": prediction,
        "risk_level": (
            "High Risk"
            if prediction == 1
            else "Lower Risk"
        ),
    }


if __name__ == "__main__":

    sample_customer = {
        "Age": 45,
        "Income": 75000,
        "LoanAmount": 150000,
        "CreditScore": 580,
        "MonthsEmployed": 48,
        "NumCreditLines": 3,
        "InterestRate": 15.5,
        "LoanTerm": 36,
        "DTIRatio": 0.65,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Married",
        "HasMortgage": "Yes",
        "HasDependents": "Yes",
        "LoanPurpose": "Home",
        "HasCoSigner": "Yes",
    }

    result = predict_default(
        sample_customer,
        threshold=0.20,
    )

    print("\nLoan Default Prediction")
    print("=" * 40)
    print(
        f"Default Probability: "
        f"{result['default_probability']:.2%}"
    )
    print(f"Prediction: {result['prediction']}")
    print(f"Risk Level: {result['risk_level']}")