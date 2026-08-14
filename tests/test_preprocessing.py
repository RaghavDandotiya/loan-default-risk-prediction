import pandas as pd

from src.preprocessing import create_features, prepare_features


def create_sample_dataframe():
    return pd.DataFrame(
        [
            {
                "LoanID": "TEST001",
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
                "Default": 0,
            }
        ]
    )


def test_create_features_adds_engineered_features():
    df = create_sample_dataframe()

    result = create_features(df)

    assert "LoanToIncomeRatio" in result.columns
    assert "CreditScoreBand" in result.columns
    assert "DTIRiskBand" in result.columns


def test_loan_to_income_ratio_is_calculated_correctly():
    df = create_sample_dataframe()

    result = create_features(df)

    expected_ratio = 150000 / 75000

    assert result["LoanToIncomeRatio"].iloc[0] == expected_ratio


def test_credit_score_band_is_created():
    df = create_sample_dataframe()

    result = create_features(df)

    assert result["CreditScoreBand"].iloc[0] == "500-599"


def test_dti_risk_band_is_created():
    df = create_sample_dataframe()

    result = create_features(df)

    assert result["DTIRiskBand"].iloc[0] == "51-70%"


def test_prepare_features_removes_id_and_target():
    df = create_sample_dataframe()

    X, y = prepare_features(df)

    assert "LoanID" not in X.columns
    assert "Default" not in X.columns
    assert len(X) == len(y)