from src.predict import predict_default


def create_sample_customer():
    return {
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


def test_prediction_returns_required_fields():
    customer = create_sample_customer()

    result = predict_default(customer)

    assert "default_probability" in result
    assert "prediction" in result
    assert "risk_level" in result


def test_prediction_probability_is_valid():
    customer = create_sample_customer()

    result = predict_default(customer)

    assert 0.0 <= result["default_probability"] <= 1.0


def test_prediction_is_binary():
    customer = create_sample_customer()

    result = predict_default(customer)

    assert result["prediction"] in [0, 1]


def test_prediction_risk_level_is_valid():
    customer = create_sample_customer()

    result = predict_default(customer)

    assert result["risk_level"] in [
        "High Risk",
        "Lower Risk",
    ]