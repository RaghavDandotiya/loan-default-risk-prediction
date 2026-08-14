# Loan Default Risk Prediction

An end-to-end Machine Learning project that predicts whether a borrower is likely to default on a loan based on financial, credit, employment, and demographic information.

The project covers the complete ML workflow including data analysis, feature engineering, preprocessing, model training, probability-based evaluation, threshold optimization, model comparison, prediction, and automated testing.

---

## Project Overview

Loan default prediction is a classification problem where the objective is to identify borrowers who are at higher risk of defaulting on their loans.

The dataset contains information related to:

- Borrower demographics
- Income
- Loan amount
- Credit score
- Employment history
- Credit lines
- Interest rate
- Loan term
- Debt-to-income ratio
- Education
- Employment type
- Marital status
- Mortgage status
- Dependents
- Loan purpose
- Co-signer information

The target variable is:

- `Default = 0` → No default
- `Default = 1` → Default

---

## Dataset

The dataset contains:

- **255,347 records**
- **18 original columns**
- **0 missing values**
- **0 duplicate rows**
- **0 duplicate LoanIDs**

### Target Distribution

| Default | Count | Percentage |
|--------:|------:|-----------:|
| 0 | 225,694 | 88.39% |
| 1 | 29,653 | 11.61% |

The target variable is imbalanced, with approximately 11.61% of the borrowers belonging to the default class.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Target distribution
- Numerical feature distributions
- Categorical feature distributions
- Default rate across categorical variables
- Relationships between numerical variables and default
- Potential zero-value patterns
- Credit score risk patterns
- Debt-to-income risk patterns
- Loan-to-income relationship

### Important Findings

`MonthsEmployed = 0` showed a higher default rate:

| Employment History | Default Rate |
|--------------------|-------------:|
| MonthsEmployed > 0 | 11.56% |
| MonthsEmployed = 0 | 18.14% |

Some categorical variables also showed differences in default rates.

For example:

- Unemployed borrowers had a default rate of approximately **13.55%**
- Full-time employees had a default rate of approximately **9.46%**
- Borrowers without dependents had a default rate of approximately **12.72%**
- Borrowers with dependents had a default rate of approximately **10.50%**
- Borrowers without a co-signer had a default rate of approximately **12.87%**
- Borrowers with a co-signer had a default rate of approximately **10.36%**

---

## Feature Engineering

Three additional features were created.

### 1. Loan-to-Income Ratio

```text
LoanToIncomeRatio = LoanAmount / Income

This feature measures the loan amount relative to the borrower's income.

The correlation between LoanToIncomeRatio and Default was approximately:

0.1790
2. Credit Score Band

Credit scores were grouped into risk-oriented categories:

<500
500-599
600-699
700-799
800+
3. DTI Risk Band

The Debt-to-Income Ratio was grouped into:

<=30%
31-50%
51-70%
>70%

These engineered features were included in the modeling pipeline.

Features Used

The final candidate modeling feature set contains 19 features.

Numerical Features
Age
Income
LoanAmount
CreditScore
MonthsEmployed
NumCreditLines
InterestRate
LoanTerm
DTIRatio
LoanToIncomeRatio
Categorical Features
Education
EmploymentType
MaritalStatus
HasMortgage
HasDependents
LoanPurpose
HasCoSigner
CreditScoreBand
DTIRiskBand

LoanID was excluded because it is an identifier and does not provide useful predictive information.

Machine Learning Models

Three classification models were trained and evaluated:

Logistic Regression
Random Forest
Gradient Boosting

The dataset was split into:

80% training data: 204,277 samples
20% testing data: 51,070 samples
Model Evaluation

Because the target variable is imbalanced, accuracy alone is not sufficient for evaluating the models.

The following metrics were considered:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
PR-AUC

Probability-based evaluation and threshold optimization were also performed.

Model Comparison

The best threshold for each model was selected based on F1 Score.

Model	Best Threshold	Accuracy	Precision	Recall	F1
Logistic Regression	0.20	0.8260	0.3214	0.4483	0.3744
Gradient Boosting	0.20	0.8272	0.3211	0.4384	0.3707
Random Forest	0.50	0.7436	0.2524	0.6157	0.3581
Model Selection

Based on the evaluated F1 score, Logistic Regression achieved the highest F1 score:

F1 Score: 0.3744

However, model selection depends on the business objective.

If identifying as many potential defaulters as possible is the primary objective, Random Forest provides the highest recall:

Recall: 0.6157

For the final prediction pipeline, Logistic Regression was selected because it achieved the best F1 score among the evaluated models.

Threshold Optimization

The default classification threshold was not treated as fixed at 0.50.

For Logistic Regression:

Threshold	Precision	Recall	F1
0.20	0.3214	0.4483	0.3744
0.25	0.3738	0.3328	0.3521
0.30	0.4329	0.2485	0.3158
0.35	0.4805	0.1809	0.2629
0.40	0.5311	0.1354	0.2158
0.45	0.5785	0.0969	0.1661
0.50	0.6134	0.0693	0.1245

The optimized threshold of 0.20 improved the F1 score from 0.1245 at the default threshold to 0.3744.

This demonstrates why probability thresholds are important in imbalanced classification problems.

Prediction Pipeline

A reusable prediction script is included in:

src/predict.py

The pipeline accepts borrower and loan information and returns:

Default probability
Binary prediction
Risk level

Example:

Default Probability: 5.08%
Prediction: 0
Risk Level: Lower Risk

The current prediction threshold is 0.20.

Project Structure

loan-default-risk-prediction/
│
├── data/
│   ├── raw/
│   │   └── Loan_default.csv
│   │
│   └── processed/
│       ├── threshold_results.csv
│       ├── random_forest_threshold_results.csv
│       ├── gradient_boosting_threshold_results.csv
│       └── model_comparison.csv
│
├── models/
│   ├── loan_default_logistic_regression.pkl
│   ├── loan_default_random_forest.pkl
│   ├── loan_default_gradient_boosting.pkl
│   └── gradient_boosting_preprocessor.pkl
│
├── notebooks/
│   └── 01_loan_default_eda.ipynb
│
├── sql/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── random_forest.py
│   ├── evaluate_random_forest.py
│   ├── random_forest_threshold.py
│   ├── gradient_boosting.py
│   ├── evaluate_gradient_boosting.py
│   ├── gradient_boosting_threshold.py
│   ├── model_comparison.py
│   └── predict.py
│
├── tests/
│   ├── test_prediction.py
│   └── test_preprocessing.py
│
├── .gitignore
├── requirements.txt
└── README.md

Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd loan-default-risk-prediction

Create a virtual environment:

Windows

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Running the Project

1. Train Logistic Regression
python src/train.py

2. Evaluate Logistic Regression
python src/evaluate.py

3. Train Random Forest
python src/random_forest.py

4. Evaluate Random Forest
python src/evaluate_random_forest.py

5. Train Gradient Boosting
python src/gradient_boosting.py

6. Evaluate Gradient Boosting
python src/evaluate_gradient_boosting.py

7. Compare Models
python src/model_comparison.py

8. Generate a Prediction
python src/predict.py

Testing

The project includes automated tests using pytest.

Run the complete test suite:

python -m pytest -v

Current test result:

9 passed

The tests cover:

Feature engineering
Loan-to-income ratio calculation
Credit score band creation
DTI risk band creation
Feature preparation
Prediction output validation
Prediction probability validation
Binary prediction validation
Risk-level validation

Technologies Used

Python
Pandas
NumPy
Scikit-learn
SciPy
Matplotlib
Joblib
Pytest
Jupyter Notebook

Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

Binary Classification
Exploratory Data Analysis
Feature Engineering
Categorical Feature Handling
Numerical Feature Processing
Logistic Regression
Random Forest
Gradient Boosting
Class Imbalance
Precision and Recall
F1 Score
ROC-AUC
PR-AUC
Probability-Based Prediction
Threshold Optimization
Model Comparison
Model Persistence
Automated Testing
Reproducible ML Workflow

Future Improvements

Potential future improvements include:

Hyperparameter tuning
Cross-validation
Class-weight optimization
Advanced boosting models
Probability calibration
Explainable AI using SHAP
Model monitoring
REST API deployment
Interactive prediction interface
Dockerization
Cloud deployment

Disclaimer

This project is developed for educational and portfolio purposes.