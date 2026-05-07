from backend.ml_pipeline import predict_credit_risk, train_credit_model


def test_train_and_predict_logistic_regression():
    result = train_credit_model(
        dataset_path="data/sample_credit_data.csv",
        target_column="default",
        model_type="logistic_regression",
        model_name="test_logistic_model.joblib",
    )

    assert result["model_name"] == "test_logistic_model.joblib"
    assert 0 <= result["metrics"]["accuracy"] <= 1

    prediction = predict_credit_risk(
        [
            {
                "age": 38,
                "annual_income": 72000,
                "loan_amount": 18000,
                "loan_term_months": 60,
                "interest_rate": 12.5,
                "employment_length_years": 5,
                "debt_to_income": 0.34,
                "credit_score": 680,
                "delinquencies_2yrs": 0,
                "revolving_utilization": 0.47,
                "inquiries_last_6m": 1,
                "open_credit_lines": 9,
                "loan_purpose": "debt_consolidation",
                "home_ownership": "RENT",
            }
        ],
        model_name=result["model_name"],
    )

    assert prediction["predictions"][0]["decision"] in {"Approve", "Review", "Reject"}

