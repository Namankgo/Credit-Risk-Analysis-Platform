from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("CREDIT_RISK_API_URL", "http://localhost:8000")
st.set_page_config(page_title="Credit Risk Analysis", page_icon="CR", layout="wide")


def api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def api_get(path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{api_url()}{path}", timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


def api_post(path: str, payload: dict[str, Any] | None = None, files: dict | None = None, data: dict | None = None):
    try:
        response = requests.post(f"{api_url()}{path}", json=payload, files=files, data=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            try:
                detail = f" - {exc.response.json().get('detail')}"
            except Exception:
                detail = f" - {exc.response.text}"
        st.error(f"API request failed: {exc}{detail}")
        return None


def draw_analytics(result: dict[str, Any]) -> None:
    metrics = result.get("metrics", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ROC AUC", f"{metrics.get('roc_auc') or 0:.3f}")
    col2.metric("Accuracy", f"{metrics.get('accuracy') or 0:.3f}")
    col3.metric("Precision", f"{metrics.get('precision') or 0:.3f}")
    col4.metric("Recall", f"{metrics.get('recall') or 0:.3f}")
    col5.metric("F1", f"{metrics.get('f1') or 0:.3f}")
    left, right = st.columns(2)
    importance = pd.DataFrame(result.get("feature_importance", []))
    if not importance.empty:
        left.markdown("#### Feature Importance")
        left.bar_chart(importance.head(12).set_index("feature")["importance"].sort_values())
    right.markdown("#### Risk Distribution")
    right.bar_chart(pd.Series(result.get("risk_distribution", {})))
    roc = metrics.get("roc_curve", {})
    if roc.get("fpr"):
        st.markdown("#### ROC Curve")
        st.line_chart(pd.DataFrame({"False Positive Rate": roc["fpr"], "True Positive Rate": roc["tpr"]}), x="False Positive Rate", y="True Positive Rate")
    st.markdown("#### Confusion Matrix")
    st.dataframe(pd.DataFrame(metrics.get("confusion_matrix", [[0, 0], [0, 0]]), index=["Actual Paid", "Actual Default"], columns=["Pred Paid", "Pred Default"]), use_container_width=True)


def manual_record_form() -> dict[str, Any]:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 90, 38)
        annual_income = st.number_input("Annual income", 0, value=72000, step=1000)
        loan_amount = st.number_input("Loan amount", 0, value=18000, step=500)
        credit_score = st.number_input("Credit score", 300, 850, 680)
    with col2:
        loan_term_months = st.selectbox("Loan term", [36, 48, 60, 72], index=2)
        interest_rate = st.number_input("Interest rate", 0.0, 40.0, 12.5, 0.1)
        employment_length_years = st.number_input("Employment length", 0, 45, 5)
        debt_to_income = st.slider("Debt-to-income", 0.0, 1.2, 0.34, 0.01)
    with col3:
        delinquencies = st.number_input("Delinquencies 2 yrs", 0, 20, 0)
        utilization = st.slider("Revolving utilization", 0.0, 1.5, 0.47, 0.01)
        inquiries = st.number_input("Inquiries last 6m", 0, 20, 1)
        open_lines = st.number_input("Open credit lines", 0, 60, 9)
    loan_purpose = st.selectbox("Loan purpose", ["debt_consolidation", "credit_card", "home_improvement", "medical", "small_business", "auto"])
    home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
    return {
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "employment_length_years": employment_length_years,
        "debt_to_income": debt_to_income,
        "credit_score": credit_score,
        "delinquencies_2yrs": delinquencies,
        "revolving_utilization": utilization,
        "inquiries_last_6m": inquiries,
        "open_credit_lines": open_lines,
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership,
    }


def show_prediction(prediction: dict[str, Any]) -> None:
    st.subheader(f"{prediction['risk_category']} Risk")
    st.write(f"Probability of Default: **{prediction['probability_default']:.2%}**")
    st.write(f"Decision Suggestion: **{prediction['decision']}**")
    explanation = prediction.get("explanation", {})
    st.write(explanation.get("summary"))
    reasons = explanation.get("business_reasons", [])
    if reasons:
        st.markdown("#### Business Reasons")
        for reason in reasons:
            st.write(f"- {reason}")
    factors = pd.DataFrame(explanation.get("top_factors", []))
    if not factors.empty:
        st.markdown("#### Top Model Factors")
        st.dataframe(factors, use_container_width=True, hide_index=True)


st.sidebar.title("Credit Risk")
st.sidebar.text_input("API base URL", DEFAULT_API_URL, key="api_url")
health = api_get("/health")
if health:
    st.sidebar.success("API connected")
else:
    st.sidebar.warning("Start FastAPI on port 8000")
capabilities = api_get("/capabilities") if health else {}

st.title("Credit Risk Analysis Platform")
st.caption("Train models, tune risk appetite, score borrowers, and explain credit decisions.")

train_tab, predict_tab, analytics_tab, history_tab = st.tabs(["Train", "Predict", "Analytics", "History"])

with train_tab:
    profile = api_get("/datasets/sample")
    if profile:
        st.dataframe(pd.DataFrame(profile["preview"]), use_container_width=True)
    uploaded_train = st.file_uploader("Upload training CSV", type=["csv"])
    dataset_path = "data/sample_credit_data.csv"
    if uploaded_train:
        upload = api_post("/datasets/upload", files={"file": (uploaded_train.name, uploaded_train.getvalue(), "text/csv")})
        if upload:
            dataset_path = upload["dataset_path"]
            st.dataframe(pd.DataFrame(upload["preview"]), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox("Model", ["logistic_regression", "random_forest", "xgboost"])
        target_column = st.text_input("Target column", "default")
    with col2:
        low_threshold = st.slider("Low risk cutoff", 0.01, 0.90, 0.30, 0.01)
        high_threshold = st.slider("High risk cutoff", 0.10, 0.99, 0.60, 0.01)
    with col3:
        decision_threshold = st.slider("Model decision threshold", 0.05, 0.95, 0.50, 0.01)
        test_size = st.slider("Test set size", 0.10, 0.50, 0.25, 0.05)
    xgboost_missing = model_type == "xgboost" and capabilities and not capabilities.get("xgboost_available")
    if xgboost_missing:
        st.warning("Install `requirements-ml-extra.txt` to enable XGBoost.")
    if st.button("Train Model", type="primary", disabled=low_threshold >= high_threshold or xgboost_missing):
        result = api_post("/train", payload={"dataset_path": dataset_path, "target_column": target_column, "model_type": model_type, "low_threshold": low_threshold, "high_threshold": high_threshold, "decision_threshold": decision_threshold, "test_size": test_size})
        if result:
            st.success(f"Saved {result['model_name']}")
            draw_analytics(result)

with predict_tab:
    models = [model["model_name"] for model in (api_get("/models") or {"models": []})["models"]]
    selected_model = st.selectbox("Model", models, index=0 if models else None, placeholder="Train a model first")
    low = st.slider("Prediction low cutoff", 0.01, 0.90, 0.30, 0.01)
    high = st.slider("Prediction high cutoff", 0.10, 0.99, 0.60, 0.01)
    record = manual_record_form()
    if st.button("Score Borrower", type="primary", disabled=not models or low >= high):
        result = api_post("/predict", payload={"records": [record], "model_name": selected_model, "low_threshold": low, "high_threshold": high})
        if result:
            show_prediction(result["predictions"][0])
    uploaded_score = st.file_uploader("Score CSV batch", type=["csv"], key="score_csv")
    if st.button("Score CSV", disabled=uploaded_score is None or not models or low >= high):
        result = api_post("/predict/file", files={"file": (uploaded_score.name, uploaded_score.getvalue(), "text/csv")}, data={"model_name": selected_model, "low_threshold": low, "high_threshold": high})
        if result:
            st.dataframe(pd.DataFrame(result["predictions"])[["probability_default", "risk_category", "decision"]], use_container_width=True)

with analytics_tab:
    models = [model["model_name"] for model in (api_get("/models") or {"models": []})["models"]]
    selected = st.selectbox("Analytics model", models, index=0 if models else None, placeholder="Train a model first")
    if selected:
        result = api_get(f"/analytics/model/{selected}")
        if result:
            draw_analytics(result)

with history_tab:
    left, right = st.columns(2)
    with left:
        runs = api_get("/history/training-runs")
        if runs:
            st.dataframe(pd.DataFrame(runs["training_runs"]), use_container_width=True, hide_index=True)
    with right:
        predictions = api_get("/history/predictions")
        if predictions:
            st.dataframe(pd.DataFrame(predictions["predictions"]), use_container_width=True, hide_index=True)

