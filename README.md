# Credit Risk Analysis Platform

A full-stack FastAPI + Streamlit app for training credit-risk models, predicting probability of default, tuning risk thresholds, and explaining lending decisions.

## Features

- Upload customer financial data as CSV or score manual borrower input.
- Preprocess missing values, numeric scaling, and categorical encoding.
- Train Logistic Regression, Random Forest, and optional XGBoost models.
- Predict probability of default (PD), risk category, and decision suggestion.
- Tune Low/Medium/High thresholds for business risk appetite.
- View feature importance, risk distribution, ROC curve, and confusion matrix.
- Explain high-risk decisions with model factors and lending-rule reasons.
- Save trained models under `models/`.
- Store training and prediction history in SQLite.
- Expose FastAPI prediction endpoints.

## Structure

```text
backend/
  config.py        # Paths, thresholds, workspace-safe path handling
  database.py      # SQLite persistence
  main.py          # FastAPI endpoints
  ml_pipeline.py   # Preprocessing, training, prediction, analytics
  schemas.py       # Pydantic request models
frontend/
  app.py           # Streamlit dashboard
data/
  sample_credit_data.csv
models/
notebooks/
tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional XGBoost and SHAP:

```bash
pip install -r requirements-ml-extra.txt
```

Run backend:

```bash
python -m uvicorn backend.main:app --reload
```

Run frontend in another terminal:

```bash
python -m streamlit run frontend/app.py
```

Open:

- Streamlit UI: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

## API Endpoints

- `GET /health`
- `GET /capabilities`
- `GET /datasets/sample`
- `POST /datasets/upload`
- `POST /train`
- `POST /train/upload`
- `GET /models`
- `POST /predict`
- `POST /predict/file`
- `GET /analytics/model/{model_name}`
- `GET /history/training-runs`
- `GET /history/predictions`

## Interview Explanation

`backend/ml_pipeline.py` is the ML module. It reads data, normalizes the binary target, removes ID columns, imputes missing values, scales numeric fields, one-hot encodes categories, trains the selected model, calculates metrics, saves a `joblib` artifact, and generates explanation factors.

`backend/main.py` is the API layer. It handles upload, training, prediction, analytics, and history routes while keeping HTTP logic separate from the ML pipeline.

`backend/database.py` stores model training metadata and prediction history in SQLite.

`frontend/app.py` is the analyst dashboard for model training, threshold tuning, borrower scoring, and explanation review.

Risk logic: PD below low threshold means `Approve`, between thresholds means `Review`, and above high threshold means `Reject`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q -p no:cacheprovider
```

