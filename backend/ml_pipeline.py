from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.config import (
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_LOW_RISK_THRESHOLD,
    MODELS_DIR,
    RANDOM_STATE,
    SAMPLE_DATASET_PATH,
    resolve_workspace_path,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import shap
except Exception:
    shap = None


MODEL_TYPES = ("logistic_regression", "random_forest", "xgboost")
POSITIVE_LABELS = {"1", "yes", "true", "default", "bad", "charged_off", "high risk"}
NEGATIVE_LABELS = {"0", "no", "false", "paid", "good", "current", "low risk"}


class ModelNotFoundError(FileNotFoundError):
    pass


def runtime_capabilities() -> dict[str, bool]:
    return {"xgboost_available": XGBClassifier is not None, "shap_available": shap is not None}


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_native(value.tolist())
    if isinstance(value, np.generic):
        return to_native(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def validate_thresholds(low_threshold: float, high_threshold: float) -> None:
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= low < high <= 1.")


def risk_category(probability_default: float, low_threshold: float, high_threshold: float) -> str:
    if probability_default < low_threshold:
        return "Low"
    if probability_default < high_threshold:
        return "Medium"
    return "High"


def decision_suggestion(category: str) -> str:
    return {"Low": "Approve", "Medium": "Review", "High": "Reject"}[category]


def read_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    path = resolve_workspace_path(dataset_path or SAMPLE_DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def dataset_profile(dataset_path: str | Path | None = None) -> dict[str, Any]:
    frame = read_dataset(dataset_path)
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "target_suggestions": [
            column for column in frame.columns if column.lower() in {"default", "target", "loan_status", "is_default"}
        ],
        "missing_values": to_native(frame.isna().sum().to_dict()),
        "preview": to_native(frame.head(10).to_dict(orient="records")),
    }


def _coerce_numeric_like_columns(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        if converted[column].dtype == object:
            numeric = pd.to_numeric(converted[column], errors="coerce")
            if numeric.notna().mean() >= 0.85:
                converted[column] = numeric
    return converted


def _normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        unique_values = sorted(series.dropna().unique())
        if set(unique_values).issubset({0, 1}):
            return series.astype(int)
        if len(unique_values) == 2:
            return series.map({unique_values[0]: 0, unique_values[1]: 1}).astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(lambda item: 1 if item in POSITIVE_LABELS else 0 if item in NEGATIVE_LABELS else np.nan)
    if mapped.notna().all():
        return mapped.astype(int)
    unique_values = sorted(normalized.dropna().unique())
    if len(unique_values) == 2:
        return normalized.map({unique_values[0]: 0, unique_values[1]: 1}).astype(int)
    raise ValueError("Target must be binary, such as 0/1, yes/no, default/paid.")


def _identifier_columns(frame: pd.DataFrame) -> list[str]:
    dropped = []
    row_count = len(frame)
    for column in frame.columns:
        lowered = column.lower()
        unique_ratio = frame[column].nunique(dropna=True) / max(row_count, 1)
        if lowered in {"id", "customer_id", "application_id"} or lowered.endswith("_id") or unique_ratio > 0.98:
            dropped.append(column)
    return dropped


def _feature_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in frame.columns if frame[column].dtype == object]
    numeric = [column for column in frame.columns if column not in categorical]
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer([("num", numeric_pipeline, numeric_features), ("cat", categorical_pipeline, categorical_features)])


def make_estimator(model_type: str):
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    if model_type == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("XGBoost is not installed. Run `pip install -r requirements-ml-extra.txt`.")
        return XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)
    raise ValueError(f"Unsupported model type: {model_type}")


def _positive_class_probability(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(frame)[:, 1]


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(name) for name in preprocessor.get_feature_names_out()]


def _humanize_feature(name: str, categorical_features: list[str]) -> str:
    clean = name.replace("num__", "", 1).replace("cat__", "", 1)
    for column in sorted(categorical_features, key=len, reverse=True):
        prefix = f"{column}_"
        if clean.startswith(prefix):
            return f"{column.replace('_', ' ').title()} = {clean[len(prefix):].replace('_', ' ').title()}"
    return clean.replace("_", " ").title()


def _source_feature(name: str, categorical_features: list[str]) -> str:
    clean = name.replace("num__", "", 1).replace("cat__", "", 1)
    for column in sorted(categorical_features, key=len, reverse=True):
        if clean.startswith(f"{column}_"):
            return column
    return clean


def _importance_rows(pipeline: Pipeline, categorical_features: list[str], limit: int = 25) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    names = _feature_names(pipeline.named_steps["preprocessor"])
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "model_feature_importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_[0], dtype=float))
        method = "absolute_logistic_coefficient"
    else:
        return []
    total = float(values.sum()) or 1.0
    rows = [
        {
            "feature": _humanize_feature(name, categorical_features),
            "source_feature": _source_feature(name, categorical_features),
            "importance": float(value / total),
            "raw_importance": float(value),
            "method": method,
        }
        for name, value in zip(names, values, strict=False)
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)[:limit]


def _metrics(y_true: pd.Series, probabilities: np.ndarray, decision_threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= decision_threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "decision_threshold": float(decision_threshold),
    }
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    result["roc_curve"] = {
        "fpr": [float(item) for item in fpr],
        "tpr": [float(item) for item in tpr],
        "thresholds": [float(item) if np.isfinite(item) else None for item in thresholds],
    }
    return result


def _risk_distribution(probabilities: np.ndarray, low_threshold: float, high_threshold: float) -> dict[str, int]:
    distribution = {"Low": 0, "Medium": 0, "High": 0}
    for probability in probabilities:
        distribution[risk_category(float(probability), low_threshold, high_threshold)] += 1
    return distribution


def train_credit_model(
    dataset_path: str | Path | None = None,
    target_column: str = "default",
    model_type: str = "logistic_regression",
    low_threshold: float = DEFAULT_LOW_RISK_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    decision_threshold: float = 0.50,
    test_size: float = 0.25,
    model_name: str | None = None,
) -> dict[str, Any]:
    validate_thresholds(low_threshold, high_threshold)
    source_path = resolve_workspace_path(dataset_path or SAMPLE_DATASET_PATH)
    frame = read_dataset(source_path)
    target = _normalize_target(frame[target_column])
    features = _coerce_numeric_like_columns(frame.drop(columns=[target_column]))
    dropped_columns = _identifier_columns(features)
    features = features.drop(columns=dropped_columns)
    numeric_features, categorical_features = _feature_groups(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=RANDOM_STATE, stratify=target)
    pipeline = Pipeline([("preprocessor", build_preprocessor(numeric_features, categorical_features)), ("model", make_estimator(model_type))])
    pipeline.fit(x_train, y_train)
    probabilities = _positive_class_probability(pipeline, x_test)
    metrics = _metrics(y_test, probabilities, decision_threshold)
    feature_importance = _importance_rows(pipeline, categorical_features)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved_name = model_name or f"{model_type}_{created_at}.joblib"
    if not saved_name.endswith(".joblib"):
        saved_name = f"{saved_name}.joblib"
    model_path = MODELS_DIR / saved_name
    metadata = {
        "model_name": saved_name,
        "model_type": model_type,
        "created_at": created_at,
        "dataset_path": str(source_path),
        "target_column": target_column,
        "feature_columns": list(features.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": dropped_columns,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "decision_threshold": decision_threshold,
        "metrics": metrics,
        "risk_distribution": _risk_distribution(probabilities, low_threshold, high_threshold),
    }
    artifact = {"pipeline": pipeline, "metadata": metadata, "feature_importance": feature_importance}
    joblib.dump(artifact, model_path)
    return to_native({**metadata, "model_path": str(model_path), "feature_importance": feature_importance})


def list_saved_models() -> list[dict[str, Any]]:
    models = []
    for path in sorted(MODELS_DIR.glob("*.joblib"), key=lambda item: item.stat().st_mtime, reverse=True):
        metadata = joblib.load(path).get("metadata", {})
        models.append(
            {
                "model_name": path.name,
                "model_path": str(path),
                "model_type": metadata.get("model_type"),
                "created_at": metadata.get("created_at"),
                "roc_auc": metadata.get("metrics", {}).get("roc_auc"),
                "accuracy": metadata.get("metrics", {}).get("accuracy"),
            }
        )
    return models


def load_model(model_name: str | None = None) -> dict[str, Any]:
    if model_name is None:
        saved = list_saved_models()
        if not saved:
            raise ModelNotFoundError("No trained model found. Train a model first.")
        model_name = saved[0]["model_name"]
    path = MODELS_DIR / model_name
    if not path.exists():
        raise ModelNotFoundError(f"Model not found: {model_name}")
    return joblib.load(path)


def _business_reason_flags(record: dict[str, Any]) -> list[str]:
    reasons = []
    credit_score = float(record.get("credit_score", 700))
    debt_to_income = float(record.get("debt_to_income", 0.25))
    delinquencies = float(record.get("delinquencies_2yrs", 0))
    utilization = float(record.get("revolving_utilization", 0.3))
    inquiries = float(record.get("inquiries_last_6m", 0))
    if credit_score < 620:
        reasons.append("Credit score is below the usual prime lending band.")
    if debt_to_income > 0.43:
        reasons.append("Debt-to-income ratio is above the common 43% affordability checkpoint.")
    if delinquencies > 0:
        reasons.append("Recent delinquencies indicate repayment stress.")
    if utilization > 0.75:
        reasons.append("Revolving credit utilization is high.")
    if inquiries >= 4:
        reasons.append("Several recent credit inquiries may signal new borrowing pressure.")
    return reasons


def _model_factors(artifact: dict[str, Any], row: pd.DataFrame) -> list[dict[str, Any]]:
    pipeline = artifact["pipeline"]
    model = pipeline.named_steps["model"]
    names = _feature_names(pipeline.named_steps["preprocessor"])
    categorical = artifact["metadata"]["categorical_features"]
    transformed = np.asarray(pipeline.named_steps["preprocessor"].transform(row))[0]
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_[0], dtype=float) * transformed
        direction = lambda item: "increases PD" if item > 0 else "decreases PD"
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float) * np.abs(transformed)
        direction = lambda item: "important model driver"
    else:
        return []
    order = np.argsort(np.abs(values))[::-1][:5]
    return [
        {
            "feature": _humanize_feature(names[index], categorical),
            "source_feature": _source_feature(names[index], categorical),
            "contribution": float(values[index]),
            "direction": direction(float(values[index])),
        }
        for index in order
    ]


def predict_credit_risk(
    records: list[dict[str, Any]],
    model_name: str | None = None,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> dict[str, Any]:
    artifact = load_model(model_name)
    metadata = artifact["metadata"]
    low = float(low_threshold if low_threshold is not None else metadata["low_threshold"])
    high = float(high_threshold if high_threshold is not None else metadata["high_threshold"])
    validate_thresholds(low, high)
    frame = _coerce_numeric_like_columns(pd.DataFrame(records))
    for column in metadata["feature_columns"]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[metadata["feature_columns"]]
    probabilities = _positive_class_probability(artifact["pipeline"], frame)
    predictions = []
    for index, probability in enumerate(probabilities):
        pd_value = float(probability)
        category = risk_category(pd_value, low, high)
        predictions.append(
            {
                "probability_default": pd_value,
                "risk_category": category,
                "decision": decision_suggestion(category),
                "explanation": {
                    "summary": {
                        "Low": "Low risk because PD is below the configured low-risk threshold.",
                        "Medium": "Review recommended because PD sits between the low and high thresholds.",
                        "High": "High risk because PD is above the configured high-risk threshold.",
                    }[category],
                    "method": "model_contribution_heuristic",
                    "top_factors": to_native(_model_factors(artifact, frame.iloc[[index]])),
                    "business_reasons": _business_reason_flags(records[index]),
                    "probability_default": pd_value,
                },
            }
        )
    return to_native({"model_name": metadata["model_name"], "thresholds": {"low": low, "high": high}, "predictions": predictions, "risk_distribution": _risk_distribution(probabilities, low, high)})


def model_analytics(model_name: str | None = None) -> dict[str, Any]:
    artifact = load_model(model_name)
    metadata = artifact["metadata"]
    return to_native(
        {
            "model_name": metadata["model_name"],
            "model_type": metadata["model_type"],
            "created_at": metadata["created_at"],
            "metrics": metadata["metrics"],
            "risk_distribution": metadata["risk_distribution"],
            "feature_importance": artifact.get("feature_importance", []),
            "feature_columns": metadata["feature_columns"],
            "dropped_columns": metadata["dropped_columns"],
            "thresholds": {"low": metadata["low_threshold"], "high": metadata["high_threshold"], "decision": metadata["decision_threshold"]},
        }
    )

