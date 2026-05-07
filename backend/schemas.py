from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ModelType = Literal["logistic_regression", "random_forest", "xgboost"]


class TrainRequest(BaseModel):
    dataset_path: str = "data/sample_credit_data.csv"
    target_column: str = "default"
    model_type: ModelType = "logistic_regression"
    low_threshold: float = Field(0.30, ge=0, le=1)
    high_threshold: float = Field(0.60, ge=0, le=1)
    decision_threshold: float = Field(0.50, ge=0, le=1)
    test_size: float = Field(0.25, gt=0, lt=0.8)
    model_name: str | None = None


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]]
    model_name: str | None = None
    low_threshold: float | None = Field(default=None, ge=0, le=1)
    high_threshold: float | None = Field(default=None, ge=0, le=1)

