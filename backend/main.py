from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import SAMPLE_DATASET_PATH, UPLOAD_DIR, resolve_workspace_path
from backend.database import init_db, list_predictions, list_training_runs, save_prediction, save_training_run
from backend.ml_pipeline import (
    ModelNotFoundError,
    dataset_profile,
    list_saved_models,
    model_analytics,
    predict_credit_risk,
    read_dataset,
    runtime_capabilities,
    to_native,
    train_credit_model,
)
from backend.schemas import PredictionRequest, TrainRequest


app = FastAPI(title="Credit Risk Analysis Platform API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _save_uploaded_csv(file: UploadFile) -> Path:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = UPLOAD_DIR / f"{timestamp}_{Path(file.filename).name.replace(' ', '_')}"
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return destination


def _handle_error(exc: Exception) -> HTTPException:
    status = 404 if isinstance(exc, (FileNotFoundError, ModelNotFoundError)) else 400
    return HTTPException(status_code=status, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/capabilities")
def capabilities() -> dict[str, bool]:
    return runtime_capabilities()


@app.get("/datasets/sample")
def sample_dataset() -> dict:
    try:
        return dataset_profile(SAMPLE_DATASET_PATH)
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.get("/datasets/profile")
def profile_dataset(dataset_path: str = "data/sample_credit_data.csv") -> dict:
    try:
        return dataset_profile(resolve_workspace_path(dataset_path))
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.post("/datasets/upload")
async def upload_dataset(file: Annotated[UploadFile, File(...)]) -> dict:
    try:
        destination = _save_uploaded_csv(file)
        return {"dataset_path": str(destination), **dataset_profile(destination)}
    except HTTPException:
        raise
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.post("/train")
def train_model(request: TrainRequest) -> dict:
    try:
        result = train_credit_model(**request.model_dump())
        return {"training_run_id": save_training_run(result), **result}
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.post("/train/upload")
async def train_uploaded_model(
    file: Annotated[UploadFile, File(...)],
    target_column: Annotated[str, Form()] = "default",
    model_type: Annotated[str, Form()] = "logistic_regression",
    low_threshold: Annotated[float, Form()] = 0.30,
    high_threshold: Annotated[float, Form()] = 0.60,
    decision_threshold: Annotated[float, Form()] = 0.50,
) -> dict:
    try:
        destination = _save_uploaded_csv(file)
        result = train_credit_model(destination, target_column, model_type, low_threshold, high_threshold, decision_threshold)
        return {"training_run_id": save_training_run(result), **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.get("/models")
def models() -> dict[str, list[dict]]:
    return {"models": list_saved_models()}


@app.get("/analytics/latest")
def latest_analytics() -> dict:
    try:
        return model_analytics()
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.get("/analytics/model/{model_name}")
def analytics(model_name: str) -> dict:
    try:
        return model_analytics(model_name)
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        result = predict_credit_risk(request.records, request.model_name, request.low_threshold, request.high_threshold)
        for record, prediction in zip(request.records, result["predictions"], strict=False):
            save_prediction(result["model_name"], prediction, record)
        return result
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.post("/predict/file")
async def predict_file(
    file: Annotated[UploadFile, File(...)],
    model_name: Annotated[str | None, Form()] = None,
    low_threshold: Annotated[float | None, Form()] = None,
    high_threshold: Annotated[float | None, Form()] = None,
) -> dict:
    try:
        destination = _save_uploaded_csv(file)
        records = to_native(read_dataset(destination).to_dict(orient="records"))
        result = predict_credit_risk(records, model_name, low_threshold, high_threshold)
        for record, prediction in zip(records, result["predictions"], strict=False):
            save_prediction(result["model_name"], prediction, record)
        return {"dataset_path": str(destination), **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise _handle_error(exc) from exc


@app.get("/history/training-runs")
def training_history(limit: int = 25) -> dict[str, list[dict]]:
    return {"training_runs": list_training_runs(limit)}


@app.get("/history/predictions")
def prediction_history(limit: int = 50) -> dict[str, list[dict]]:
    return {"predictions": list_predictions(limit)}

