from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from backend.config import DATABASE_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                target_column TEXT NOT NULL,
                roc_auc REAL,
                accuracy REAL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                probability_default REAL NOT NULL,
                risk_category TEXT NOT NULL,
                decision TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                explanation_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_training_run(result: dict[str, Any]) -> int:
    metrics = result.get("metrics", {})
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO training_runs (
                model_name, model_type, dataset_path, target_column,
                roc_auc, accuracy, created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["model_name"],
                result["model_type"],
                result["dataset_path"],
                result["target_column"],
                metrics.get("roc_auc"),
                metrics.get("accuracy"),
                utc_now(),
                json.dumps(result, default=str),
            ),
        )
        return int(cursor.lastrowid)


def save_prediction(model_name: str, prediction: dict[str, Any], payload: dict[str, Any]) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                model_name, probability_default, risk_category, decision,
                payload_json, explanation_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                prediction["probability_default"],
                prediction["risk_category"],
                prediction["decision"],
                json.dumps(payload, default=str),
                json.dumps(prediction.get("explanation", {}), default=str),
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)


def list_training_runs(limit: int = 25) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, model_name, model_type, dataset_path, target_column,
                   roc_auc, accuracy, created_at
            FROM training_runs ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_predictions(limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, model_name, probability_default, risk_category, decision,
                   payload_json, explanation_json, created_at
            FROM predictions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    history = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        item["explanation"] = json.loads(item.pop("explanation_json"))
        history.append(item)
    return history

