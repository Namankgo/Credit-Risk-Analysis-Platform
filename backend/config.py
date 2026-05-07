from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
DATABASE_PATH = DATA_DIR / "credit_risk.sqlite3"
SAMPLE_DATASET_PATH = DATA_DIR / "sample_credit_data.csv"
RANDOM_STATE = 42
DEFAULT_LOW_RISK_THRESHOLD = 0.30
DEFAULT_HIGH_RISK_THRESHOLD = 0.60

for directory in (DATA_DIR, UPLOAD_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def resolve_workspace_path(path_value: str | Path) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(BASE_DIR.resolve())
    except ValueError as exc:
        raise ValueError("Path must stay inside the project workspace.") from exc
    return candidate

