"""FastAPI inference API for RUL prediction.

Endpoints:
- GET /health      - liveness/readiness
- GET /model-info  - architecture, window, metrics
- POST /predict    - accepts a window of sensor readings, returns predicted RUL
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config

logger = logging.getLogger("rul-api")

app = FastAPI(
    title="Predictive Maintenance RUL API",
    description="Remaining Useful Life prediction for turbofan engines (LSTM).",
    version="0.1.0",
)

# Application state populated at startup
_model: Optional[torch.jit.ScriptModule] = None
_scaler: Any = None
_feature_columns: list[str] = []
_feature_means: Optional[np.ndarray] = None
_window_length: int = 30
_model_meta: dict = {}
_alert_threshold: float = 30.0
_rul_cap: int = 125


class PredictRequest(BaseModel):
    readings: Annotated[
        list[list[float]],
        Field(
            min_length=1,
            max_length=2000,
            description="List of sensor reading vectors, one per cycle, ordered oldest→newest. "
            "Each vector must match the model's feature column order.",
        ),
    ]
    feature_columns: Optional[list[str]] = Field(
        None, description="Optional override of feature column order (must match scaler)."
    )


class PredictResponse(BaseModel):
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life in cycles.")
    rul_capped: bool = Field(..., description="Whether the training label was capped at RUL cap.")
    window_length: int
    alert: bool = Field(..., description="True if predicted RUL below the alert threshold.")
    alert_threshold: float


class ModelInfo(BaseModel):
    architecture: str
    input_size: int
    window_length: int
    feature_columns: list[str]
    best_val_rmse: Optional[float] = None


@app.on_event("startup")
def _load_artifacts() -> None:
    global _model, _scaler, _feature_columns, _feature_means  # noqa: PLW0603
    global _window_length, _model_meta, _alert_threshold, _rul_cap  # noqa: PLW0603

    model_dir = Path("models")
    meta_path = model_dir / "model_meta.json"
    scaler_path = model_dir / "scaler.pkl"
    columns_path = model_dir / "feature_columns.json"
    means_path = model_dir / "feature_means.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            _model_meta = json.load(fh)
    _window_length = int(_model_meta.get("window_length", 30))
    _rul_cap = int(_model_meta.get("rul_cap", 125))

    # Load TorchScript model
    scripted_path = model_dir / "best_model_scripted.pt"
    state_path = model_dir / "best_model.pt"
    if scripted_path.exists():
        _model = torch.jit.load(str(scripted_path))
        _model.eval()
        logger.info("Loaded TorchScript model from %s", scripted_path)
    elif state_path.exists():
        logger.warning(
            "TorchScript model not found; state dict loading requires model definition at runtime."
        )
    else:
        logger.warning(
            "No model artifact found at %s; /predict will error until a model is trained.", model_dir
        )

    # Load scaler and feature means (means used for padding short sequences)
    if scaler_path.exists():
        _scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler from %s", scaler_path)
    if columns_path.exists():
        with open(columns_path, "r", encoding="utf-8") as fh:
            _feature_columns = json.load(fh)
        logger.info("Loaded %s feature columns", len(_feature_columns))
    if means_path.exists():
        with open(means_path, "r", encoding="utf-8") as fh:
            _feature_means = np.array(json.load(fh), dtype=np.float32)
        logger.info("Loaded %s feature means", len(_feature_means))
    elif _scaler is not None and hasattr(_scaler, "mean_"):
        _feature_means = np.asarray(_scaler.mean_, dtype=np.float32)

    try:
        from src.config import load_config

        config = load_config("configs/config.yaml")
        _alert_threshold = float(config.deployment.rul_alert_threshold)
    except Exception:
        logger.warning("Could not load config; using default alert threshold %.1f", _alert_threshold)

    logger.info("Startup complete. window=%d alert_threshold=%.1f rul_cap=%d", _window_length, _alert_threshold, _rul_cap)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
        "window_length": _window_length,
    }


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    if not _model_meta:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return ModelInfo(
        architecture=_model_meta.get("architecture", "unknown"),
        input_size=int(_model_meta.get("input_size", len(_feature_columns))),
        window_length=_window_length,
        feature_columns=_feature_columns,
        best_val_rmse=_model_meta.get("best_val_rmse"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and export a model first.")

    feature_cols = req.feature_columns or _feature_columns
    if not feature_cols:
        raise HTTPException(status_code=500, detail="Feature columns not configured.")

    readings = np.array(req.readings, dtype=float)  # [n_cycles, n_features]
    expected_ndim = 2
    if readings.ndim != expected_ndim or readings.shape[1] != len(feature_cols):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected readings shape (n_cycles, {len(feature_cols)}), "
                f"got {readings.shape}."
            ),
        )

    # Use the last `window_length` cycles; pad with training means if fewer
    window = _window_length
    n_cycles = readings.shape[0]
    if n_cycles < window:
        if _feature_means is not None:
            pad = np.tile(_feature_means, (window - n_cycles, 1))
        else:
            pad = np.zeros((window - n_cycles, readings.shape[1]), dtype=float)
        readings = np.vstack([pad, readings])
    else:
        readings = readings[-window:]

    # Normalize with the fitted scaler
    if _scaler is not None:
        readings = _scaler.transform(readings)

    x = torch.from_numpy(readings.astype(np.float32)).unsqueeze(0)  # [1, window, features]
    with torch.no_grad():
        pred = _model(x).cpu().numpy().squeeze()
    predicted_rul = float(pred)

    # Sanity clamp: model predicts uncapped RUL; guard against out-of-range values
    predicted_rul = float(max(0.0, min(predicted_rul, float(_rul_cap) * 2.0)))

    alert = predicted_rul < _alert_threshold

    return PredictResponse(
        predicted_rul=round(predicted_rul, 2),
        rul_capped=False,
        window_length=window,
        alert=alert,
        alert_threshold=_alert_threshold,
    )
