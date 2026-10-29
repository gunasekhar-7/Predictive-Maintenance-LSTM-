"""Evaluation metrics including the NASA asymmetric scoring function.

The NASA score penalizes late (over-optimistic) RUL predictions more heavily
than early (conservative) ones, reflecting the higher cost of unplanned downtime.

Per sample i:
  - if y_true - y_pred >= 0  (early/conservative):  exp((y_true - y_pred)/13) - 1
  - else                                  (late):   exp((y_pred - y_true)/10) - 1
Total score = sum over samples (lower is better).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def nasa_score(y_true, y_pred, alpha_early: int = 13, alpha_late: int = 10) -> float:
    """NASA asymmetric scoring function. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    scores = np.where(
        diff >= 0,
        np.exp(diff / alpha_early) - 1.0,
        np.exp(-diff / alpha_late) - 1.0,
    )
    return float(np.sum(scores))


def evaluate_regression(
    y_true,
    y_pred,
    model_name: str = "model",
    alpha_early: int = 13,
    alpha_late: int = 10,
) -> dict[str, float]:
    """Compute all regression metrics and log them."""
    metrics = {
        "model": model_name,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred, alpha_early, alpha_late),
    }
    logger.info(
        "%s -> RMSE=%.4f MAE=%.4f R2=%.4f NASA=%.4f",
        model_name,
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
        metrics["nasa_score"],
    )
    return metrics


def per_engine_errors(y_true, y_pred, engine_ids) -> dict[int, dict[str, float]]:
    """Per-engine RMSE/MAE for error analysis."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    engine_ids = np.asarray(engine_ids)
    result = {}
    for eng in np.unique(engine_ids):
        mask = engine_ids == eng
        result[int(eng)] = {
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "mae": mae(y_true[mask], y_pred[mask]),
            "n_samples": int(mask.sum()),
        }
    return result


def rul_bucket_errors(y_true, y_pred, buckets: Optional[list[tuple[int, int]]] = None) -> dict:
    """Aggregate errors by RUL bucket (e.g., early-life vs end-of-life)."""
    if buckets is None:
        buckets = [(0, 30), (30, 75), (75, 125), (125, 10000)]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    result = {}
    for lo, hi in buckets:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        result[f"{lo}-{hi}"] = {
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "mae": mae(y_true[mask], y_pred[mask]),
            "n_samples": int(mask.sum()),
        }
    return result
