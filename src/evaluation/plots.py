"""Evaluation plots: predicted vs actual, health curves, error analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_predicted_vs_actual(y_true, y_pred, output_path: str | Path, model_name: str = "model") -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20, edgecolor="k", linewidth=0.3)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", label="Ideal (y=x)")
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"{model_name}: Predicted vs Actual RUL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved predicted vs actual plot to %s", output_path)


def plot_health_curve(
    cycles,
    y_true,
    y_pred,
    output_path: str | Path,
    engine_id: int | str,
    alert_threshold: float = 30.0,
) -> None:
    cycles = np.asarray(cycles)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cycles, y_true, label="Actual RUL", color="#1f77b4", linewidth=2)
    ax.plot(cycles, y_pred, label="Predicted RUL", color="#ff7f0e", linewidth=2, linestyle="--")
    ax.axhline(alert_threshold, color="#d62728", linestyle=":", label=f"Alert threshold ({alert_threshold})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.set_title(f"Health Curve - Engine {engine_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved health curve for engine %s to %s", engine_id, output_path)


def plot_error_histogram(errors, output_path: str | Path, model_name: str = "model") -> None:
    errors = np.asarray(errors, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=40, color="#2ca02c", edgecolor="k", alpha=0.8)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel("Prediction error (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name}: Error distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved error histogram to %s", output_path)


def plot_metrics_comparison(results: list[dict], output_path: str | Path) -> None:
    """Bar chart comparing RMSE/MAE across models."""
    models = [r["model"] for r in results]
    rmse_vals = [r["rmse"] for r in results]
    mae_vals = [r["mae"] for r in results]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, rmse_vals, width, label="RMSE", color="#1f77b4")
    ax.bar(x + width / 2, mae_vals, width, label="MAE", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Error (cycles)")
    ax.set_title("Model comparison: RMSE vs MAE")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved metrics comparison to %s", output_path)
