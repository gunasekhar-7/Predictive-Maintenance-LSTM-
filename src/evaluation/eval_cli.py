"""CLI for evaluation: computes metrics and generates plots/report."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.sequences import RULSequenceDataset
from src.evaluation.metrics import (
    evaluate_regression,
    per_engine_errors,
    rul_bucket_errors,
)
from src.evaluation.plots import (
    plot_error_histogram,
    plot_health_curve,
    plot_predicted_vs_actual,
)
from src.models.lstm_model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained RUL models")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model-dir", default="models", help="Directory with model artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config.dataset.processed_dir)
    model_dir = Path(args.model_dir)

    test = pd.read_parquet(processed_dir / "test_processed.parquet")
    rul_test = pd.read_parquet(processed_dir / "rul_test.parquet")
    test = test.merge(rul_test, on="unit_number", how="left").rename(columns={"rul": "rul_capped"})
    with open(processed_dir / "feature_columns.json", "r", encoding="utf-8") as fh:
        feature_columns = json.load(fh)
    with open(model_dir / "model_meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # Build test sequences
    test_ds = RULSequenceDataset(
        test,
        feature_columns=feature_columns,
        target_column=config.sequences.target_column,
        window_length=config.sequences.window_length,
        stride=config.sequences.stride,
    )

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config, len(feature_columns)).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_engines = []
    all_cycles = []
    for i in range(len(test_ds)):
        x, y, eng, cyc = test_ds[i]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).cpu().numpy().squeeze()
        all_preds.append(pred)
        all_targets.append(y.numpy())
        all_engines.append(eng.numpy())
        all_cycles.append(cyc.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_engines = np.array(all_engines)
    all_cycles = np.array(all_cycles)

    results_dir = Path("reports")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Primary metrics
    metrics = evaluate_regression(
        all_targets, all_preds,
        model_name=config.model.architecture,
        alpha_early=config.evaluation.nasa_alpha_early,
        alpha_late=config.evaluation.nasa_alpha_late,
    )

    # Plots
    plot_predicted_vs_actual(all_targets, all_preds, results_dir / "pred_vs_actual.png", meta["architecture"])
    plot_error_histogram(all_targets - all_preds, results_dir / "error_hist.png", meta["architecture"])

    # Health curve for a few sample engines
    sample_engines = np.unique(all_engines)[:3]
    for eng in sample_engines:
        mask = all_engines == eng
        plot_health_curve(
            all_cycles[mask], all_targets[mask], all_preds[mask],
            results_dir / f"health_curve_engine_{eng}.png", engine_id=eng,
            alert_threshold=config.deployment.rul_alert_threshold,
        )

    # Per-engine + bucket analysis
    per_eng = per_engine_errors(all_targets, all_preds, all_engines)
    buckets = rul_bucket_errors(all_targets, all_preds)

    report = {
        "model": meta["architecture"],
        "metrics": metrics,
        "per_engine": per_eng,
        "rul_buckets": buckets,
        "n_samples": int(len(all_targets)),
    }
    with open(results_dir / "evaluation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Evaluation report saved to %s/evaluation_report.json", results_dir)
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
