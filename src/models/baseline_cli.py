"""CLI for baseline training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.features.rolling import RollingFeatures
from src.models.baseline import train_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost / Random Forest baselines")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config.dataset.processed_dir)

    train = pd.read_parquet(processed_dir / "train_processed.parquet")
    test = pd.read_parquet(processed_dir / "test_processed.parquet")
    rul_test = pd.read_parquet(processed_dir / "rul_test.parquet")["rul"]

    # Build rolling features per engine
    featurizer = RollingFeatures(
        windows=config.features.rolling_windows,
        stats=config.features.rolling_stats,
    )
    featurizer.fit(train)
    train_feat = featurizer.transform(train)
    test_feat = featurizer.transform(test)

    feature_cols = [c for c in train_feat.columns if c not in {"unit_number", "time_in_cycles", "rul", "rul_capped"}]

    # Use final cycle of each engine as the baseline sample
    train_last = train_feat.sort_values("time_in_cycles").groupby("unit_number").tail(1)
    test_last = test_feat.sort_values("time_in_cycles").groupby("unit_number").tail(1)

    X_train = train_last[feature_cols]
    y_train = train_last["rul_capped"]
    X_test = test_last[feature_cols]
    y_test = rul_test.reindex(test_last["unit_number"]).reset_index(drop=True)
    groups = train_last["unit_number"].reset_index(drop=True)

    results = train_baselines(
        X_train, y_train, groups, X_test, y_test, output_dir="models/baselines"
    )
    logger.info("Baseline results: %s", results)


if __name__ == "__main__":
    main()
