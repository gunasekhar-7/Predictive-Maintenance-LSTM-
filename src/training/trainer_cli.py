"""CLI for training the deep RUL model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.training.trainer import RULTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM/GRU RUL model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config.dataset.processed_dir)

    train = pd.read_parquet(processed_dir / "train_processed.parquet")
    test = pd.read_parquet(processed_dir / "test_processed.parquet")

    with open(processed_dir / "feature_columns.json", "r", encoding="utf-8") as fh:
        feature_columns = json.load(fh)

    trainer = RULTrainer(config)
    meta = trainer.fit(train, test, feature_columns, model_dir=args.model_dir)
    logger.info("Training complete. Meta: %s", meta)


if __name__ == "__main__":
    main()
