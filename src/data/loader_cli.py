"""CLI entry point for data loading and preprocessing."""

from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.data.loader import CMAPSSLoader, preprocess, save_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and preprocess C-MAPSS data")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", default=None, help="Override raw data directory")
    args = parser.parse_args()

    config = load_config(args.config)
    loader = CMAPSSLoader(config)

    train_df = loader.load_train(args.data_dir)
    test_df = loader.load_test(args.data_dir)
    rul_test = loader.load_rul_test(args.data_dir)

    processed = preprocess(config, train_df, test_df, rul_test)
    save_artifacts(processed, config.dataset.processed_dir)

    logger.info("Preprocessing complete. Features: %s", len(processed.feature_columns))


if __name__ == "__main__":
    main()
