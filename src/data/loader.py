"""NASA C-MAPSS dataset loader, cleaning, normalization, and RUL labeling.

Handles the turbofan engine degradation dataset:
- train_FD001.txt, test_FD001.txt, RUL_FD001.txt (and FD002-FD004 variants)
- Drops near-constant sensors, imputes missing values, normalizes per config,
- Computes piecewise-linear RUL labels capped at a configurable threshold.

All transforms are fitted on training data only to prevent data leakage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import Config, load_config

logger = logging.getLogger(__name__)

SENSOR_PREFIX = "sensor_"
OP_SETTING_PREFIX = "op_setting_"


@dataclass
class ProcessedData:
    """Container for processed train/test splits and preprocessing artifacts."""

    train: pd.DataFrame
    test: pd.DataFrame
    rul_test: pd.Series
    feature_columns: list[str]
    scaler: BaseEstimator
    config: Config


class CMAPSSLoader:
    """Loads and cleans raw C-MAPSS text files into long-format DataFrames."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.columns = list(config.dataset.columns)

    def load_train(self, data_dir: Optional[str | Path] = None) -> pd.DataFrame:
        """Load training data with RUL computed from max cycle per engine."""
        data_dir = Path(data_dir or self.config.dataset.data_dir)
        path = data_dir / self.config.dataset.train_file
        if not path.exists():
            raise FileNotFoundError(f"Training file not found: {path}")
        df = self._read_file(path)
        df = self._compute_rul(df)
        logger.info("Loaded train data: %s rows, %s engines", len(df), df["unit_number"].nunique())
        return df

    def load_test(self, data_dir: Optional[str | Path] = None) -> pd.DataFrame:
        """Load test data (no RUL column in raw test file)."""
        data_dir = Path(data_dir or self.config.dataset.data_dir)
        path = data_dir / self.config.dataset.test_file
        if not path.exists():
            raise FileNotFoundError(f"Test file not found: {path}")
        df = self._read_file(path)
        logger.info("Loaded test data: %s rows, %s engines", len(df), df["unit_number"].nunique())
        return df

    def load_rul_test(self, data_dir: Optional[str | Path] = None) -> pd.Series:
        """Load ground-truth RUL at the final cycle for each test engine."""
        data_dir = Path(data_dir or self.config.dataset.data_dir)
        path = data_dir / self.config.dataset.rul_file
        if not path.exists():
            raise FileNotFoundError(f"RUL file not found: {path}")
        rul = pd.read_csv(path, header=None, names=["rul"])
        rul.index += 1  # 1-indexed to match unit_number
        logger.info("Loaded RUL ground truth: %s engines", len(rul))
        return rul["rul"]

    def _read_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=self.columns)
        # Drop trailing empty columns if present (some C-MAPSS exports have 2 extra NaN cols)
        df = df.dropna(axis=1, how="all")
        # Ensure monotonic cycles per engine
        df = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)
        return df

    def _compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """RUL = max cycle per engine - current cycle."""
        max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df = df.copy()
        df["rul"] = (max_cycles - df["time_in_cycles"]).astype(float)
        return df


class DataCleaner(BaseEstimator, TransformerMixin):
    """Cleans sensor data: drops near-constant sensors, imputes, removes duplicates."""

    def __init__(
        self,
        variance_threshold: float = 1e-6,
        corr_threshold: float = 0.98,
        rul_cap: int = 125,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.rul_cap = rul_cap
        self.feature_columns_: list[str] = []
        self.sensor_columns_: list[str] = []

    def fit(self, df: pd.DataFrame, y=None) -> "DataCleaner":
        non_feature = {
            "unit_number",
            "time_in_cycles",
            "rul",
            "rul_capped",
        }
        candidate_cols = [c for c in df.columns if c not in non_feature]

        # Identify sensor columns
        self.sensor_columns_ = [c for c in candidate_cols if c.startswith(SENSOR_PREFIX)]
        op_cols = [c for c in candidate_cols if c.startswith(OP_SETTING_PREFIX)]

        # Build a clean numeric view of the sensor columns (pandas 3.x may read raw C-MAPSS as object)
        sensor_df = df[self.sensor_columns_].apply(pd.to_numeric, errors="coerce")

        # Drop near-constant sensors
        var_list = []
        for col in sensor_df.columns:
            try:
                v = float(np.nanvar(sensor_df[col].values.astype(float)))
            except Exception:
                v = float("nan")
            var_list.append((col, v))
        constant_sensors = []
        for col, v in var_list:
            try:
                if float(v) <= self.variance_threshold:
                    constant_sensors.append(col)
            except (TypeError, ValueError):
                continue
        if constant_sensors:
            logger.info("Dropping near-constant sensors: %s", constant_sensors)
            self.sensor_columns_ = [c for c in self.sensor_columns_ if c not in constant_sensors]

        # Drop highly correlated sensors (redundancy)
        if len(self.sensor_columns_) > 1:
            corr = sensor_df[self.sensor_columns_].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = []
            for column in upper.columns:
                try:
                    if any(float(x) > self.corr_threshold for x in upper[column].values if pd.notna(x)):
                        to_drop.append(column)
                except (TypeError, ValueError):
                    continue
            if to_drop:
                logger.info("Dropping highly correlated sensors: %s", to_drop)
                self.sensor_columns_ = [c for c in self.sensor_columns_ if c not in to_drop]

        self.feature_columns_ = self.sensor_columns_ + op_cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Remove exact duplicates
        before = len(df)
        df = df.drop_duplicates()
        if len(df) < before:
            logger.info("Removed %s duplicate rows", before - len(df))

        # Ensure feature columns are numeric (raw C-MAPSS may read as object in pandas 3.x)
        for col in self.feature_columns_:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Impute missing values per column (forward-fill within engine, then median)
        for col in self.feature_columns_:
            if col in df.columns:
                df[col] = df.groupby("unit_number")[col].transform(lambda s: s.ffill().bfill())
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

        # Cap RUL (piecewise-linear label)
        if "rul" in df.columns:
            df["rul_capped"] = np.minimum(df["rul"], self.rul_cap)

        return df


class Normalizer(BaseEstimator, TransformerMixin):
    """Fits a scaler on training data and applies it to train/test consistently."""

    def __init__(self, scaler_type: str = "standard") -> None:
        self.scaler_type = scaler_type
        self.scaler_: BaseEstimator
        self.feature_columns_: list[str] = []

    def fit(self, df: pd.DataFrame, feature_columns: list[str], y=None) -> "Normalizer":
        self.feature_columns_ = list(feature_columns)
        if self.scaler_type == "minmax":
            self.scaler_ = MinMaxScaler()
        else:
            self.scaler_ = StandardScaler()
        self.scaler_.fit(df[self.feature_columns_])
        logger.info(
            "Fitted %s scaler on %s features using %s rows",
            self.scaler_type,
            len(self.feature_columns_),
            len(df),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.feature_columns_] = self.scaler_.transform(df[self.feature_columns_])
        return df


def preprocess(
    config: Config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_test: pd.Series,
) -> ProcessedData:
    """End-to-end preprocessing pipeline fitted on train only."""
    cleaner = DataCleaner(
        variance_threshold=config.preprocessing.variance_threshold,
        corr_threshold=config.preprocessing.corr_threshold,
        rul_cap=config.preprocessing.rul_cap,
    )
    cleaner.fit(train_df)
    train_clean = cleaner.transform(train_df)
    test_clean = cleaner.transform(test_df)

    # Align test RUL ground truth to test engines present
    test_units = sorted(test_clean["unit_number"].unique())
    rul_aligned = rul_test.reindex(test_units)

    normalizer = Normalizer(scaler_type=config.preprocessing.scaler_type)
    normalizer.fit(train_clean, cleaner.feature_columns_)
    train_norm = normalizer.transform(train_clean)
    test_norm = normalizer.transform(test_clean)

    return ProcessedData(
        train=train_norm,
        test=test_norm,
        rul_test=rul_aligned,
        feature_columns=cleaner.feature_columns_,
        scaler=normalizer.scaler_,
        config=config,
    )


def build_dataset_splits(
    processed: ProcessedData,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test for baselines using final-cycle rows."""
    train = processed.train
    test = processed.test
    feature_cols = processed.feature_columns

    # For baselines, use the last cycle of each engine as the sample
    train_last = train.sort_values("time_in_cycles").groupby("unit_number").tail(1)
    test_last = test.sort_values("time_in_cycles").groupby("unit_number").tail(1)

    X_train = train_last[feature_cols]
    y_train = train_last["rul_capped"]
    X_test = test_last[feature_cols]
    y_test = processed.rul_test.reindex(test_last["unit_number"]).reset_index(drop=True)

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def save_artifacts(processed: ProcessedData, output_dir: str | Path) -> None:
    """Persist processed parquet artifacts and scaler for inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed.train.to_parquet(output_dir / "train_processed.parquet", index=False)
    processed.test.to_parquet(output_dir / "test_processed.parquet", index=False)
    # Save rul_test with unit_number as an explicit column for robust alignment
    rul_df = processed.rul_test.rename_axis("unit_number").reset_index(name="rul")
    rul_df.to_parquet(output_dir / "rul_test.parquet", index=False)

    with open(output_dir / "feature_columns.json", "w", encoding="utf-8") as fh:
        json.dump(processed.feature_columns, fh, indent=2)
    # Save feature means for padding short inference sequences with in-distribution values
    feature_means = (
        processed.train[processed.feature_columns].mean(axis=0).astype(float).tolist()
    )
    with open(output_dir / "feature_means.json", "w", encoding="utf-8") as fh:
        json.dump(feature_means, fh, indent=2)

    joblib.dump(processed.scaler, output_dir / "scaler.pkl")
    logger.info("Saved processed artifacts to %s", output_dir)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
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
