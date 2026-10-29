"""Rolling statistical feature engineering per engine.

Computes trailing-window statistics (mean, std, min, max, slope) over configurable
windows. Uses only past/current cycles (no look-ahead bias) via pandas rolling
with closed='both' and min_periods=1 so early cycles still produce features.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def _slope(values: np.ndarray) -> float:
    """Linear regression slope of a 1D array; returns 0 if insufficient variance."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    x_var = ((x - x_mean) ** 2).sum()
    if x_var == 0:
        return 0.0
    cov = ((x - x_mean) * (values - y_mean)).sum()
    return float(cov / x_var)


class RollingFeatures(BaseEstimator, TransformerMixin):
    """Per-engine trailing rolling statistics.

    Parameters
    ----------
    windows : list[int]
        Window lengths in cycles.
    stats : list[str]
        Statistics to compute: mean, std, min, max, slope.
    group_col : str
        Column identifying each engine (default 'unit_number').
    time_col : str
        Column identifying cycle order (default 'time_in_cycles').
    """

    def __init__(
        self,
        windows: Optional[list[int]] = None,
        stats: Optional[list[str]] = None,
        group_col: str = "unit_number",
        time_col: str = "time_in_cycles",
    ) -> None:
        self.windows = windows or [5, 15, 30]
        self.stats = stats or ["mean", "std", "min", "max", "slope"]
        self.group_col = group_col
        self.time_col = time_col
        self.feature_names_out_: list[str] = []

    def fit(self, df: pd.DataFrame, y=None) -> "RollingFeatures":
        self.feature_names_out_ = self._build_feature_names(df.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([self.group_col, self.time_col]).copy()
        feature_cols = [c for c in df.columns if c not in {self.group_col, self.time_col, "rul", "rul_capped"}]

        out_frames = []
        for _, group in df.groupby(self.group_col, sort=False):
            group = group.sort_values(self.time_col)
            col_dict: dict[str, pd.Series] = {}
            for col in feature_cols:
                series = group[col]
                for window in self.windows:
                    rolled = series.rolling(window=window, min_periods=1)
                    if "mean" in self.stats:
                        col_dict[f"{col}_w{window}_mean"] = rolled.mean()
                    if "std" in self.stats:
                        col_dict[f"{col}_w{window}_std"] = rolled.std()
                    if "min" in self.stats:
                        col_dict[f"{col}_w{window}_min"] = rolled.min()
                    if "max" in self.stats:
                        col_dict[f"{col}_w{window}_max"] = rolled.max()
                    if "slope" in self.stats:
                        col_dict[f"{col}_w{window}_slope"] = rolled.apply(_slope, raw=True)
            feats = pd.DataFrame(col_dict, index=group.index)
            out_frames.append(feats)

        result = pd.concat(out_frames).sort_index()
        result = result[[c for c in self.feature_names_out_ if c in result.columns]]
        # Fill NaNs introduced at cycle boundaries with 0
        result = result.fillna(0.0)

        # Preserve identifying + target columns alongside rolling features (target may be absent on test)
        id_cols = df[[c for c in (self.group_col, self.time_col, "rul_capped") if c in df.columns]].reset_index(
            drop=True
        )
        return pd.concat([id_cols, result.reset_index(drop=True)], axis=1)

    def _build_feature_names(self, all_columns) -> list[str]:
        feature_cols = [
            c for c in all_columns if c not in {"unit_number", "time_in_cycles", "rul", "rul_capped"}
        ]
        names = []
        for col in feature_cols:
            for window in self.windows:
                for stat in self.stats:
                    names.append(f"{col}_w{window}_{stat}")
        return names


def add_running_max_cycle(df: pd.DataFrame, group_col: str = "unit_number", time_col: str = "time_in_cycles") -> pd.DataFrame:
    """Add the maximum cycle seen so far per engine (trailing, no look-ahead)."""
    df = df.sort_values([group_col, time_col]).copy()
    df["max_cycle_so_far"] = df.groupby(group_col)[time_col].cummax()
    return df
