"""Sliding-window sequence dataset builder for RUL prediction.

Builds input sequences of length `window_length` from normalized per-engine
trajectories. Target is the RUL at the window's end cycle. No look-ahead bias:
target at cycle t uses only data up to and including cycle t.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RULSequenceDataset(Dataset):
    """PyTorch Dataset of sliding-window sensor sequences -> RUL targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "rul_capped",
        window_length: int = 30,
        stride: int = 1,
        group_col: str = "unit_number",
        time_col: str = "time_in_cycles",
        rul_test: Optional[pd.Series] = None,
    ) -> None:
        self.window_length = window_length
        self.stride = stride
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.group_col = group_col
        self.time_col = time_col

        self.sequences: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.engine_ids: Optional[np.ndarray] = None
        self.cycle_ids: Optional[np.ndarray] = None

        # Test data may not carry rul_capped; join ground-truth RUL if provided.
        if rul_test is not None and target_column not in df.columns:
            df = df.copy()
            df[target_column] = df[self.group_col].map(rul_test)

        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        df = df.sort_values([self.group_col, self.time_col]).copy()
        seqs: list[np.ndarray] = []
        tgts: list[float] = []
        eng_ids: list[int] = []
        cyc_ids: list[int] = []
        n_engines_total = 0
        n_engines_dropped = 0
        for engine_id, group in df.groupby(self.group_col, sort=False):
            n_engines_total += 1
            group = group.sort_values(self.time_col)
            values = group[self.feature_columns].values.astype(np.float32)
            targets = group[self.target_column].values.astype(np.float32)
            cycles = group[self.time_col].values.astype(np.int64)
            n = len(values)
            if n < self.window_length:
                n_engines_dropped += 1
                continue
            # Sliding windows: window ending at cycle t (0-indexed end position)
            for end in range(self.window_length - 1, n, self.stride):
                start = end - self.window_length + 1
                seqs.append(values[start : end + 1])
                tgts.append(float(targets[end]))
                eng_ids.append(int(engine_id))  # type: ignore[arg-type]  # unit_number is int
                cyc_ids.append(int(cycles[end]))

        if not seqs:
            raise ValueError(
                f"No sequences built. window_length={self.window_length} may exceed "
                "shortest engine trajectory."
            )

        self.sequences = np.stack(seqs)
        self.targets = np.stack(tgts)
        self.engine_ids = np.stack(eng_ids)
        self.cycle_ids = np.stack(cyc_ids)

        logger.info(
            "Built %s sequences (window=%d, stride=%d) from %s/%s engines "
            "(%s dropped: shorter than window)",
            len(self),
            self.window_length,
            self.stride,
            n_engines_total - n_engines_dropped,
            n_engines_total,
            n_engines_dropped,
        )

    def __len__(self) -> int:
        assert self.targets is not None
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.sequences is not None
        assert self.targets is not None
        assert self.engine_ids is not None
        assert self.cycle_ids is not None
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.engine_ids[idx], dtype=torch.int64),
            torch.tensor(self.cycle_ids[idx], dtype=torch.int64),
        )


def build_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    config,
    target_column: str = "rul_capped",
    rul_test: Optional[pd.Series] = None,
) -> tuple[RULSequenceDataset, RULSequenceDataset]:
    """Build train and validation/test sequence datasets."""
    # Test data may not carry rul_capped; join the ground-truth RUL if provided.
    if rul_test is not None and target_column not in test_df.columns:
        test_df = test_df.copy()
        test_df[target_column] = test_df["unit_number"].map(rul_test)
    train_ds = RULSequenceDataset(
        train_df,
        feature_columns=feature_columns,
        target_column=target_column,
        window_length=config.sequences.window_length,
        stride=config.sequences.stride,
    )
    test_ds = RULSequenceDataset(
        test_df,
        feature_columns=feature_columns,
        target_column=target_column,
        window_length=config.sequences.window_length,
        stride=config.sequences.stride,
    )
    return train_ds, test_ds
    return train_ds, test_ds
