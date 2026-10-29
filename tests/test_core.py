"""Test suite for predictive-maintenance-lstm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.loader import DataCleaner

EXPECTED_DATASET_NAME = "FD001"
EXPECTED_RUL_CAP = 125
EXPECTED_WINDOW_LENGTH = 30
EXPECTED_SEED = 42
from src.data.sequences import RULSequenceDataset
from src.evaluation.metrics import mae, nasa_score, r2, rmse
from src.features.rolling import RollingFeatures
from src.models.lstm_model import build_model


def test_rmse_mae_r2_known():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    assert abs(rmse(y_true, y_pred) - np.sqrt((4 + 4 + 0) / 3)) < 1e-6
    assert abs(mae(y_true, y_pred) - (2 + 2 + 0) / 3) < 1e-6
    assert abs(r2(y_true, y_pred) - 0.96) < 1e-6


def test_nasa_score_penalizes_late_more():
    y_true = np.array([50.0, 50.0])
    # Same magnitude error: early (-10) vs late (+10)
    early_pred = np.array([40.0])   # true - pred = +10 (early)
    late_pred = np.array([60.0])    # true - pred = -10 (late)
    s_early = nasa_score(np.array([50.0]), early_pred)
    s_late = nasa_score(np.array([50.0]), late_pred)
    assert s_late > s_early, f"Late should penalize more: {s_late} vs {s_early}"
    # Non-negative scores
    assert s_early >= 0 and s_late >= 0


def test_rolling_features_shape_and_no_nan():
    n_engines = 3
    n_cycles = 40
    frames = []
    for eng in range(1, n_engines + 1):
        df = pd.DataFrame(
            {
                "unit_number": eng,
                "time_in_cycles": np.arange(1, n_cycles + 1),
                "sensor_1": np.random.rand(n_cycles) * 10,
                "sensor_2": np.random.rand(n_cycles) * 10,
                "rul": np.arange(n_cycles - 1, -1, -1).astype(float),
            }
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    featurizer = RollingFeatures(windows=[5, 15], stats=["mean", "std"])
    featurizer.fit(df)
    out = featurizer.transform(df)

    assert not out.isna().any().any(), "Rolling features should have no NaNs"
    assert len(out) == len(df)
    expected_cols = [f"sensor_{i}_w{w}_{s}" for i in (1, 2) for w in (5, 15) for s in ("mean", "std")]
    for c in expected_cols:
        assert c in out.columns, f"Missing column {c}"


def test_sequence_dataset_no_lookahead():
    n_engines = 2
    n_cycles = 40
    frames = []
    for eng in range(1, n_engines + 1):
        frames.append(
            pd.DataFrame(
                {
                    "unit_number": eng,
                    "time_in_cycles": np.arange(1, n_cycles + 1),
                    "sensor_1": np.arange(n_cycles, dtype=float),
                    "rul_capped": np.arange(n_cycles - 1, -1, -1).astype(float),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    ds = RULSequenceDataset(
        df,
        feature_columns=["sensor_1"],
        target_column="rul_capped",
        window_length=10,
        stride=1,
    )
    assert len(ds) > 0
    # First sequence's target should equal rul at cycle index 9 (0-based end of first window)
    x, y, eng, cyc = ds[0]
    assert x.shape == (10, 1)
    # rul decreases linearly; at cycle 10 (index 9), rul = 40-1-9 = 30
    assert abs(float(y) - 30.0) < 1e-4


def test_build_model_all_architectures():

    cfg = load_config("configs/config.yaml")
    for arch in ["lstm", "gru", "cnn_lstm", "attention_lstm"]:
        cfg.model.architecture = arch
        model = build_model(cfg, input_size=5)
        x = torch.randn(2, cfg.sequences.window_length, 5)
        out = model(x)
        assert out.shape == (2, 1), f"{arch} output shape wrong: {out.shape}"


def test_config_loads():
    cfg = load_config("configs/config.yaml")
    assert cfg.dataset.name == EXPECTED_DATASET_NAME
    assert cfg.preprocessing.rul_cap == EXPECTED_RUL_CAP
    assert cfg.sequences.window_length == EXPECTED_WINDOW_LENGTH
    assert cfg.training.seed == EXPECTED_SEED


def test_data_cleaner_drops_constant():
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 2, 2],
            "time_in_cycles": [1, 2, 1, 2],
            "sensor_1": [5.0, 5.0, 5.0, 5.0],  # constant
            "sensor_2": [1.0, 2.0, 3.0, 4.0],
            "rul": [10.0, 9.0, 10.0, 9.0],
        }
    )
    cleaner = DataCleaner(variance_threshold=1e-6)
    cleaner.fit(df)
    assert "sensor_1" not in cleaner.sensor_columns_
    assert "sensor_2" in cleaner.sensor_columns_
