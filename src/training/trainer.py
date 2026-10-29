"""Training loop for RUL sequence models.

Handles DataLoader creation, training/validation, early stopping, checkpointing,
gradient clipping, LR scheduling, and TorchScript export.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.config import Config, load_config
from src.data.sequences import RULSequenceDataset, build_datasets
from src.models.lstm_model import build_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RULTrainer:
    def __init__(self, config: Config, device: Optional[str] = None) -> None:
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.best_val_rmse: float = float("inf")
        self.best_state: Optional[dict] = None
        self.history: list[dict] = []
        self.input_size: Optional[int] = None

    def _make_loaders(
        self,
        train_ds: RULSequenceDataset,
        val_split: float,
    ) -> tuple[DataLoader, DataLoader, int]:
        # Engine-grouped split: no engine appears in both train and validation,
        # preventing leakage from overlapping windowed sequences of the same engine.
        engine_ids = train_ds.engine_ids
        assert engine_ids is not None
        unique_engines = np.unique(engine_ids)
        rng = np.random.default_rng(self.config.training.seed)
        permuted = rng.permutation(unique_engines)
        n_val_engines = max(1, int(len(permuted) * val_split))
        val_engines = set(permuted[:n_val_engines].tolist())
        train_engines = set(permuted[n_val_engines:].tolist())

        train_indices = [
            i for i, e in enumerate(engine_ids.tolist()) if e in train_engines
        ]
        val_indices = [
            i for i, e in enumerate(engine_ids.tolist()) if e in val_engines
        ]

        from torch.utils.data import Subset

        train_subset = Subset(train_ds, train_indices)
        val_subset = Subset(train_ds, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return train_loader, val_loader, len(train_ds)

    def fit(
        self,
        train_df,
        test_df,
        feature_columns: list[str],
        model_dir: str | Path = "models",
        rul_test: Optional[pd.Series] = None,
    ) -> dict:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        train_ds, test_ds = build_datasets(
            train_df,
            test_df,
            feature_columns,
            self.config,
            rul_test=rul_test,
        )
        self.input_size = len(feature_columns)

        self.model = build_model(self.config, self.input_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.training.learning_rate),
            weight_decay=float(self.config.training.weight_decay),
        )

        # LR scheduler
        sched_name = self.config.training.lr_scheduler.lower()
        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.epochs
            )
        elif sched_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.config.training.epochs // 3), gamma=0.5
            )
        else:
            self.scheduler = None

        train_loader, val_loader, n_total = self._make_loaders(
            train_ds, self.config.training.val_split
        )
        logger.info("Training on %s sequences, device=%s", n_total, self.device)

        criterion = nn.MSELoss()
        patience_counter = 0

        for epoch in range(1, self.config.training.epochs + 1):
            train_loss = self._train_epoch(train_loader, criterion)
            val_rmse, val_mae = self._evaluate(val_loader, criterion)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse, "val_mae": val_mae}
            )
            logger.info(
                "Epoch %d/%d train_loss=%.4f val_rmse=%.4f val_mae=%.4f",
                epoch,
                self.config.training.epochs,
                train_loss,
                val_rmse,
                val_mae,
            )

            # Early stopping + checkpoint
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                torch.save(self.model.state_dict(), model_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # Export TorchScript
        self.model.eval()
        example_input = torch.randn(1, self.config.sequences.window_length, self.input_size).to(self.device)
        scripted = torch.jit.trace(self.model, example_input)
        torch.jit.save(scripted, model_dir / "best_model_scripted.pt")

        # Save metadata
        meta = {
            "architecture": self.config.model.architecture,
            "input_size": self.input_size,
            "window_length": self.config.sequences.window_length,
            "feature_columns": feature_columns,
            "best_val_rmse": self.best_val_rmse,
            "rul_cap": self.config.preprocessing.rul_cap,
        }
        with open(model_dir / "model_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Training complete. Best val RMSE=%.4f", self.best_val_rmse)
        return meta

    def _train_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        model = self.model
        optimizer = self.optimizer
        assert model is not None
        assert optimizer is not None
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x_batch, y_batch, _, _ in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(-1)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()

            if self.config.training.gradient_clip_norm and self.config.training.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clip_norm)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        model = self.model
        assert model is not None
        model.eval()
        all_preds = []
        all_targets = []
        for x_batch, y_batch, _, _ in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(-1)
            preds = model(x_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
        error = np.concatenate(all_preds) - np.concatenate(all_targets)
        rmse = float(math.sqrt(np.mean(error ** 2)))
        mae = float(np.mean(np.abs(error)))
        return rmse, mae

    @torch.no_grad()
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        model = self.model
        if model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        model.eval()
        x = torch.from_numpy(sequences.astype(np.float32)).to(self.device)
        preds = model(x)
        return preds.cpu().numpy().squeeze(-1)


if __name__ == "__main__":
    import argparse

    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Train LSTM/GRU RUL model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config.dataset.processed_dir)

    train = pd.read_parquet(processed_dir / "train_processed.parquet")
    test = pd.read_parquet(processed_dir / "test_processed.parquet")
    rul_test = pd.read_parquet(processed_dir / "rul_test.parquet")
    test = test.merge(rul_test, on="unit_number", how="left").rename(columns={"rul": "rul_capped"})

    import json

    with open(processed_dir / "feature_columns.json", "r", encoding="utf-8") as fh:
        feature_columns = json.load(fh)

    trainer = RULTrainer(config)
    meta = trainer.fit(
        train, test, feature_columns, model_dir=args.model_dir
    )
    logger.info("Training complete. Meta: %s", meta)
