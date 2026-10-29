"""Non-deep-learning baselines: XGBoost and Random Forest regressors.

Trains on hand-built rolling features with group-aware engine splits to prevent
data leakage. Logs RMSE/MAE and persists models + feature importances as joblib.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import load_config

logger = logging.getLogger(__name__)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def evaluate_model(name: str, y_true, y_pred) -> dict[str, float]:
    metrics = {
        "model": name,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }
    logger.info("%s -> RMSE=%.4f MAE=%.4f R2=%.4f", name, metrics["rmse"], metrics["mae"], metrics["r2"])
    return metrics


class BaselineTrainer:
    """Trains and tunes XGBoost / Random Forest baselines."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.models_: dict[str, Pipeline] = {}
        self.results_: list[dict] = []

    def _make_pipeline(self, model_name: str) -> tuple[Pipeline, dict]:
        preprocessor = StandardScaler()
        if model_name == "xgboost":
            model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                objective="reg:squarederror",
            )
            param_dist = {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [4, 6, 8, 10],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.8, 0.9],
                "model__colsample_bytree": [0.7, 0.8, 0.9],
            }
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            )
            param_dist = {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5],
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)],
        )
        return pipeline, param_dist

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups: pd.Series,
        model_name: str,
        n_iter: int = 30,
        cv: int = 5,
    ) -> Pipeline:
        """Fit with RandomizedSearchCV using GroupKFold (engine-grouped)."""
        pipeline, param_dist = self._make_pipeline(model_name)
        cv_splitter = GroupKFold(n_splits=cv)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv_splitter,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train, groups=groups)

        self.models_[model_name] = search.best_estimator_
        logger.info(
            "%s best params: %s (best CV RMSE=%.4f)",
            model_name,
            search.best_params_,
            -search.best_score_,
        )
        return search.best_estimator_

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not fitted yet.")
        return self.models_[model_name].predict(X)

    def feature_importance(self, model_name: str, feature_names: list[str]) -> pd.DataFrame:
        model = self.models_[model_name].named_steps["model"]
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            return imp
        return pd.DataFrame()

    def save(self, model_name: str, path: str | Path) -> None:
        joblib.dump(self.models_[model_name], path)
        logger.info("Saved %s model to %s", model_name, path)


def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Convenience: train XGBoost + Random Forest, evaluate, save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = BaselineTrainer(random_state=random_state)
    results: dict[str, dict[str, float]] = {}

    for name in ("xgboost", "random_forest"):
        trainer.fit(X_train, y_train, groups, model_name=name)
        y_pred = trainer.predict(name, X_test)
        results[name] = evaluate_model(name, y_test, y_pred)
        trainer.save(name, output_dir / f"{name}_baseline.joblib")

        imp = trainer.feature_importance(name, list(X_train.columns))
        if not imp.empty:
            imp.to_csv(output_dir / f"{name}_feature_importance.csv", index=False)

    # Save metrics
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Train XGBoost / Random Forest baselines")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config.dataset.processed_dir)

    import pandas as pd

    train = pd.read_parquet(processed_dir / "train_processed.parquet")
    test = pd.read_parquet(processed_dir / "test_processed.parquet")
    rul_test = pd.read_parquet(processed_dir / "rul_test.parquet")
    test = test.merge(rul_test, on="unit_number", how="left")

    from src.features.rolling import RollingFeatures

    featurizer = RollingFeatures(
        windows=config.features.rolling_windows,
        stats=config.features.rolling_stats,
    )
    featurizer.fit(train)
    train_feat = featurizer.transform(train)
    test_feat = featurizer.transform(test)

    feature_cols = [
        c for c in train_feat.columns if c not in {"unit_number", "time_in_cycles", "rul", "rul_capped"}
    ]

    train_last = train_feat.sort_values("time_in_cycles").groupby("unit_number").tail(1)
    test_last = test_feat.sort_values("time_in_cycles").groupby("unit_number").tail(1)

    X_train = train_last[feature_cols]
    y_train = train_last["rul_capped"]
    X_test = test_last[feature_cols]
    # Map each test engine's last-cycle row to its ground-truth RUL via unit_number
    rul_lookup = test[["unit_number", "rul"]].drop_duplicates("unit_number").set_index("unit_number")["rul"]
    y_test = test_last["unit_number"].map(rul_lookup).reset_index(drop=True)

    # Defensive: ensure no NaN in features/targets (sklearn will reject them)
    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)
    y_train = y_train.fillna(0.0)
    y_test = y_test.fillna(0.0)

    groups = train_last["unit_number"].reset_index(drop=True)

    train_baselines(
        X_train, y_train, groups, X_test, y_test, output_dir="models/baselines"
    )
    logger.info("Baseline training complete.")
