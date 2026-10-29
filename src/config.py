from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    train_file: str
    test_file: str
    rul_file: str
    data_dir: str
    processed_dir: str
    columns: List[str]


@dataclass
class PreprocessingConfig:
    variance_threshold: float
    corr_threshold: float
    rul_cap: int
    scaler_type: str
    fit_on_train_only: bool


@dataclass
class FeaturesConfig:
    rolling_windows: List[int]
    rolling_stats: List[str]


@dataclass
class SequencesConfig:
    window_length: int
    stride: int
    target_column: str


@dataclass
class ModelConfig:
    architecture: str
    input_size: Optional[int]
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    attention: bool
    cnn_filters: List[int]
    cnn_kernel: int


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    gradient_clip_norm: float
    lr_scheduler: str
    seed: int
    val_split: float
    group_split_by_engine: bool


@dataclass
class EvaluationConfig:
    metrics: List[str]
    nasa_alpha_early: int
    nasa_alpha_late: int
    bootstrap_samples: int


@dataclass
class DeploymentConfig:
    model_path: str
    scaler_path: str
    columns_path: str
    rul_alert_threshold: int
    api_host: str
    api_port: int


@dataclass
class Config:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    features: FeaturesConfig
    sequences: SequencesConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    deployment: DeploymentConfig
    raw: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    return Config(
        dataset=DatasetConfig(**data["dataset"]),
        preprocessing=PreprocessingConfig(**data["preprocessing"]),
        features=FeaturesConfig(**data["features"]),
        sequences=SequencesConfig(**data["sequences"]),
        model=ModelConfig(**data["model"]),
        training=TrainingConfig(**data["training"]),
        evaluation=EvaluationConfig(**data["evaluation"]),
        deployment=DeploymentConfig(**data["deployment"]),
        raw=data,
    )
