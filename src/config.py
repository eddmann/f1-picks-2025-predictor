"""
Centralized configuration management for F1 Picks ML system.

Configuration is loaded from:
1. Default values defined in this module
2. Environment variables (F1_* prefix)
3. Optional config file (config.yaml)

Usage:
    from src.config import config

    data_dir = config.data_dir
    model_params = config.model.qualifying
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file and go up to find pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to current working directory
    return Path.cwd()


@dataclass
class DataConfig:
    """Data storage configuration."""

    data_dir: Path = field(default_factory=lambda: _get_project_root() / "data" / "fastf1")
    cache_dir: Path = field(default_factory=lambda: _get_project_root() / ".fastf1_cache")
    predictions_dir: Path = field(
        default_factory=lambda: _get_project_root() / "data" / "predictions"
    )

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.predictions_dir = Path(self.predictions_dir)

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"


@dataclass
class ModelConfig:
    """Model training configuration."""

    models_dir: Path = field(default_factory=lambda: _get_project_root() / "models" / "saved")
    min_year: int = 2020
    cv_splits: int = 5
    random_state: int = 42

    def __post_init__(self):
        self.models_dir = Path(self.models_dir)


@dataclass
class QualifyingModelParams:
    """LightGBM parameters for qualifying model."""

    objective: str = "lambdarank"
    n_estimators: int = 100
    num_leaves: int = 31
    learning_rate: float = 0.1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0


@dataclass
class RaceModelParams:
    """LightGBM parameters for race model (Optuna-tuned)."""

    objective: str = "lambdarank"
    n_estimators: int = 192
    num_leaves: int = 20
    learning_rate: float = 0.133
    min_child_samples: int = 39
    subsample: float = 0.74
    colsample_bytree: float = 0.61
    reg_alpha: float = 0.003
    reg_lambda: float = 0.009


@dataclass
class SprintModelParams:
    """LightGBM parameters for sprint models (higher regularization)."""

    objective: str = "lambdarank"
    n_estimators: int = 50
    num_leaves: int = 15
    learning_rate: float = 0.05
    min_child_samples: int = 30
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1


@dataclass
class ModelParams:
    """All model parameters."""

    qualifying: QualifyingModelParams = field(default_factory=QualifyingModelParams)
    race: RaceModelParams = field(default_factory=RaceModelParams)
    sprint_quali: SprintModelParams = field(default_factory=SprintModelParams)
    sprint_race: SprintModelParams = field(default_factory=SprintModelParams)


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    rolling_windows: list[int] = field(default_factory=lambda: [3, 5, 10])
    ewm_windows: list[int] = field(default_factory=lambda: [3, 5])
    elo_k_driver: int = 32
    elo_k_constructor: int = 24
    elo_initial_rating: float = 1500.0


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    model_params: ModelParams = field(default_factory=ModelParams)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Convenience properties for common paths
    @property
    def data_dir(self) -> Path:
        return self.data.data_dir

    @property
    def models_dir(self) -> Path:
        return self.model.models_dir

    @property
    def cache_dir(self) -> Path:
        return self.data.cache_dir

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        self.data.data_dir.mkdir(parents=True, exist_ok=True)
        self.data.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.data.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.data.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model.models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        # Override data paths from environment
        if data_dir := os.environ.get("F1_DATA_DIR"):
            config.data.data_dir = Path(data_dir)
        if cache_dir := os.environ.get("F1_CACHE_DIR"):
            config.data.cache_dir = Path(cache_dir)
        if models_dir := os.environ.get("F1_MODELS_DIR"):
            config.model.models_dir = Path(models_dir)

        # Override model settings
        if min_year := os.environ.get("F1_MIN_YEAR"):
            config.model.min_year = int(min_year)
        if cv_splits := os.environ.get("F1_CV_SPLITS"):
            config.model.cv_splits = int(cv_splits)

        return config

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        config = cls()

        if not path.exists():
            return config

        with open(path) as f:
            yaml_config = yaml.safe_load(f) or {}

        # Apply YAML overrides
        if data_config := yaml_config.get("data"):
            if data_dir := data_config.get("data_dir"):
                config.data.data_dir = Path(data_dir)
            if cache_dir := data_config.get("cache_dir"):
                config.data.cache_dir = Path(cache_dir)

        if model_config := yaml_config.get("model"):
            if models_dir := model_config.get("models_dir"):
                config.model.models_dir = Path(models_dir)
            if min_year := model_config.get("min_year"):
                config.model.min_year = int(min_year)

        return config

    @classmethod
    def load(cls) -> "Config":
        """
        Load configuration with precedence:
        1. Default values
        2. config.yaml (if exists)
        3. Environment variables (highest priority)
        """
        # Start with defaults
        config = cls()

        # Try to load from config.yaml
        config_path = _get_project_root() / "config.yaml"
        if config_path.exists():
            config = cls.from_yaml(config_path)

        # Apply environment variable overrides
        env_config = cls.from_env()

        # Merge env overrides (env takes precedence)
        if os.environ.get("F1_DATA_DIR"):
            config.data.data_dir = env_config.data.data_dir
        if os.environ.get("F1_CACHE_DIR"):
            config.data.cache_dir = env_config.data.cache_dir
        if os.environ.get("F1_MODELS_DIR"):
            config.model.models_dir = env_config.model.models_dir
        if os.environ.get("F1_MIN_YEAR"):
            config.model.min_year = env_config.model.min_year
        if os.environ.get("F1_CV_SPLITS"):
            config.model.cv_splits = env_config.model.cv_splits

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return {
            "data": {
                "data_dir": str(self.data.data_dir),
                "cache_dir": str(self.data.cache_dir),
                "predictions_dir": str(self.data.predictions_dir),
            },
            "model": {
                "models_dir": str(self.model.models_dir),
                "min_year": self.model.min_year,
                "cv_splits": self.model.cv_splits,
                "random_state": self.model.random_state,
            },
            "features": {
                "rolling_windows": self.features.rolling_windows,
                "ewm_windows": self.features.ewm_windows,
                "elo_k_driver": self.features.elo_k_driver,
                "elo_k_constructor": self.features.elo_k_constructor,
            },
        }


# Global config instance - use this throughout the application
config = Config.load()
