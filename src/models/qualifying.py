"""
Qualifying prediction models.

Implements LightGBM LambdaRank model for predicting top-3 qualifying positions.
"""

import logging
from pathlib import Path

import joblib

from src.models.base_ranker import BaseLGBMRanker

logger = logging.getLogger(__name__)


def save_model(model: object, model_path: Path, metadata: dict | None = None):
    """
    Save trained model to disk.

    Args:
        model: Trained model instance
        model_path: Path to save model
        metadata: Optional metadata to save alongside model
    """
    logger.info(f"Saving model to {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, model_path)

    # Save metadata if provided
    if metadata:
        metadata_path = model_path.with_suffix(".json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info("Model saved successfully")


def load_model(model_path: Path) -> object:
    """
    Load trained model from disk.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model instance
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return model


class QualifyingLGBMRanker(BaseLGBMRanker):
    """
    LightGBM Learning-to-Rank model for qualifying predictions.

    Extends BaseLGBMRanker with qualifying-specific defaults.
    Uses LambdaRank objective for pairwise ranking optimization.
    Each qualifying session forms a query group for ranking.
    """

    session_type = "Q"

    def __init__(
        self,
        objective: str = "lambdarank",
        n_estimators: int = 259,
        num_leaves: int = 15,
        learning_rate: float = 0.082,
        min_child_samples: int = 30,
        subsample: float = 0.766,
        colsample_bytree: float = 0.782,
        reg_alpha: float = 0.000685,
        reg_lambda: float = 0.0000783,
        random_state: int = 42,
        label_gain: list[float] | None = None,
    ):
        """
        Initialize Qualifying LGBMRanker model.

        Hyperparameters tuned via Optuna (50 trials, 5-fold temporal CV).
        Best CV score: 3.07 avg game points.

        Args:
            objective: Ranking objective ("lambdarank" or "rank_xendcg")
            n_estimators: Number of boosting rounds
            num_leaves: Maximum tree leaves
            learning_rate: Learning rate
            min_child_samples: Minimum data in a leaf
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            label_gain: Relevance gain for positions (default: inverse position)
        """
        super().__init__(
            objective=objective,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            label_gain=label_gain,
        )
