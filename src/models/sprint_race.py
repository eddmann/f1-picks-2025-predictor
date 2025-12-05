"""
Sprint Race prediction models.

Implements ML models for predicting top-3 sprint race finishers.
Uses LightGBM Learning-to-Rank with sprint-specific optimizations.

Sprint races are shorter (~20 laps) than main races (~50-70 laps), meaning:
- Grid position is even more important (less time for position changes)
- First lap performance is crucial
- Strategy plays less role than in main race
"""

import logging

from src.models.base_ranker import BaseLGBMRanker

logger = logging.getLogger(__name__)


class SprintRaceLGBMRanker(BaseLGBMRanker):
    """
    LightGBM Learning-to-Rank model for sprint race predictions.

    Optimized for sprint races where:
    - Grid position (from SQ) is the most important feature
    - Race is shorter (~20 laps), so fewer position changes
    - First lap performance is crucial
    - Limited data (~6 events/year since 2021)

    Uses stronger regularization similar to sprint qualifying.
    """

    session_type = "S"

    def __init__(
        self,
        objective: str = "lambdarank",
        n_estimators: int = 80,  # Tuned via Optuna
        num_leaves: int = 39,  # Tuned via Optuna
        learning_rate: float = 0.0112,  # Tuned via Optuna
        min_child_samples: int = 46,  # Tuned via Optuna
        subsample: float = 0.629,  # Tuned via Optuna
        colsample_bytree: float = 0.831,  # Tuned via Optuna
        reg_alpha: float = 6.4e-06,  # Tuned via Optuna
        reg_lambda: float = 0.00048,  # Tuned via Optuna
        random_state: int = 42,
        label_gain: list[float] | None = None,
    ):
        """
        Initialize Sprint Race LGBMRanker model.

        Uses conservative hyperparameters due to:
        - Limited sprint data (~6 events/year)
        - Shorter race format with fewer overtakes

        Args:
            objective: Ranking objective ("lambdarank" or "rank_xendcg")
            n_estimators: Number of boosting rounds
            num_leaves: Maximum tree leaves
            learning_rate: Learning rate
            min_child_samples: Minimum data in a leaf
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns
            random_state: Random seed
            label_gain: Relevance gain for positions
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
