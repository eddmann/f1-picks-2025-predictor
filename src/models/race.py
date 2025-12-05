"""
Race prediction models.

Implements ML models for predicting top-3 race finishers.
Uses LightGBM Learning-to-Rank with grid position as key feature.
"""

import logging

from src.models.base_ranker import BaseLGBMRanker

logger = logging.getLogger(__name__)


class RaceLGBMRanker(BaseLGBMRanker):
    """
    LightGBM Learning-to-Rank model for race predictions.

    Optimized for race finish prediction where grid position is the most
    important feature (~60-70% of variance in race results).

    Uses LambdaRank objective for pairwise ranking optimization.
    Each race session forms a query group for ranking.
    """

    session_type = "R"

    def __init__(
        self,
        objective: str = "lambdarank",
        n_estimators: int = 80,
        num_leaves: int = 39,
        learning_rate: float = 0.0112,
        min_child_samples: int = 46,
        subsample: float = 0.629,
        colsample_bytree: float = 0.831,
        reg_alpha: float = 6.4e-06,
        reg_lambda: float = 0.00048,
        random_state: int = 42,
        label_gain: list[float] | None = None,
    ):
        """
        Initialize Race LGBMRanker model.

        Hyperparameters tuned via Optuna (50 trials, 5-fold temporal CV).
        Best CV score: 3.15 avg game points.

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
