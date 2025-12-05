"""
Sprint Qualifying prediction models.

Implements ML models for predicting top-3 sprint qualifying finishers.
Uses LightGBM Learning-to-Rank with stronger regularization due to limited data.

Sprint qualifying has limited data (~6 events/year since 2021), so:
- Uses higher min_child_samples for regularization
- Uses more conservative num_leaves
- Relies more on historical features
"""

import logging

from src.models.base_ranker import BaseLGBMRanker

logger = logging.getLogger(__name__)


class SprintQualiLGBMRanker(BaseLGBMRanker):
    """
    LightGBM Learning-to-Rank model for sprint qualifying predictions.

    Optimized for sprint qualifying where:
    - Data is limited (~6 events/year since 2021)
    - Only FP1 data from current weekend is available
    - Performance correlates with main qualifying

    Uses stronger regularization to prevent overfitting on limited data.
    """

    session_type = "SQ"

    def __init__(
        self,
        objective: str = "lambdarank",
        n_estimators: int = 298,  # Tuned via Optuna
        num_leaves: int = 47,  # Tuned via Optuna
        learning_rate: float = 0.182,  # Tuned via Optuna
        min_child_samples: int = 50,  # Higher regularization
        subsample: float = 0.871,  # Tuned via Optuna
        colsample_bytree: float = 0.781,  # Tuned via Optuna
        reg_alpha: float = 5.5e-07,  # Tuned via Optuna
        reg_lambda: float = 2.5e-06,  # Tuned via Optuna
        random_state: int = 42,
        label_gain: list[float] | None = None,
    ):
        """
        Initialize Sprint Qualifying LGBMRanker model.

        Uses more conservative hyperparameters due to limited sprint data.

        Args:
            objective: Ranking objective ("lambdarank" or "rank_xendcg")
            n_estimators: Number of boosting rounds (default lower for limited data)
            num_leaves: Maximum tree leaves (default lower for regularization)
            learning_rate: Learning rate (default lower for limited data)
            min_child_samples: Minimum data in a leaf (default higher for regularization)
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
