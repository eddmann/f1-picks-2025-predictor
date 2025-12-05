"""
Feature selection utilities based on importance scores.

Reduces feature set to top N most important features to reduce overfitting
and improve model interpretability. With 120 features and ~1500 samples,
we risk overfitting; reducing to 30-50 features improves generalization.
"""

import logging
from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

logger = logging.getLogger(__name__)


def get_nonzero_features(
    model,
    feature_names: list[str],
    min_importance: float = 0.0,
) -> list[str]:
    """
    Get features with non-zero importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
               (can be LGBMRanker, RandomForest, etc.)
        feature_names: List of feature names matching model's features
        min_importance: Minimum importance threshold (default 0 = only remove zeros)

    Returns:
        List of feature names with importance > min_importance
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model must have feature_importances_ attribute")

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: model has {len(importances)}, got {len(feature_names)} names"
        )

    nonzero_features = [
        f for f, imp in zip(feature_names, importances, strict=False) if imp > min_importance
    ]

    logger.info(
        f"Feature selection: {len(nonzero_features)}/{len(feature_names)} "
        f"features have importance > {min_importance}"
    )

    return nonzero_features


def select_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 40,
    method: Literal["top_n", "threshold", "cumulative"] = "top_n",
    cumulative_threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select top features based on RandomForest importance.

    Args:
        X: Feature matrix
        y: Target labels
        n_features: Number of features to keep (for top_n method)
        method: Selection method:
            - "top_n": Keep top N features by importance
            - "threshold": Use sklearn's SelectFromModel
            - "cumulative": Keep features until cumulative importance >= threshold
        cumulative_threshold: Cumulative importance threshold (for cumulative method)

    Returns:
        Tuple of (filtered X, list of selected feature names)
    """
    logger.info(f"Selecting features using {method} method...")
    logger.info(f"Input features: {len(X.columns)}")

    # Train a quick RF to get importances
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    if method == "top_n":
        selected_features = importances.head(n_features).index.tolist()
    elif method == "threshold":
        selector = SelectFromModel(rf, prefit=True, max_features=n_features)
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
    elif method == "cumulative":
        cumsum = importances.cumsum()
        # Find how many features needed to reach threshold
        n_needed = (cumsum < cumulative_threshold).sum() + 1
        n_needed = max(n_needed, 10)  # Minimum 10 features
        selected_features = importances.head(n_needed).index.tolist()
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Selected {len(selected_features)} features")
    logger.info(f"Top 10: {selected_features[:10]}")

    return X[selected_features], selected_features
