"""
Base LightGBM ranking model for F1 predictions.

Provides abstract base class for session-type-specific ranker models.
Uses LambdaRank objective for pairwise ranking optimization.

Security Note - Model Serialization:
    Models are saved using joblib/pickle format for convenience and compatibility.
    Pickle files can execute arbitrary code during deserialization.

    ONLY load models from trusted sources:
    - Models you trained yourself
    - Models from this repository's releases
    - Models from trusted teammates

    DO NOT load models from:
    - Unknown internet sources
    - Untrusted third parties
    - User-uploaded files without verification

    If you need to load untrusted models, consider:
    - Exporting to ONNX format (safer, cross-platform)
    - Using the .lgbm.txt format (LightGBM native, human-readable)
    - Implementing hash verification for model files
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseLGBMRanker:
    """
    Base LightGBM Learning-to-Rank model for F1 predictions.

    Uses LambdaRank objective for pairwise ranking optimization.
    Each race session forms a query group for ranking.
    Subclasses define session-specific hyperparameter defaults.
    """

    # Session type identifier (Q, SQ, S, R) - set by subclass
    session_type: str = ""

    def __init__(
        self,
        objective: str = "lambdarank",
        n_estimators: int = 100,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int = 42,
        label_gain: list[float] | None = None,
    ):
        """
        Initialize LightGBM Ranker model.

        Args:
            objective: Ranking objective ("lambdarank" or "rank_xendcg")
            n_estimators: Number of boosting rounds
            num_leaves: Maximum tree leaves
            learning_rate: Learning rate
            min_child_samples: Minimum data in a leaf
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed
            label_gain: Relevance gain for positions (default: inverse position)
        """
        self.objective = objective
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.label_gain = label_gain or self._default_label_gain()

        self.model = None
        self.feature_names = None
        logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(session_type={self.session_type}, objective={objective})"
        )

    def _default_label_gain(self) -> list[float]:
        """
        Define relevance gain for positions 1-20.
        Higher gain for top positions (P1 is most valuable).
        """
        # Inverse position weighting: P1=20, P2=19, ..., P20=1
        return [21 - i for i in range(1, 22)]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group: np.ndarray,
        eval_set: tuple | None = None,
        eval_group: np.ndarray | None = None,
    ) -> "BaseLGBMRanker":
        """
        Train the ranker model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Relevance labels (higher = better, e.g., P1=20, P20=1)
            group: Array of group sizes (number of drivers per race)
            eval_set: Optional (X_val, y_val) tuple for validation
            eval_group: Group sizes for validation set

        Returns:
            Self for method chaining
        """
        import lightgbm as lgb

        logger.info(
            f"Training {self.__class__.__name__} on {len(X)} samples, {len(X.columns)} features"
        )
        logger.info(f"Groups: {len(group)} sessions, avg {np.mean(group):.1f} drivers per session")

        self.feature_names = X.columns.tolist()

        self.model = lgb.LGBMRanker(
            objective=self.objective,
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            importance_type="gain",
            n_jobs=-1,
            verbose=-1,
        )

        fit_params = {"group": group}
        if eval_set is not None and eval_group is not None:
            fit_params["eval_set"] = [eval_set]
            fit_params["eval_group"] = [eval_group]

        self.model.fit(X, y, **fit_params)
        logger.info("Training complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict relevance scores for ranking.

        Returns raw scores - higher score = better predicted position.
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_top3(
        self,
        X: pd.DataFrame,
        driver_codes: list[str],
    ) -> dict:
        """
        Extract top-3 predictions with confidence scores.

        Args:
            X: Feature matrix for all drivers in a single session
            driver_codes: List of driver codes corresponding to rows in X

        Returns:
            Dict with 'top3' (ordered list) and 'predictions' (detailed info)
        """
        scores = self.predict(X)

        # Create DataFrame for sorting
        results = pd.DataFrame(
            {
                "driver_code": driver_codes,
                "score": scores,
            }
        )
        results = results.sort_values("score", ascending=False)

        # Calculate confidence as normalized score gaps
        max_score = results["score"].max()
        min_score = results["score"].min()
        score_range = max_score - min_score if max_score != min_score else 1.0

        predictions = []
        for i, (_, row) in enumerate(results.head(3).iterrows()):
            confidence = ((row["score"] - min_score) / score_range) * 100

            predictions.append(
                {
                    "position": i + 1,
                    "driver_code": row["driver_code"],
                    "score": float(row["score"]),
                    "confidence": round(confidence, 1),
                }
            )

        return {
            "top3": [p["driver_code"] for p in predictions],
            "predictions": predictions,
            "full_ranking": results["driver_code"].tolist(),
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Extract feature importance scores."""
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be trained first")

        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances, strict=False))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def prepare_ranking_data(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    """
    Transform data for LightGBM Ranker training.

    Args:
        X: Feature matrix
        y: Position labels (1-20, lower is better)
        meta: Metadata with session_key for grouping

    Returns:
        Tuple of (X_sorted, y_relevance, group_sizes, meta_sorted)
        - X_sorted: Features sorted by session
        - y_relevance: Relevance labels (higher = better, inverted from position)
        - group_sizes: Array of drivers per session for LGBMRanker group param
        - meta_sorted: Metadata sorted to match X_sorted
    """
    # Combine for sorting
    data = X.copy()
    data["_position"] = y.values
    data["_session_key"] = meta["session_key"].values
    data["_original_idx"] = X.index

    # Sort by session to ensure contiguous groups
    data = data.sort_values(["_session_key", "_position"])

    # Calculate group sizes (drivers per session)
    group_sizes = data.groupby("_session_key").size().values

    # Convert position to relevance (higher = better)
    max_position = 20
    y_relevance = max_position - data["_position"].clip(upper=max_position) + 1

    # Get sorted metadata with position
    meta_sorted = meta.loc[data["_original_idx"]].copy()
    meta_sorted["position"] = data["_position"].values
    meta_sorted.index = range(len(meta_sorted))

    # Remove helper columns and reset index
    X_sorted = data.drop(columns=["_position", "_session_key", "_original_idx"])
    X_sorted.index = range(len(X_sorted))
    y_relevance.index = range(len(y_relevance))

    return X_sorted, y_relevance.astype(int), group_sizes.astype(int), meta_sorted


def create_temporal_cv_splits_with_groups(
    meta: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create temporal cross-validation splits that respect session groups.

    LGBMRanker requires complete groups, so we split by sessions not samples.

    Returns:
        List of (train_idx, val_idx, train_groups, val_groups) tuples
    """
    # Get unique sessions in temporal order
    session_order = meta["session_key"].unique().tolist()
    n_sessions = len(session_order)

    # Create mapping of session to sample indices
    session_to_idx = {}
    for session in session_order:
        session_to_idx[session] = np.where(meta["session_key"] == session)[0]

    fold_size = n_sessions // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        val_end = min(train_end + fold_size, n_sessions)

        train_sessions = session_order[:train_end]
        val_sessions = session_order[train_end:val_end]

        if not val_sessions:
            continue

        train_idx = np.concatenate([session_to_idx[s] for s in train_sessions])
        val_idx = np.concatenate([session_to_idx[s] for s in val_sessions])

        train_groups = np.array([len(session_to_idx[s]) for s in train_sessions])
        val_groups = np.array([len(session_to_idx[s]) for s in val_sessions])

        splits.append((train_idx, val_idx, train_groups, val_groups))

    return splits


def train_ranker_model(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    ranker_class: type,
    objective: str = "lambdarank",
    cv_splits: int = 5,
    verbose_diagnostics: bool = True,
    **model_kwargs,
) -> tuple["BaseLGBMRanker", dict]:
    """
    Train LightGBM ranking model with temporal cross-validation.

    Generic training function that works with any BaseLGBMRanker subclass.

    Args:
        X: Feature matrix
        y: Position labels (will be converted to relevance)
        meta: Metadata with session_key for grouping
        ranker_class: Subclass of BaseLGBMRanker to instantiate
        objective: "lambdarank" or "rank_xendcg"
        cv_splits: Number of temporal CV splits
        verbose_diagnostics: If True, print detailed diagnostic report
        **model_kwargs: Arguments for ranker initialization

    Returns:
        Tuple of (trained model, results dict with CV scores and diagnostics)
    """
    from src.evaluation.ranking_metrics import DiagnosticCollector
    from src.evaluation.scoring import calculate_game_points

    logger.info(f"Training {ranker_class.__name__} with {objective} objective")

    # Prepare ranking data
    X_sorted, y_relevance, group_sizes, meta_sorted = prepare_ranking_data(X, y, meta)

    # Temporal CV with group-aware splits
    splits = create_temporal_cv_splits_with_groups(meta_sorted, n_splits=cv_splits)

    cv_scores = []
    # Collect diagnostics across all CV folds
    diagnostic_collector = DiagnosticCollector()

    for fold_idx, (train_idx, val_idx, train_groups, _val_groups) in enumerate(splits):
        X_train = X_sorted.iloc[train_idx]
        y_train = y_relevance.iloc[train_idx]
        X_val = X_sorted.iloc[val_idx]
        meta_val = meta_sorted.iloc[val_idx]

        # Train fold model
        fold_model = ranker_class(objective=objective, **model_kwargs)
        fold_model.fit(X_train, y_train, train_groups)

        # Evaluate on validation set using game points
        fold_game_points = []
        for session_key in meta_val["session_key"].unique():
            session_mask = meta_val["session_key"] == session_key
            session_X = X_val.loc[session_mask]
            session_meta = meta_val.loc[session_mask]

            if len(session_X) < 3:
                continue

            session_with_pos = session_meta[["driver_code", "position"]].copy()
            actual_top3 = session_with_pos.nsmallest(3, "position")["driver_code"].tolist()

            if len(actual_top3) < 3:
                continue

            pred_result = fold_model.predict_top3(
                session_X,
                session_meta["driver_code"].tolist(),
            )
            pred_top3 = pred_result["top3"]

            points = calculate_game_points(pred_top3, actual_top3)
            fold_game_points.append(points)

            # Collect for diagnostics
            diagnostic_collector.add_prediction(pred_result, actual_top3)

        avg_points = np.mean(fold_game_points) if fold_game_points else 0
        cv_scores.append(avg_points)
        logger.info(
            f"Fold {fold_idx + 1}: {avg_points:.2f} avg game points "
            f"({len(fold_game_points)} sessions)"
        )

    # Train final model on all data
    final_model = ranker_class(objective=objective, **model_kwargs)
    final_model.fit(X_sorted, y_relevance, group_sizes)

    # Get diagnostic report
    diagnostics = diagnostic_collector.get_report()

    results = {
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores,
        "n_features": len(X.columns),
        "n_samples": len(X),
        "n_sessions": len(group_sizes),
        "objective": objective,
        "session_type": ranker_class.session_type,
        "top_5_features": list(final_model.get_feature_importance().keys())[:5],
        "diagnostics": diagnostics,
    }

    logger.info(
        f"Training complete: {results['cv_mean']:.2f} +/- {results['cv_std']:.2f} avg game points"
    )

    # Print detailed diagnostics if requested
    if verbose_diagnostics:
        print(diagnostic_collector.format_report())

    return final_model, results
