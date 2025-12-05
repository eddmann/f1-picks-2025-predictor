"""
Hyperparameter tuning for LightGBM ranking models.

Uses Optuna for Bayesian optimization with temporal cross-validation.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.evaluation.scoring import calculate_game_points
from src.features.qualifying_pipeline import QualifyingFeaturePipeline
from src.features.race_pipeline import RaceFeaturePipeline
from src.models.base_ranker import (
    create_temporal_cv_splits_with_groups,
    prepare_ranking_data,
)
from src.models.qualifying import QualifyingLGBMRanker
from src.models.race import RaceLGBMRanker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_objective(
    X_sorted,
    y_relevance,
    meta_sorted,
    splits,
    ranker_class,
):
    """Create Optuna objective function for hyperparameter tuning."""

    def objective(trial):
        # Sample hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        cv_scores = []
        for train_idx, val_idx, train_groups, _val_groups in splits:
            X_train = X_sorted.iloc[train_idx]
            y_train = y_relevance.iloc[train_idx]
            X_val = X_sorted.iloc[val_idx]
            meta_val = meta_sorted.iloc[val_idx]

            # Train fold model
            model = ranker_class(objective="lambdarank")
            # Set params manually since LGBMRanker doesn't accept reg_alpha/lambda in init
            model.n_estimators = params["n_estimators"]
            model.num_leaves = params["num_leaves"]
            model.learning_rate = params["learning_rate"]
            model.min_child_samples = params["min_child_samples"]
            model.subsample = params["subsample"]
            model.colsample_bytree = params["colsample_bytree"]

            # Need to modify fit to pass reg params
            import lightgbm as lgb

            model.feature_names = X_train.columns.tolist()
            model.model = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=params["n_estimators"],
                num_leaves=params["num_leaves"],
                learning_rate=params["learning_rate"],
                min_child_samples=params["min_child_samples"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                random_state=42,
                importance_type="gain",
                n_jobs=-1,
                verbose=-1,
            )
            model.model.fit(X_train, y_train, group=train_groups)

            # Evaluate on validation set
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

                pred_result = model.predict_top3(
                    session_X,
                    session_meta["driver_code"].tolist(),
                )
                pred_top3 = pred_result["top3"]

                points = calculate_game_points(pred_top3, actual_top3)
                fold_game_points.append(points)

            avg_points = np.mean(fold_game_points) if fold_game_points else 0
            cv_scores.append(avg_points)

        return np.mean(cv_scores)

    return objective


def tune_model(
    prediction_type: str,
    n_trials: int = 50,
    min_year: int = 2020,
) -> dict:
    """
    Tune hyperparameters for a prediction type.

    Args:
        prediction_type: "qualifying" or "race"
        n_trials: Number of Optuna trials
        min_year: Minimum year for training data

    Returns:
        Dict with best params and score
    """
    logger.info(f"Tuning {prediction_type} model with {n_trials} trials...")

    # Load data
    if prediction_type == "qualifying":
        pipeline = QualifyingFeaturePipeline()
        ranker_class = QualifyingLGBMRanker
    elif prediction_type == "race":
        pipeline = RaceFeaturePipeline()
        ranker_class = RaceLGBMRanker
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    X, y, meta = pipeline.build_features(min_year=min_year, for_ranking=True)
    logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")

    # Prepare ranking data
    X_sorted, y_relevance, group_sizes, meta_sorted = prepare_ranking_data(X, y, meta)

    # Create CV splits
    splits = create_temporal_cv_splits_with_groups(meta_sorted, n_splits=5)
    logger.info(f"Created {len(splits)} CV splits")

    # Create and run study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = create_objective(X_sorted, y_relevance, meta_sorted, splits, ranker_class)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"\nBest score: {best_score:.3f} avg game points")
    logger.info(f"Best params: {json.dumps(best_params, indent=2)}")

    return {
        "prediction_type": prediction_type,
        "best_score": best_score,
        "best_params": best_params,
        "n_trials": n_trials,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune LightGBM hyperparameters")
    parser.add_argument(
        "--type",
        choices=["qualifying", "race", "all"],
        default="all",
        help="Prediction type to tune",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2020,
        help="Minimum year for training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tuning_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    results = {}

    if args.type in ["qualifying", "all"]:
        results["qualifying"] = tune_model(
            "qualifying", n_trials=args.trials, min_year=args.min_year
        )

    if args.type in ["race", "all"]:
        results["race"] = tune_model("race", n_trials=args.trials, min_year=args.min_year)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    for pred_type, result in results.items():
        print(f"\n{pred_type.upper()}:")
        print(f"  Best score: {result['best_score']:.3f} avg game points")
        print("  Best params:")
        for k, v in result["best_params"].items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
