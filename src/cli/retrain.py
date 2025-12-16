"""
CLI for retraining F1 prediction models.

Provides command-line interface for model retraining with FastF1 data.
Supports multiple prediction types:
- qualifying: Predict qualifying top-3
- sprint_quali: Predict sprint qualifying top-3
- sprint_race: Predict sprint race top-3
- race: Predict race top-3
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import config
from src.data.loaders import F1DataLoader
from src.features.imputation import FeatureImputer
from src.features.qualifying_pipeline import QualifyingFeaturePipeline
from src.features.race_pipeline import RaceFeaturePipeline
from src.features.selection import get_nonzero_features, select_features_by_importance
from src.features.sprint_quali_pipeline import SprintQualiFeaturePipeline
from src.features.sprint_race_pipeline import SprintRaceFeaturePipeline
from src.models.base_ranker import train_ranker_model
from src.models.qualifying import (
    QualifyingLGBMRanker,
    save_model,
)
from src.models.race import RaceLGBMRanker
from src.models.sprint_qualifying import SprintQualiLGBMRanker
from src.models.sprint_race import SprintRaceLGBMRanker

logger = logging.getLogger(__name__)


def parse_race_id(race_id: str, loader: F1DataLoader) -> tuple[int, int]:
    """
    Parse a race ID string to (year, round) tuple.

    Args:
        race_id: Race ID like "2025-24" or "2025-qatar"
        loader: F1DataLoader instance for looking up circuit names

    Returns:
        Tuple of (year, round)

    Raises:
        ValueError: If race_id format is invalid or race cannot be found
    """
    parts = race_id.split("-", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid race_id format: '{race_id}'. Use format: YYYY-round or YYYY-circuit")

    try:
        year = int(parts[0])
    except ValueError as e:
        raise ValueError(f"Invalid year in race_id: '{parts[0]}'") from e

    round_or_name = parts[1]

    # Try to parse as round number
    try:
        round_num = int(round_or_name)
        return (year, round_num)
    except ValueError:
        pass

    # Try to find by circuit name
    events = loader.load_events()
    year_events = events[events["year"] == year]
    match = year_events[
        year_events["circuit"].str.lower().str.contains(round_or_name.lower().replace("-", " "))
    ]

    if match.empty:
        raise ValueError(f"Could not find race: {race_id}")

    round_num = int(match.iloc[0]["round"])
    return (year, round_num)


def prepare_features(
    min_year: int | None = None,
    prediction_type: str = "qualifying",
    for_ranking: bool = True,
    up_to_race: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare features and labels for training using the appropriate pipeline.

    Args:
        min_year: Minimum year to include in training (default: from config)
        prediction_type: Type of prediction (qualifying, sprint_quali, sprint_race, race)
        for_ranking: If True, returns position as target; if False, returns is_top3
        up_to_race: Optional (year, round) tuple to filter data up to but not including

    Returns:
        Tuple of (features DataFrame, labels Series, metadata DataFrame)
    """
    # Use config default if not specified
    if min_year is None:
        min_year = config.model.min_year

    data_dir = config.data.data_dir

    # Select appropriate pipeline
    if prediction_type == "qualifying":
        pipeline = QualifyingFeaturePipeline(data_dir)
    elif prediction_type == "sprint_quali":
        pipeline = SprintQualiFeaturePipeline(data_dir)
        min_year = max(min_year, 2021)  # Sprint format started in 2021
    elif prediction_type == "sprint_race":
        pipeline = SprintRaceFeaturePipeline(data_dir)
        min_year = max(min_year, 2021)  # Sprint format started in 2021
    elif prediction_type == "race":
        pipeline = RaceFeaturePipeline(data_dir)
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    X, y, meta = pipeline.build_features(
        min_year=min_year, up_to_race=up_to_race, for_ranking=for_ranking
    )

    logger.info(
        f"Prepared {len(X)} samples with {len(X.columns)} features for {prediction_type} prediction"
    )
    return X, y, meta


def get_ranker_class(prediction_type: str):
    """Get the appropriate ranker class for the prediction type."""
    ranker_classes = {
        "qualifying": QualifyingLGBMRanker,
        "sprint_quali": SprintQualiLGBMRanker,
        "sprint_race": SprintRaceLGBMRanker,
        "race": RaceLGBMRanker,
    }
    return ranker_classes.get(prediction_type)


def export_lgbm_native(model, feature_names: list[str], output_path: Path) -> None:
    """
    Export LightGBM model to native text format.

    Note: ONNX doesn't support LambdaRank objective, so we use LightGBM's
    native format which can be loaded by LightGBM libraries in other languages
    (Python, C++, Java, etc.)

    Args:
        model: Trained QualifyingLGBMRanker wrapper
        feature_names: List of feature names
        output_path: Path to save model (use .txt extension)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save using LightGBM's native format
    model.model.booster_.save_model(str(output_path))

    # Also save feature names alongside
    feature_path = output_path.with_suffix(".features.json")
    import json

    with open(feature_path, "w") as f:
        json.dump({"features": feature_names}, f, indent=2)

    logger.info(f"LightGBM model saved to {output_path}")
    logger.info(f"Feature names saved to {feature_path}")


def train_ranker_for_type(
    prediction_type: str,
    args,
    min_year: int,
    up_to_race: tuple[int, int] | None = None,
) -> bool:
    """
    Train a ranker model for a specific prediction type.

    Args:
        prediction_type: Type of prediction (qualifying, sprint_quali, sprint_race, race)
        args: CLI arguments
        min_year: Minimum year for training data
        up_to_race: Optional (year, round) tuple to filter data up to but not including

    Returns:
        True if training succeeded, False otherwise
    """
    print(f"\nTraining {prediction_type.replace('_', ' ').title()} LGBMRanker...")

    # Get appropriate ranker class
    ranker_class = get_ranker_class(prediction_type)
    if ranker_class is None:
        print(f"  ERROR: Unknown prediction type: {prediction_type}")
        return False

    # Prepare features
    X_rank, y_rank, meta_rank = prepare_features(
        min_year=min_year,
        prediction_type=prediction_type,
        for_ranking=True,
        up_to_race=up_to_race,
    )

    if X_rank.empty:
        print(f"  ERROR: No training data available for {prediction_type}")
        return False

    # Apply smart imputation
    imputer = FeatureImputer()
    X_rank = imputer.fit_transform(X_rank)

    feature_names_rank = X_rank.columns.tolist()
    print(f"  Data: {len(X_rank)} samples, {len(feature_names_rank)} features")

    # Optional feature selection
    if args.select_features:
        print(f"  Selecting top {args.select_features} features...")
        X_rank, selected_features = select_features_by_importance(
            X_rank, (y_rank <= 3).astype(int), n_features=args.select_features
        )
        feature_names_rank = selected_features

    # Two-pass training with feature pruning
    if args.prune_features:
        print("  Pass 1: Training to identify zero-importance features...")
        initial_model, _ = train_ranker_model(
            X_rank,
            y_rank,
            meta_rank,
            ranker_class=ranker_class,
            objective=args.objective or "lambdarank",
            cv_splits=5,
        )

        # Identify non-zero features
        kept_features = get_nonzero_features(
            initial_model.model, feature_names_rank, min_importance=0.0
        )
        n_removed = len(feature_names_rank) - len(kept_features)
        print(f"  Pruned {n_removed} zero-importance features")
        print(f"  Kept {len(kept_features)} features")

        # Retrain with pruned features
        X_rank = X_rank[kept_features]
        feature_names_rank = kept_features
        print("  Pass 2: Retraining with pruned feature set...")

    # Train with cross-validation
    model, results = train_ranker_model(
        X_rank,
        y_rank,
        meta_rank,
        ranker_class=ranker_class,
        objective=args.objective or "lambdarank",
        cv_splits=5,
    )

    # Model naming convention
    model_name = f"{prediction_type}_ranker"
    model_path = config.models_dir / f"{model_name}.pkl"
    metadata = {
        "model_type": "LGBMRanker",
        "objective": results["objective"],
        "prediction_type": prediction_type,
        "session_type": ranker_class.session_type,
        "trained_at": datetime.now().isoformat(),
        "n_samples": results["n_samples"],
        "n_features": results["n_features"],
        "n_sessions": results["n_sessions"],
        "cv_mean": results["cv_mean"],
        "cv_std": results["cv_std"],
        "features": feature_names_rank,
        "min_year": min_year,
        "up_to_race": f"{up_to_race[0]}-{up_to_race[1]}" if up_to_race else None,
        "data_source": "fastf1",
    }
    save_model(model, model_path, metadata)

    # Export to native LightGBM format
    lgbm_path = config.models_dir / f"{model_name}.lgbm.txt"
    try:
        export_lgbm_native(model, feature_names_rank, lgbm_path)
        print(f"  LightGBM: {lgbm_path}")
    except Exception as e:
        logger.warning(f"LightGBM export failed: {e}")

    print(f"  Pickle: {model_path}")
    print(f"  CV Score: {results['cv_mean']:.2f} +/- {results['cv_std']:.2f} game points")

    # Print feature importance
    importance = model.get_feature_importance()
    print("\n  Top 10 features:")
    for i, (name, imp) in enumerate(list(importance.items())[:10]):
        print(f"    {i + 1}. {name}: {imp:.4f}")

    return True


def retrain_model(args):
    """Retrain a specific model or all models."""
    # Determine which prediction types to train
    if args.type == "all":
        prediction_types = ["qualifying", "sprint_quali", "sprint_race", "race"]
    else:
        prediction_types = [args.type]

    logger.info(f"Retraining models for types: {prediction_types}")

    print("\nModel Retraining")
    print("=" * 60)

    min_year = args.min_year or 2020
    models_trained = []

    # Parse race ID if provided
    up_to_race = None
    if args.race:
        loader = F1DataLoader(config.data.data_dir)
        up_to_race = parse_race_id(args.race, loader)
        print(f"Training with data up to (not including) {up_to_race[0]} R{up_to_race[1]}")

    # Train rankers for each prediction type
    for prediction_type in prediction_types:
        print(f"\n{'=' * 60}")
        print(f"Prediction Type: {prediction_type.upper()}")
        print("=" * 60)

        effective_min_year = min_year
        if prediction_type in ["sprint_quali", "sprint_race"]:
            effective_min_year = max(min_year, 2021)
            print(f"(Using min_year={effective_min_year} for sprint data)")

        if train_ranker_for_type(prediction_type, args, effective_min_year, up_to_race):
            models_trained.append(f"{prediction_type}_ranker")

    print("\n" + "=" * 60)
    print(f"Training complete. Models saved: {', '.join(models_trained)}")
    print(f"Location: {config.models_dir.absolute()}")


def compare_models(args):
    """Compare two model versions."""
    logger.info(f"Comparing models: {args.model_a} vs {args.model_b}...")

    print("\nModel Comparison")
    print("=" * 60)

    # Load metadata for both models
    for model_name in [args.model_a, args.model_b]:
        metadata_path = config.models_dir / f"{model_name}.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            print(f"\n{model_name}:")
            print(f"  Type: {meta.get('model_type', 'unknown')}")
            print(f"  Trained: {meta.get('trained_at', 'unknown')}")
            print(f"  Samples: {meta.get('n_samples', 'unknown')}")
            print(f"  Features: {meta.get('n_features', 'unknown')}")
            print(f"  Data source: {meta.get('data_source', 'unknown')}")
            print(f"  Min year: {meta.get('min_year', 'unknown')}")
        else:
            print(f"\n{model_name}: metadata not found")

    print("\n" + "=" * 60)


def show_data_status():
    """Show current data status."""
    print("\nData Status")
    print("=" * 60)

    loader = F1DataLoader(config.data.data_dir)

    # Check available years
    years = loader.get_available_years()
    sessions = loader.get_available_sessions()

    print(f"\nData directory: {config.data.data_dir.absolute()}")
    print(f"Available years: {years if years else 'None'}")
    print(f"Total sessions: {len(sessions)}")

    if sessions:
        # Group by year
        by_year = {}
        for s in sessions:
            parts = s.split("_")
            if parts:
                year = parts[0]
                by_year[year] = by_year.get(year, 0) + 1

        print("\nSessions per year:")
        for year, count in sorted(by_year.items()):
            print(f"  {year}: {count} sessions")

    print("\n" + "=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Retrain F1 prediction models")
    parser.add_argument(
        "--type",
        choices=["qualifying", "sprint_quali", "sprint_race", "race", "all"],
        default="qualifying",
        help="Prediction type to train (default: qualifying, use 'all' for all types)",
    )
    parser.add_argument("--min-year", type=int, default=2020, help="Minimum year for training data")
    parser.add_argument(
        "--race",
        type=str,
        default=None,
        help="Train with data up to (but not including) this race (e.g., 2025-24 or 2025-qatar)",
    )
    parser.add_argument(
        "--objective",
        choices=["lambdarank", "rank_xendcg"],
        default="lambdarank",
        help="Ranking objective for LGBMRanker (default: lambdarank)",
    )
    parser.add_argument(
        "--select-features",
        type=int,
        default=None,
        help="Select top N features by importance (reduces overfitting)",
    )
    parser.add_argument(
        "--prune-features",
        action="store_true",
        help="Two-pass training: first identify then remove zero-importance features",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune hyperparameters with Optuna (slower but may improve results)",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for tuning (default: 50)",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also train legacy RF/GB models (qualifying only)",
    )
    parser.add_argument("--compare-models", action="store_true", help="Compare two model versions")
    parser.add_argument("--model-a", help="First model for comparison")
    parser.add_argument("--model-b", help="Second model for comparison")
    parser.add_argument("--status", action="store_true", help="Show data status")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Execute command
    if args.status:
        show_data_status()
    elif args.compare_models:
        if not args.model_a or not args.model_b:
            parser.error("--model-a and --model-b required for comparison")
        compare_models(args)
    else:
        retrain_model(args)


if __name__ == "__main__":
    main()
