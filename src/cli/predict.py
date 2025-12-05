"""
CLI for generating F1 predictions.

Uses the LGBMRanker with 170 features including current weekend FP data.
"""

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import config
from src.data.loaders import F1DataLoader
from src.evaluation.explanation import ModelExplainer
from src.features.imputation import FeatureImputer
from src.features.qualifying_pipeline import QualifyingFeaturePipeline
from src.features.race_pipeline import RaceFeaturePipeline
from src.features.sprint_quali_pipeline import SprintQualiFeaturePipeline
from src.features.sprint_race_pipeline import SprintRaceFeaturePipeline
from src.models.qualifying import load_model

logger = logging.getLogger(__name__)


class PredictionValidationError(Exception):
    """Raised when prediction input validation fails."""

    pass


# Valid race ID patterns: YYYY-NN (round number) or YYYY-circuit-name
RACE_ID_PATTERN = r"^(20[0-9]{2})-(0?[1-9]|1[0-9]|2[0-4]|[a-z][a-z0-9-]*)$"


def validate_race_id(race_id: str) -> None:
    """
    Validate race ID format to prevent path traversal attacks.

    Args:
        race_id: Race ID like "2025-abu-dhabi" or "2025-24"

    Raises:
        PredictionValidationError: If race_id format is invalid or contains path traversal

    Security:
        Prevents path traversal attacks by validating format strictly.
        Rejects any race_id containing: .., /, \\, or null bytes.
    """
    import re

    # Check for path traversal characters
    dangerous_patterns = ["..", "/", "\\", "\x00", "~"]
    for pattern in dangerous_patterns:
        if pattern in race_id:
            raise PredictionValidationError(
                f"Invalid race_id: contains forbidden character '{pattern}'. "
                f"Use format: YYYY-round or YYYY-circuit-name (e.g., 2025-24 or 2025-qatar)"
            )

    # Validate format with regex
    if not re.match(RACE_ID_PATTERN, race_id.lower()):
        raise PredictionValidationError(
            f"Invalid race_id format: '{race_id}'. "
            f"Use format: YYYY-round or YYYY-circuit-name (e.g., 2025-24 or 2025-qatar)"
        )

    # Additional length check
    if len(race_id) > 50:
        raise PredictionValidationError(
            f"Invalid race_id: too long ({len(race_id)} chars, max 50)"
        )


def validate_model_features(model_features: list[str], input_features: list[str]) -> None:
    """
    Validate that input features match what the model expects.

    Args:
        model_features: Features the model was trained on
        input_features: Features in the input data

    Raises:
        PredictionValidationError: If features don't match
    """
    model_set = set(model_features)
    input_set = set(input_features)

    missing = model_set - input_set
    extra = input_set - model_set

    if missing:
        raise PredictionValidationError(
            f"Input data missing {len(missing)} features required by model: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )

    if extra:
        logger.warning(
            f"Input data has {len(extra)} extra features not used by model: "
            f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
        )


def validate_prediction_input(
    X: pd.DataFrame,
    driver_codes: list[str],
    min_drivers: int = 3,
) -> None:
    """
    Validate prediction input data.

    Args:
        X: Feature matrix
        driver_codes: List of driver codes
        min_drivers: Minimum number of drivers required

    Raises:
        PredictionValidationError: If validation fails
    """
    if X.empty:
        raise PredictionValidationError("Feature matrix is empty")

    if len(driver_codes) < min_drivers:
        raise PredictionValidationError(
            f"Need at least {min_drivers} drivers, got {len(driver_codes)}"
        )

    if len(X) != len(driver_codes):
        raise PredictionValidationError(
            f"Feature matrix rows ({len(X)}) don't match driver count ({len(driver_codes)})"
        )

    # Check for excessive NaN values
    nan_pct = X.isna().sum().sum() / (X.shape[0] * X.shape[1]) * 100
    if nan_pct > 50:
        raise PredictionValidationError(
            f"Feature matrix has {nan_pct:.1f}% NaN values - data may be missing"
        )

    # Check for infinite values
    inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        raise PredictionValidationError(f"Feature matrix contains {inf_count} infinite values")


def generate_prediction_id() -> str:
    """Generate unique prediction ID."""
    return f"pred-{uuid.uuid4().hex[:12]}"


def save_prediction(prediction: dict, output_dir: Path) -> Path:
    """Save prediction to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{prediction['prediction_id']}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(prediction, f, indent=2, default=str)

    logger.info(f"Prediction saved to {filepath}")
    return filepath


def get_upcoming_race_info(loader: F1DataLoader, race_id: str) -> dict:
    """
    Parse race ID and get race info.

    Args:
        loader: F1DataLoader instance
        race_id: Race ID like "2025-abu-dhabi" or "2025-24"

    Returns:
        Dict with year, round, circuit info

    Raises:
        PredictionValidationError: If race_id format is invalid
        ValueError: If race cannot be found
    """
    # Validate race_id format before processing (security: prevents path traversal)
    validate_race_id(race_id)

    parts = race_id.split("-", 1)
    year = int(parts[0])

    # Load events to find the race
    events = loader.load_events()
    year_events = events[events["year"] == year]

    if len(parts) > 1:
        round_or_name = parts[1]

        # Try to parse as round number
        try:
            round_num = int(round_or_name)
            event = year_events[year_events["round"] == round_num]
        except ValueError:
            # Try to match by circuit name
            event = year_events[
                year_events["circuit"]
                .str.lower()
                .str.contains(round_or_name.lower().replace("-", " "))
            ]

    if event.empty:
        # Get the latest/next event
        event = year_events.sort_values("round").tail(1)

    if event.empty:
        raise ValueError(f"Could not find race for {race_id}")

    event = event.iloc[0]
    return {
        "year": int(event["year"]),
        "round": int(event["round"]),
        "circuit": event["circuit"],
        "event_name": event.get("event_name", event["circuit"]),
    }


def get_pipeline_for_type(
    prediction_type: str, data_dir: Path
) -> (
    QualifyingFeaturePipeline
    | RaceFeaturePipeline
    | SprintQualiFeaturePipeline
    | SprintRaceFeaturePipeline
):
    """
    Get the appropriate feature pipeline for the prediction type.

    Args:
        prediction_type: One of "qualifying", "race", "sprint_quali", "sprint_race"
        data_dir: Path to data directory

    Returns:
        Initialized feature pipeline instance

    Raises:
        ValueError: If prediction_type is unknown
    """
    pipelines: dict[str, type] = {
        "qualifying": QualifyingFeaturePipeline,
        "race": RaceFeaturePipeline,
        "sprint_quali": SprintQualiFeaturePipeline,
        "sprint_race": SprintRaceFeaturePipeline,
    }
    pipeline_class = pipelines.get(prediction_type)
    if pipeline_class is None:
        raise ValueError(f"Unknown prediction type: {prediction_type}")
    return pipeline_class(data_dir)


def build_prediction_features(
    prediction_type: str,
    loader: F1DataLoader,
    race_info: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build features for prediction including current weekend FP data.

    Args:
        prediction_type: Type of prediction (qualifying, race, sprint_quali, sprint_race)
        loader: F1DataLoader instance
        race_info: Dict with year, round, circuit info

    Returns:
        Tuple of (features DataFrame, metadata DataFrame, driver codes list)
    """
    # Get appropriate pipeline for prediction type
    pipeline = get_pipeline_for_type(prediction_type, config.data.data_dir)

    # Determine min_year based on prediction type
    min_year = 2020
    if prediction_type in ["sprint_quali", "sprint_race"]:
        min_year = 2021  # Sprint format started in 2021

    # Build all historical features
    X, y, meta = pipeline.build_features(min_year=min_year, for_ranking=True)

    # Apply imputation
    imputer = FeatureImputer()
    X = imputer.fit_transform(X)

    # Get the target session for prediction
    year = race_info["year"]
    round_num = race_info["round"]

    # Check if we have FP data for this weekend
    practice = loader.load_practice_sessions(min_year=year)
    weekend_practice = practice[(practice["year"] == year) & (practice["round"] == round_num)]

    if weekend_practice.empty:
        logger.warning(f"No practice data found for {year} round {round_num}")
        logger.warning("Predictions will be based on historical data only.")
        logger.warning(
            "Run: uv run python -m src.data.fastf1_sync --season {year} --round {round_num}"
        )
    else:
        fp_sessions = weekend_practice["session_type"].unique()
        logger.info(f"Found practice data: {', '.join(fp_sessions)}")

    # Find drivers who have data for this session
    # Use the most recent session as template for drivers
    quali = loader.load_qualifying_results(min_year=year)

    # Get drivers from latest qualifying
    latest_quali = quali.sort_values(["year", "round"]).groupby("driver_code").last().reset_index()
    driver_codes = latest_quali["driver_code"].tolist()

    # Filter to target race weekend if we have the data
    target_session_key = f"{year}_{round_num:02d}_Q"

    if target_session_key in meta["session_key"].values:
        # We have qualifying data for this race - use those features
        target_mask = meta["session_key"] == target_session_key
        X_pred = X.loc[target_mask].copy()
        meta_pred = meta.loc[target_mask].copy()
        driver_codes = meta_pred["driver_code"].tolist()
    else:
        # Need to build features for upcoming race
        # Use most recent data for each driver
        X_pred = pd.DataFrame()
        meta_pred = pd.DataFrame()

        for driver in driver_codes:
            driver_mask = meta["driver_code"] == driver
            if driver_mask.any():
                # Get most recent data for this driver
                driver_idx = meta.loc[driver_mask].sort_values(["year", "round"]).index[-1]
                driver_features = X.loc[[driver_idx]].copy()
                driver_meta = meta.loc[[driver_idx]].copy()

                # Update with current weekend FP data if available
                driver_practice = weekend_practice[weekend_practice["driver_code"] == driver]
                if not driver_practice.empty:
                    # Calculate FP rankings
                    fp3_times = weekend_practice[weekend_practice["session_type"] == "Practice 3"]
                    if not fp3_times.empty:
                        fp3_best = fp3_times.groupby("driver_code")["lap_time_ms"].min()
                        if driver in fp3_best.index:
                            driver_fp3_time = fp3_best[driver]
                            driver_features["current_fp3_best_ms"] = driver_fp3_time
                            driver_features["current_fp3_rank"] = (
                                fp3_best <= driver_fp3_time
                            ).sum()
                            driver_features["current_fp3_gap_ms"] = driver_fp3_time - fp3_best.min()
                            driver_features["current_fp3_gap_pct"] = (
                                (driver_fp3_time - fp3_best.min()) / fp3_best.min() * 100
                            )

                    # Overall practice ranking
                    practice_best = weekend_practice.groupby("driver_code")["lap_time_ms"].min()
                    if driver in practice_best.index:
                        driver_practice_time = practice_best[driver]
                        driver_features["current_practice_best_ms"] = driver_practice_time
                        driver_features["current_practice_rank"] = (
                            practice_best <= driver_practice_time
                        ).sum()
                        driver_features["current_practice_gap_ms"] = (
                            driver_practice_time - practice_best.min()
                        )
                        driver_features["current_practice_gap_pct"] = (
                            (driver_practice_time - practice_best.min()) / practice_best.min() * 100
                        )

                X_pred = pd.concat([X_pred, driver_features], ignore_index=True)
                meta_pred = pd.concat([meta_pred, driver_meta], ignore_index=True)

        meta_pred["driver_code"] = driver_codes[: len(meta_pred)]

    return X_pred, meta_pred, driver_codes[: len(X_pred)]


def generate_prediction(args, prediction_type: str):
    """Generate predictions using the appropriate LGBMRanker model."""
    logger.info(f"Generating {prediction_type} predictions for {args.race_id}...")

    # Map prediction type to model name
    model_names = {
        "qualifying": "qualifying_ranker",
        "race": "race_ranker",
        "sprint_quali": "sprint_quali_ranker",
        "sprint_race": "sprint_race_ranker",
    }
    model_name = args.model or model_names.get(prediction_type, "qualifying_ranker")
    model_path = config.models_dir / f"{model_name}.pkl"

    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Available models:")
        for p in config.models_dir.glob("*.pkl"):
            print(f"  - {p.stem}")
        print("\nRun 'uv run python -m src.cli.retrain --type all' to train models.")
        return None

    model = load_model(model_path)

    # Load metadata
    metadata_path = model_path.with_suffix(".json")
    model_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            model_metadata = json.load(f)

    # Initialize data loader
    loader = F1DataLoader(config.data.data_dir)

    # Get race info
    try:
        race_info = get_upcoming_race_info(loader, args.race_id)
    except ValueError as e:
        print(f"\nError: {e}")
        return None

    # Format prediction type for display
    prediction_display = prediction_type.replace("_", " ").upper()
    print(f"\n{'=' * 60}")
    print(f"F1 {prediction_display} PREDICTION")
    print(f"{'=' * 60}")
    print(f"Race: {race_info['event_name']} ({race_info['circuit']})")
    print(f"Year: {race_info['year']}, Round: {race_info['round']}")
    print(f"Model: {model_name}")

    # Build features using the appropriate pipeline
    print("\nBuilding features...")
    X_pred, meta_pred, driver_codes = build_prediction_features(prediction_type, loader, race_info)

    if X_pred.empty:
        print("\nError: No driver data available for prediction")
        return None

    print(f"Drivers: {len(driver_codes)}")

    # Get feature columns (same as training)
    feature_cols = model_metadata.get("features", X_pred.columns.tolist())

    # Filter to available features
    missing_features = [f for f in feature_cols if f not in X_pred.columns]

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features, filling with 0")
        for f in missing_features:
            X_pred[f] = 0

    X_input = X_pred[feature_cols].copy()

    # Generate predictions
    print("\nGenerating predictions...")

    # Check if model is a ranker or classifier
    if hasattr(model, "predict_top3"):
        # LGBMRanker - get raw scores from model
        raw_scores = model.predict(X_input)
        driver_scores = list(zip(driver_codes, raw_scores, strict=False))
        driver_scores.sort(key=lambda x: x[1], reverse=True)
    else:
        # Classification model
        probas = model.predict_proba(X_input)
        if probas.shape[1] > 1:
            top3_proba = probas[:, 1]
        else:
            top3_proba = probas[:, 0]

        driver_scores = list(zip(driver_codes, top3_proba, strict=False))
        driver_scores.sort(key=lambda x: x[1], reverse=True)

    # Get driver info
    quali = loader.load_qualifying_results(min_year=2024)
    driver_info = (
        quali.drop_duplicates("driver_code").set_index("driver_code")[["team"]].to_dict("index")
    )

    # Get scores for normalization
    all_scores = [s for _, s in driver_scores]
    max_score = max(all_scores) if all_scores else 1
    min_score = min(all_scores) if all_scores else 0
    score_range = max_score - min_score if max_score != min_score else 1

    # Build prediction object
    prediction = {
        "prediction_id": generate_prediction_id(),
        "race_id": args.race_id,
        "race_info": race_info,
        "prediction_type": "qualifying",
        "prediction_timestamp": datetime.now().isoformat(),
        "model_version": model_name,
        "model_trained_at": model_metadata.get("trained_at", "unknown"),
        "model_cv_score": model_metadata.get("cv_mean", "unknown"),
        "n_features": len(feature_cols),
        "predictions": {},
        "all_drivers": [],
    }

    # Add top 3
    for i, (driver, score) in enumerate(driver_scores[:3]):
        team = driver_info.get(driver, {}).get("team", "Unknown")
        confidence = ((score - min_score) / score_range * 100) if score_range else 50

        prediction["predictions"][f"p{i + 1}"] = {
            "driver_code": driver,
            "team": team,
            "confidence": round(confidence, 1),
            "raw_score": round(float(score), 4),
        }

    # Add all drivers
    for driver, score in driver_scores:
        team = driver_info.get(driver, {}).get("team", "Unknown")
        prediction["all_drivers"].append(
            {
                "driver_code": driver,
                "team": team,
                "raw_score": round(float(score), 4),
            }
        )

    # Save prediction
    output_path = save_prediction(prediction, config.data.predictions_dir)

    # Display prediction
    print(f"\n{'=' * 60}")
    print("PREDICTED TOP 3:")
    print("-" * 40)
    for i in range(3):
        p = prediction["predictions"][f"p{i + 1}"]
        print(f"  P{i + 1}: {p['driver_code']:3} ({p['team']}) - score: {p['raw_score']:.3f}")

    print(f"\n{'=' * 60}")
    print("FULL GRID PREDICTION:")
    print("-" * 40)
    for i, d in enumerate(prediction["all_drivers"][:10], 1):
        marker = " *" if i <= 3 else ""
        print(f"  P{i:2}: {d['driver_code']:3} ({d['team']}) - {d['raw_score']:.3f}{marker}")

    if len(prediction["all_drivers"]) > 10:
        print(f"  ... and {len(prediction['all_drivers']) - 10} more drivers")

    print(f"\n{'=' * 60}")
    print(f"Prediction saved: {output_path}")
    print(f"Model CV score: {model_metadata.get('cv_mean', 'N/A'):.2f} game points")
    print(f"{'=' * 60}\n")

    # Generate SHAP explanation if requested
    if getattr(args, "explain", False):
        print("Generating SHAP explanation...")
        try:
            explainer = ModelExplainer(model, feature_names=feature_cols)
            explainer.fit(X_input, sample_size=min(100, len(X_input)))

            # Get predicted ranking
            predicted_ranking = [d["driver_code"] for d in prediction["all_drivers"]]

            # Generate explanation
            explanation = explainer.explain_ranking(
                X_input, driver_codes, predicted_ranking, top_k=5
            )

            # Format and print
            print(explainer.format_explanation(explanation))

            # Save explanation alongside prediction
            explanation_path = output_path.with_suffix(".explanation.json")
            explainer.save_explanation(explanation, explanation_path, format="json")
            print(f"\nExplanation saved: {explanation_path}")

        except Exception as e:
            logger.warning(f"Could not generate SHAP explanation: {e}")
            print(f"\nWarning: SHAP explanation failed: {e}")

    return prediction


def predict_qualifying(args):
    """Generate qualifying predictions using qualifying-specific model."""
    return generate_prediction(args, "qualifying")


def predict_race(args):
    """Generate race predictions using race-specific model."""
    return generate_prediction(args, "race")


def predict_sprint_quali(args):
    """Generate sprint qualifying predictions using sprint quali model."""
    return generate_prediction(args, "sprint_quali")


def predict_sprint_race(args):
    """Generate sprint race predictions using sprint race model."""
    return generate_prediction(args, "sprint_race")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate F1 predictions using LGBMRanker models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict qualifying for Abu Dhabi 2025
  uv run python -m src.cli.predict --type qualifying --race-id 2025-abu-dhabi

  # Predict race results (uses qualifying grid as key feature)
  uv run python -m src.cli.predict --type race --race-id 2025-24

  # Sprint predictions (sprint weekends only)
  uv run python -m src.cli.predict --type sprint_quali --race-id 2025-qatar
  uv run python -m src.cli.predict --type sprint_race --race-id 2025-qatar

  # First sync the FP data for best predictions
  uv run python -m src.data.fastf1_sync --season 2025 --round 24
        """,
    )
    parser.add_argument(
        "--type",
        choices=["qualifying", "race", "sprint_quali", "sprint_race"],
        required=True,
        help="Type of prediction (qualifying, race, sprint_quali, sprint_race)",
    )
    parser.add_argument(
        "--race-id",
        required=True,
        help="Race ID (e.g., 2025-abu-dhabi or 2025-24)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: auto-select based on type)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show SHAP-based explanation of why drivers were ranked",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Generate prediction using the appropriate pipeline and model
    generate_prediction(args, args.type)


if __name__ == "__main__":
    main()
