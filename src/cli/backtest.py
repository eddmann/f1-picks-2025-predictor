"""
CLI for season backtesting of F1 prediction models.

Performs proper time-series validation by training fresh models
for each race in a season using only data available before that race.

Usage:
    make backtest SEASON=2025
    make backtest SEASON=2025 TYPE=qualifying
    make backtest SEASON=2025 CACHE=1  # Cache models for faster re-runs
"""

import argparse
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.cli.retrain import get_ranker_class, prepare_features
from src.config import config
from src.data.loaders import F1DataLoader
from src.evaluation.ranking_metrics import evaluate_predictions
from src.evaluation.scoring import calculate_game_points
from src.features.imputation import FeatureImputer
from src.models.base_ranker import train_ranker_model
from src.models.baselines import (
    FP1BasedBaseline,
    PracticeBasedBaseline,
    RaceGridBaseline,
    SprintQualiBasedBaseline,
)

logger = logging.getLogger(__name__)

SessionType = Literal["qualifying", "sprint_quali", "sprint_race", "race"]


# =============================================================================
# Data Classes (self-contained, not exported)
# =============================================================================


@dataclass
class SessionPrediction:
    """Result for a single session prediction."""

    driver_codes: list[str]
    actual_codes: list[str]
    game_points: int
    ndcg_3: float
    map_3: float
    top3_inclusion: float
    exact_match_rate: float


@dataclass
class BaselinePrediction:
    """Baseline prediction for comparison."""

    name: str
    driver_codes: list[str]
    game_points: int


@dataclass
class RaceBacktestResult:
    """Complete backtest result for a single race."""

    year: int
    round: int
    circuit: str
    event_name: str
    is_sprint_weekend: bool
    session_results: dict[str, SessionPrediction] = field(default_factory=dict)
    baseline_results: dict[str, dict[str, BaselinePrediction]] = field(default_factory=dict)
    training_samples: dict[str, int] = field(default_factory=dict)
    training_features: dict[str, int] = field(default_factory=dict)


@dataclass
class SeasonBacktestResult:
    """Complete backtest result for a season."""

    season: int
    races: list[RaceBacktestResult]
    summary: dict
    generated_at: str


# =============================================================================
# Core Functions
# =============================================================================


def get_season_races(loader: F1DataLoader, season: int) -> list[dict]:
    """
    Get all races in a season with available result data.

    Returns list of dicts with: year, round, circuit, event_name, is_sprint_weekend
    """
    events = loader.load_events(min_year=season)
    season_events = events[events["year"] == season].sort_values("round")

    races = []
    for _, event in season_events.iterrows():
        round_num = int(event["round"])
        is_sprint = loader.is_sprint_weekend(season, round_num)

        # Check if we have actual results (Q and R sessions exist)
        q_exists = (loader.sessions_dir / f"{season}_{round_num:02d}_Q.parquet").exists()
        r_exists = (loader.sessions_dir / f"{season}_{round_num:02d}_R.parquet").exists()

        if q_exists and r_exists:
            races.append(
                {
                    "year": season,
                    "round": round_num,
                    "circuit": event.get("circuit", ""),
                    "event_name": event.get("event_name", event.get("circuit", "")),
                    "is_sprint_weekend": is_sprint,
                }
            )

    return races


def get_session_types_for_race(race_info: dict) -> list[str]:
    """
    Determine which session types to evaluate for a race.

    Standard weekend: [qualifying, race]
    Sprint weekend: [sprint_quali, sprint_race, qualifying, race]

    Returns in chronological order for the weekend.
    """
    if race_info["is_sprint_weekend"]:
        return ["sprint_quali", "sprint_race", "qualifying", "race"]
    return ["qualifying", "race"]


def get_actual_results(
    loader: F1DataLoader,
    year: int,
    round_num: int,
    session_type: str,
) -> list[str]:
    """
    Get actual top-3 results for a session.

    Returns list of 3 driver codes in finishing order.
    """
    if session_type == "qualifying":
        results = loader.load_qualifying_results(min_year=year)
    elif session_type == "race":
        results = loader.load_race_results(min_year=year)
    elif session_type == "sprint_race":
        results = loader.load_sprint_results(min_year=year)
    elif session_type == "sprint_quali":
        results = loader.load_sprint_qualifying_results(min_year=year)
    else:
        return []

    session_results = results[(results["year"] == year) & (results["round"] == round_num)].copy()

    if session_results.empty:
        return []

    session_results = session_results[session_results["position"].notna()]
    session_results = session_results.sort_values("position")

    return session_results.head(3)["driver_code"].tolist()


def train_model_for_race(
    prediction_type: str,
    up_to_race: tuple[int, int],
    cache_dir: Path | None = None,
) -> tuple:
    """
    Train a model using data up to (but not including) the specified race.

    Args:
        prediction_type: qualifying, sprint_quali, sprint_race, or race
        up_to_race: (year, round) tuple
        cache_dir: Optional directory to cache/load models

    Returns:
        Tuple of (model, feature_names, training_info_dict)
    """
    # Check cache first
    if cache_dir:
        cache_path = cache_dir / f"{prediction_type}_{up_to_race[0]}_{up_to_race[1]:02d}.pkl"
        if cache_path.exists():
            logger.debug(f"Loading cached model from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["model"], cached["features"], cached["training_info"]

    # Prepare features with temporal cutoff
    min_year = 2020 if prediction_type not in ["sprint_quali", "sprint_race"] else 2021

    X, y, meta = prepare_features(
        min_year=min_year,
        prediction_type=prediction_type,
        for_ranking=True,
        up_to_race=up_to_race,
    )

    if X.empty or len(X) < 50:  # Minimum samples check
        return None, [], {"n_samples": 0, "n_features": 0}

    # Apply imputation
    imputer = FeatureImputer()
    X = imputer.fit_transform(X)
    feature_names = X.columns.tolist()

    # Train model (matching regular training parameters exactly)
    ranker_class = get_ranker_class(prediction_type)
    model, results = train_ranker_model(
        X,
        y,
        meta,
        ranker_class=ranker_class,
        objective="lambdarank",
        cv_splits=5,  # Same as regular training
    )

    training_info = {
        "n_samples": len(X),
        "n_features": len(feature_names),
        "cv_mean": results["cv_mean"],
    }

    # Save to cache if enabled
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "features": feature_names,
                    "training_info": training_info,
                },
                f,
            )
        logger.debug(f"Cached model to {cache_path}")

    return model, feature_names, training_info


def predict_session(
    model,
    feature_names: list[str],
    loader: F1DataLoader,
    prediction_type: str,
    year: int,
    round_num: int,
) -> list[str]:
    """
    Generate prediction for a specific session using trained model.

    This function matches the approach in predict.py's build_prediction_features
    to ensure backtest predictions match manual train+predict workflow.

    Args:
        model: Trained ranker model
        feature_names: Feature columns expected by model
        loader: F1DataLoader instance
        prediction_type: Session type
        year: Target year
        round_num: Target round

    Returns:
        List of 3 predicted driver codes
    """
    # Build features excluding target race (same as predict.py)
    min_year = 2020 if prediction_type not in ["sprint_quali", "sprint_race"] else 2021

    X, y, meta = prepare_features(
        min_year=min_year,
        prediction_type=prediction_type,
        for_ranking=True,
        up_to_race=(year, round_num),  # Exclude target race (same as predict.py)
    )

    if X.empty:
        return []

    # Apply imputation on all historical data first (same as predict.py)
    imputer = FeatureImputer()
    X = imputer.fit_transform(X)

    # Get drivers who have data for this session
    quali = loader.load_qualifying_results(min_year=year)
    latest_quali = quali.sort_values(["year", "round"]).groupby("driver_code").last().reset_index()
    driver_codes = latest_quali["driver_code"].tolist()

    # Build prediction features by getting most recent data per driver (same as predict.py)
    X_pred = pd.DataFrame()
    meta_pred = pd.DataFrame()

    for driver in driver_codes:
        driver_mask = meta["driver_code"] == driver
        if driver_mask.any():
            # Get most recent data for this driver
            driver_idx = meta.loc[driver_mask].sort_values(["year", "round"]).index[-1]
            driver_features = X.loc[[driver_idx]].copy()
            driver_meta = meta.loc[[driver_idx]].copy()

            # Update with current weekend practice data if available (same as predict.py)
            practice = loader.load_practice_sessions(min_year=year)
            weekend_practice = practice[(practice["year"] == year) & (practice["round"] == round_num)]

            driver_practice = weekend_practice[weekend_practice["driver_code"] == driver]
            if not driver_practice.empty:
                # Calculate FP rankings
                fp3_times = weekend_practice[weekend_practice["session_type"] == "Practice 3"]
                if not fp3_times.empty:
                    fp3_best = fp3_times.groupby("driver_code")["lap_time_ms"].min()
                    if driver in fp3_best.index:
                        driver_fp3_time = fp3_best[driver]
                        driver_features["current_fp3_best_ms"] = driver_fp3_time
                        driver_features["current_fp3_rank"] = (fp3_best <= driver_fp3_time).sum()
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

    if X_pred.empty:
        return []

    meta_pred["driver_code"] = driver_codes[: len(meta_pred)]
    driver_codes_final = driver_codes[: len(X_pred)]

    # Align features - add missing, select only needed (same as predict.py)
    for col in feature_names:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[feature_names]

    # Generate prediction
    try:
        pred_result = model.predict_top3(
            X_pred,
            driver_codes_final,
        )
        return pred_result["top3"]
    except Exception as e:
        logger.warning(f"Prediction failed: {e}")
        return []


def get_baseline_prediction(
    loader: F1DataLoader,
    session_type: str,
    year: int,
    round_num: int,
) -> dict[str, list[str]]:
    """
    Get baseline predictions for comparison.

    Returns dict mapping baseline name to predicted top-3.

    Baselines:
    - Qualifying (standard): FP3 fastest lap times
    - Qualifying (sprint): SQ finishing positions (no FP3 exists)
    - Race: Qualifying grid positions
    - Sprint Qualifying: FP1 fastest lap times
    - Sprint Race: SQ grid positions
    """
    baselines = {}

    if session_type == "qualifying":
        # FP3-based baseline (falls back to SQ on sprint weekends)
        fp3_baseline = PracticeBasedBaseline()
        baselines["FP3/SQ"] = fp3_baseline.predict_from_loader(loader, year, round_num)

    elif session_type == "sprint_quali":
        # FP1-based baseline
        fp1_baseline = FP1BasedBaseline()
        baselines["FP1"] = fp1_baseline.predict_from_loader(loader, year, round_num)

    elif session_type == "sprint_race":
        # SQ grid-based baseline
        sq_baseline = SprintQualiBasedBaseline()
        baselines["SQ Grid"] = sq_baseline.predict_from_loader(loader, year, round_num)

    elif session_type == "race":
        # Q grid-based baseline
        grid_baseline = RaceGridBaseline()
        baselines["Q Grid"] = grid_baseline.predict_from_loader(loader, year, round_num)

    return baselines


def backtest_race(
    loader: F1DataLoader,
    race_info: dict,
    session_types: list[str],
    cache_dir: Path | None = None,
) -> RaceBacktestResult:
    """
    Run backtest for a single race.

    For each session type:
    1. Train model with data up to this race
    2. Generate prediction
    3. Compare to actual results
    4. Generate baseline predictions for comparison
    """
    year = race_info["year"]
    round_num = race_info["round"]
    up_to_race = (year, round_num)

    result = RaceBacktestResult(
        year=year,
        round=round_num,
        circuit=race_info["circuit"],
        event_name=race_info["event_name"],
        is_sprint_weekend=race_info["is_sprint_weekend"],
    )

    for session_type in session_types:
        logger.info(f"  Processing {session_type}...")

        # Get actual results
        actual = get_actual_results(loader, year, round_num, session_type)
        if len(actual) < 3:
            logger.warning(f"    Skipping - insufficient actual results")
            continue

        # Train model
        model, feature_names, train_info = train_model_for_race(
            session_type, up_to_race, cache_dir
        )

        result.training_samples[session_type] = train_info["n_samples"]
        result.training_features[session_type] = train_info.get("n_features", 0)

        if model is None:
            logger.warning(f"    Skipping - insufficient training data")
            continue

        # Generate prediction
        predicted = predict_session(model, feature_names, loader, session_type, year, round_num)

        if len(predicted) < 3:
            logger.warning(f"    Skipping - prediction failed")
            continue

        # Score prediction
        metrics = evaluate_predictions(predicted, actual)

        result.session_results[session_type] = SessionPrediction(
            driver_codes=predicted,
            actual_codes=actual,
            game_points=metrics["game_points"],
            ndcg_3=metrics["ndcg_3"],
            map_3=metrics["map_3"],
            top3_inclusion=metrics["top3_inclusion"],
            exact_match_rate=metrics["exact_match_rate"],
        )

        # Get baseline predictions
        baseline_preds = get_baseline_prediction(loader, session_type, year, round_num)
        result.baseline_results[session_type] = {}

        for baseline_name, baseline_pred in baseline_preds.items():
            if len(baseline_pred) >= 3:
                baseline_points = calculate_game_points(baseline_pred[:3], actual)
                result.baseline_results[session_type][baseline_name] = BaselinePrediction(
                    name=baseline_name,
                    driver_codes=baseline_pred[:3],
                    game_points=baseline_points,
                )

    return result


def run_season_backtest(
    season: int,
    session_types: list[str] | None = None,
    cache_models: bool = False,
    start_round: int = 1,
) -> SeasonBacktestResult:
    """
    Run full season backtest.

    Args:
        season: Year to backtest
        session_types: Filter to specific types (None = all)
        cache_models: If True, cache trained models for faster re-runs
        start_round: First round to backtest (skip earlier rounds)

    Returns:
        SeasonBacktestResult with all race results and aggregated summary
    """
    loader = F1DataLoader(config.data.data_dir)
    races = get_season_races(loader, season)

    if not races:
        raise ValueError(f"No races found for season {season}")

    # Filter to start_round
    races = [r for r in races if r["round"] >= start_round]

    cache_dir = None
    if cache_models:
        cache_dir = config.models_dir / "backtest_cache" / str(season)

    results = []

    for i, race_info in enumerate(races):
        race_session_types = get_session_types_for_race(race_info)

        # Filter session types if specified
        if session_types:
            race_session_types = [s for s in race_session_types if s in session_types]

        if not race_session_types:
            continue

        print(f"Backtesting R{race_info['round']} {race_info['event_name']} ({i + 1}/{len(races)})")

        try:
            result = backtest_race(loader, race_info, race_session_types, cache_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    # Aggregate summary
    summary = aggregate_backtest_results(results)

    return SeasonBacktestResult(
        season=season,
        races=results,
        summary=summary,
        generated_at=datetime.now().isoformat(),
    )


def aggregate_backtest_results(results: list[RaceBacktestResult]) -> dict:
    """Compute aggregate metrics across all races."""
    summary = {
        "total_races": len(results),
        "session_metrics": {},
        "vs_baseline": {},
    }

    for session_type in ["qualifying", "sprint_quali", "sprint_race", "race"]:
        session_data = [
            r.session_results[session_type] for r in results if session_type in r.session_results
        ]

        if not session_data:
            continue

        game_points = [s.game_points for s in session_data]
        ndcg_scores = [s.ndcg_3 for s in session_data]
        map_scores = [s.map_3 for s in session_data]
        top3_inclusion = [s.top3_inclusion for s in session_data]
        exact_match = [s.exact_match_rate for s in session_data]

        summary["session_metrics"][session_type] = {
            "n_sessions": len(session_data),
            "game_points_mean": float(np.mean(game_points)),
            "game_points_std": float(np.std(game_points)),
            "game_points_total": sum(game_points),
            "game_points_max_possible": len(session_data) * 6,
            "ndcg_mean": float(np.mean(ndcg_scores)),
            "map_mean": float(np.mean(map_scores)),
            "top3_inclusion_mean": float(np.mean(top3_inclusion)),
            "exact_match_mean": float(np.mean(exact_match)),
            "distribution": {str(i): game_points.count(i) for i in range(7)},
        }

        # Compare to baselines
        baseline_points = {}
        for r in results:
            if session_type in r.baseline_results:
                for bname, bpred in r.baseline_results[session_type].items():
                    if bname not in baseline_points:
                        baseline_points[bname] = []
                    baseline_points[bname].append(bpred.game_points)

        summary["vs_baseline"][session_type] = {
            bname: {
                "mean": float(np.mean(pts)),
                "total": sum(pts),
            }
            for bname, pts in baseline_points.items()
        }

    return summary


# =============================================================================
# Output Formatting
# =============================================================================


def format_console_summary(result: SeasonBacktestResult) -> str:
    """Format backtest results for console output."""
    lines = [
        "",
        "=" * 70,
        f"SEASON {result.season} BACKTEST RESULTS",
        "=" * 70,
        f"Generated: {result.generated_at}",
        f"Total Races: {result.summary['total_races']}",
        "",
    ]

    for session_type, metrics in result.summary["session_metrics"].items():
        lines.append(f"{session_type.upper().replace('_', ' ')}:")
        lines.append(f"  Sessions: {metrics['n_sessions']}")
        lines.append(
            f"  Game Points: {metrics['game_points_mean']:.2f} +/- {metrics['game_points_std']:.2f}"
        )
        lines.append(f"  Total: {metrics['game_points_total']}/{metrics['game_points_max_possible']}")
        lines.append(f"  NDCG@3: {metrics['ndcg_mean']:.3f}")
        lines.append(f"  Top-3 Inclusion: {metrics['top3_inclusion_mean']:.1%}")

        # Distribution
        dist_parts = [f"{i}pts:{metrics['distribution'].get(str(i), 0)}" for i in range(7)]
        lines.append(f"  Distribution: {' | '.join(dist_parts)}")

        # Baseline comparison
        if session_type in result.summary["vs_baseline"]:
            for bname, bmetrics in result.summary["vs_baseline"][session_type].items():
                delta = metrics["game_points_mean"] - bmetrics["mean"]
                sign = "+" if delta > 0 else ""
                lines.append(f"  vs {bname}: {sign}{delta:.2f} pts/session")

        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_markdown_summary(result: SeasonBacktestResult) -> str:
    """Format results as markdown for README."""
    lines = [
        f"## {result.season} Season Performance",
        "",
        f"*Backtested with per-race model training using only data available before each race.*",
        "",
        "### Summary",
        "",
        "| Session Type | N | Avg Points | Total | Max | NDCG@3 | vs Baseline |",
        "|--------------|---|------------|-------|-----|--------|-------------|",
    ]

    for stype, metrics in result.summary["session_metrics"].items():
        # Get baseline comparison
        baseline_delta = ""
        if stype in result.summary["vs_baseline"]:
            for bname, bmetrics in result.summary["vs_baseline"][stype].items():
                delta = metrics["game_points_mean"] - bmetrics["mean"]
                sign = "+" if delta > 0 else ""
                baseline_delta = f"{sign}{delta:.2f} vs {bname}"
                break  # Just show first baseline

        lines.append(
            f"| {stype.replace('_', ' ').title()} | "
            f"{metrics['n_sessions']} | "
            f"{metrics['game_points_mean']:.2f} | "
            f"{metrics['game_points_total']} | "
            f"{metrics['game_points_max_possible']} | "
            f"{metrics['ndcg_mean']:.3f} | "
            f"{baseline_delta} |"
        )

    # Collapsible race-by-race details
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Race-by-Race Breakdown</summary>",
            "",
            "| Round | Circuit | Q | R | SQ | S |",
            "|-------|---------|---|---|----|----|",
        ]
    )

    for race in result.races:
        q_pts = (
            race.session_results["qualifying"].game_points
            if "qualifying" in race.session_results
            else "-"
        )
        r_pts = (
            race.session_results["race"].game_points if "race" in race.session_results else "-"
        )
        sq_pts = (
            race.session_results["sprint_quali"].game_points
            if "sprint_quali" in race.session_results
            else "-"
        )
        s_pts = (
            race.session_results["sprint_race"].game_points
            if "sprint_race" in race.session_results
            else "-"
        )

        lines.append(f"| R{race.round} | {race.circuit} | {q_pts} | {r_pts} | {sq_pts} | {s_pts} |")

    lines.extend(
        [
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def save_results(result: SeasonBacktestResult, output_dir: Path) -> tuple[Path, Path]:
    """Save results to JSON and markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON with full details
    json_path = output_dir / f"backtest_{result.season}.json"

    # Convert dataclasses to dicts
    result_dict = {
        "season": result.season,
        "generated_at": result.generated_at,
        "summary": result.summary,
        "races": [],
    }
    for race in result.races:
        race_dict = {
            "year": race.year,
            "round": race.round,
            "circuit": race.circuit,
            "event_name": race.event_name,
            "is_sprint_weekend": race.is_sprint_weekend,
            "training_samples": race.training_samples,
            "training_features": race.training_features,
            "session_results": {k: asdict(v) for k, v in race.session_results.items()},
            "baseline_results": {
                k: {bk: asdict(bv) for bk, bv in v.items()}
                for k, v in race.baseline_results.items()
            },
        }
        result_dict["races"].append(race_dict)

    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    # Markdown summary
    md_path = output_dir / f"backtest_{result.season}.md"
    with open(md_path, "w") as f:
        f.write(format_markdown_summary(result))

    return json_path, md_path


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run season backtest for F1 prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest full 2025 season (all session types)
  make backtest SEASON=2025

  # Backtest only qualifying predictions
  make backtest SEASON=2025 TYPE=qualifying

  # Backtest with model caching (faster re-runs)
  make backtest SEASON=2025 CACHE=1

  # Start from specific round
  make backtest SEASON=2025 START=10
        """,
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year to backtest",
    )
    parser.add_argument(
        "--type",
        choices=["qualifying", "sprint_quali", "sprint_race", "race", "all"],
        default="all",
        help="Session type to evaluate (default: all)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache trained models for faster re-runs",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="First round to backtest (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: data/backtest/)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Determine session types
    session_types = None
    if args.type != "all":
        session_types = [args.type]

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else config.data.data_dir.parent / "backtest"

    # Run backtest
    print(f"\nStarting season {args.season} backtest...")
    print(f"Session types: {session_types or 'all'}")
    print(f"Model caching: {'enabled' if args.cache else 'disabled'}")
    print(f"Starting from round: {args.start_round}")
    print("")

    result = run_season_backtest(
        season=args.season,
        session_types=session_types,
        cache_models=args.cache,
        start_round=args.start_round,
    )

    # Display console summary
    print(format_console_summary(result))

    # Save results
    json_path, md_path = save_results(result, output_dir)
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
