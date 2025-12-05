"""
CLI for evaluating and scoring F1 predictions.

Provides command-line interface for scoring predictions and viewing historical performance.
"""

import argparse
import json
import logging
from pathlib import Path

from src.evaluation.scoring import score_prediction_breakdown

logger = logging.getLogger(__name__)


def load_prediction(prediction_id: str, predictions_dir: Path) -> dict:
    """Load prediction from file."""
    pred_file = predictions_dir / f"{prediction_id}.json"

    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    with open(pred_file) as f:
        prediction = json.load(f)

    logger.info(f"Loaded prediction {prediction_id}")
    return prediction


def load_actual_results(results_file: Path) -> dict:
    """Load actual race results from JSON."""
    with open(results_file) as f:
        results = json.load(f)

    logger.info(f"Loaded actual results from {results_file}")
    return results


def score_prediction(args):
    """Score a prediction against actual results."""
    logger.info(f"Scoring prediction {args.prediction_id}...")

    # Load prediction
    predictions_dir = Path(args.data_dir) / "predictions"
    prediction = load_prediction(args.prediction_id, predictions_dir)

    # Load actual results
    actual_results = load_actual_results(Path(args.actual_results))

    # Determine prediction type
    pred_type = prediction.get("prediction_type", "qualifying")

    # Extract predicted and actual top-3
    if pred_type == "qualifying":
        predicted = [
            prediction["qualifying_p1"],
            prediction["qualifying_p2"],
            prediction["qualifying_p3"],
        ]
        actual = [
            actual_results["qualifying"][0]["driver_id"],
            actual_results["qualifying"][1]["driver_id"],
            actual_results["qualifying"][2]["driver_id"],
        ]
    else:  # race
        predicted = [
            prediction["race_p1"],
            prediction["race_p2"],
            prediction["race_p3"],
        ]
        actual = [
            actual_results["race"][0]["driver_id"],
            actual_results["race"][1]["driver_id"],
            actual_results["race"][2]["driver_id"],
        ]

    # Score prediction
    breakdown = score_prediction_breakdown(predicted, actual)

    # Display results
    print("\nPrediction Scoring Results")
    print("=" * 60)
    print(f"Prediction ID: {args.prediction_id}")
    print(f"Race: {prediction['race_id']}")
    print(f"Type: {pred_type.capitalize()}")
    print("")
    print(f"Predicted: {', '.join(predicted)}")
    print(f"Actual:    {', '.join(actual)}")
    print("")
    print("Scoring Breakdown:")

    for detail in breakdown["position_details"]:
        result_icon = "✓" if detail["result"] in ["exact_match", "driver_match"] else "✗"
        print(
            f"  {result_icon} {detail['position']}: {detail['predicted']} vs {detail['actual']} "
            f"[+{detail['points']} points] {detail.get('note', '')}"
        )

    print("")
    print(f"Total Points: {breakdown['total_points']} / 6")
    print(f"Exact matches: {breakdown['exact_matches']}/3")
    print(f"Driver matches: {breakdown['driver_matches']}/3")
    print(f"Accuracy: {breakdown['accuracy_percent']:.1f}%")

    # Update prediction file with score
    if pred_type == "qualifying":
        prediction["qualifying_actual_score"] = breakdown["total_points"]
    else:
        prediction["race_actual_score"] = breakdown["total_points"]

    # Save updated prediction
    pred_file = predictions_dir / f"{args.prediction_id}.json"
    with open(pred_file, "w") as f:
        json.dump(prediction, f, indent=2, default=str)

    print("\nScore saved to prediction record.")

    return breakdown


def view_history(args):
    """View historical prediction performance."""
    logger.info(f"Viewing prediction history for season {args.season}...")

    predictions_dir = Path(args.data_dir) / "predictions"

    if not predictions_dir.exists():
        print("No predictions found.")
        return

    # Load all predictions
    predictions = []
    for pred_file in predictions_dir.glob("*.json"):
        try:
            with open(pred_file) as f:
                pred = json.load(f)
                predictions.append(pred)
        except Exception as e:
            logger.warning(f"Failed to load {pred_file}: {e}")

    if not predictions:
        print("No predictions found.")
        return

    # Filter by season if specified
    if args.season:
        predictions = [p for p in predictions if p["race_id"].startswith(str(args.season))]

    # Display summary
    print(f"\n{args.season or 'All'} Season Prediction Performance")
    print("=" * 60)
    print(f"Total Predictions: {len(predictions)}")

    # Calculate statistics
    scored_predictions = [
        p
        for p in predictions
        if p.get("qualifying_actual_score") is not None or p.get("race_actual_score") is not None
    ]

    if scored_predictions:
        total_score = sum(
            p.get("qualifying_actual_score", 0) + p.get("race_actual_score", 0)
            for p in scored_predictions
        )
        avg_score = total_score / len(scored_predictions)

        print(f"Scored Predictions: {len(scored_predictions)}")
        print(f"Average Score: {avg_score:.2f} / 6.0 points")

        # Best predictions
        best_preds = sorted(
            scored_predictions,
            key=lambda p: p.get("qualifying_actual_score", 0) + p.get("race_actual_score", 0),
            reverse=True,
        )[:5]

        print("\nTop 5 Predictions:")
        for pred in best_preds:
            score = pred.get("qualifying_actual_score", 0) + pred.get("race_actual_score", 0)
            print(f"  {pred['race_id']:20s} {score:.0f}/6 points")

    else:
        print("No scored predictions yet.")

    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate F1 predictions")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Score command
    score_parser = subparsers.add_parser("score", help="Score a prediction")
    score_parser.add_argument("--prediction-id", required=True, help="Prediction ID to score")
    score_parser.add_argument(
        "--actual-results", required=True, help="Path to JSON file with actual results"
    )
    score_parser.add_argument(
        "--data-dir", default="data/", help="Data directory containing predictions"
    )

    # History command
    history_parser = subparsers.add_parser("history", help="View prediction history")
    history_parser.add_argument("--season", type=int, help="Filter by season year")
    history_parser.add_argument(
        "--data-dir", default="data/", help="Data directory containing predictions"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Execute command
    if args.command == "score":
        score_prediction(args)
    elif args.command == "history":
        view_history(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
