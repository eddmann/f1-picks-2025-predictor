"""
CLI for exporting computed features to JSON for cross-platform inference.

Exports driver features for a specific race in a format that can be consumed
by non-Python inference runtimes (PHP, Rust, etc.) using ONNX models.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.cli.predict import (
    build_prediction_features,
    get_upcoming_race_info,
    validate_race_id,
)
from src.config import config
from src.data.loaders import F1DataLoader

logger = logging.getLogger(__name__)

PREDICTION_TYPES = ["qualifying", "race", "sprint_quali", "sprint_race"]


def export_features_for_race(
    race_id: str,
    prediction_type: str,
    output_dir: Path | None = None,
) -> Path:
    """
    Export computed features for a race to JSON.

    Args:
        race_id: Race identifier (e.g., "2025-24" or "2025-abu-dhabi")
        prediction_type: One of "qualifying", "race", "sprint_quali", "sprint_race"
        output_dir: Output directory (default: models/saved/onnx/features/)

    Returns:
        Path to the exported JSON file
    """
    validate_race_id(race_id)

    if prediction_type not in PREDICTION_TYPES:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    if output_dir is None:
        output_dir = config.models_dir / "onnx" / "features"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load race info
    loader = F1DataLoader(config.data.data_dir)
    race_info = get_upcoming_race_info(loader, race_id)

    logger.info(
        f"Exporting {prediction_type} features for {race_info['event_name']} "
        f"({race_info['year']} R{race_info['round']})"
    )

    # Build features using the same pipeline as prediction
    X_pred, meta_pred, driver_codes = build_prediction_features(prediction_type, loader, race_info)

    if X_pred.empty:
        raise ValueError(f"No feature data available for {race_id}")

    # Load model metadata to get expected feature order
    model_name = f"{prediction_type}_ranker"
    metadata_path = config.models_dir / "onnx" / f"{model_name}_features.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Feature metadata not found: {metadata_path}\n"
            f"Run 'make export/onnx/{prediction_type}' first."
        )

    with open(metadata_path) as f:
        model_metadata = json.load(f)

    feature_names = model_metadata["feature_names"]

    # Ensure features match expected order, fill missing with 0
    missing_features = [f for f in feature_names if f not in X_pred.columns]
    if missing_features:
        logger.warning(f"Filling {len(missing_features)} missing features with 0")
        for f in missing_features:
            X_pred[f] = 0.0

    # Reorder to match model expectations
    X_ordered = X_pred[feature_names].copy()

    # Fill any NaN values
    X_ordered = X_ordered.fillna(0.0)

    # Build export structure
    drivers_data = []
    for i, driver_code in enumerate(driver_codes):
        driver_features = X_ordered.iloc[i].tolist()
        drivers_data.append(
            {
                "driver_code": driver_code,
                "team": meta_pred.iloc[i].get("team", "Unknown")
                if len(meta_pred) > i
                else "Unknown",
                "features": driver_features,
            }
        )

    export_data = {
        "race_id": race_id,
        "race_info": {
            "year": race_info["year"],
            "round": race_info["round"],
            "circuit": race_info["circuit"],
            "event_name": race_info["event_name"],
        },
        "prediction_type": prediction_type,
        "model_type": model_name,
        "exported_at": datetime.now().isoformat(),
        "n_features": len(feature_names),
        "n_drivers": len(drivers_data),
        "feature_names": feature_names,
        "drivers": drivers_data,
    }

    # Save to JSON
    output_file = output_dir / f"{race_id}_{prediction_type}.json"
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    logger.info(f"Exported features to {output_file}")
    logger.info(f"  Drivers: {len(drivers_data)}")
    logger.info(f"  Features per driver: {len(feature_names)}")

    return output_file


def export_all_types_for_race(race_id: str, output_dir: Path | None = None) -> list[Path]:
    """Export features for all prediction types for a race."""
    exported = []

    # Check if it's a sprint weekend
    loader = F1DataLoader(config.data.data_dir)
    race_info = get_upcoming_race_info(loader, race_id)

    # Determine which types to export
    types_to_export = ["qualifying", "race"]
    if loader.is_sprint_weekend(race_info["year"], race_info["round"]):
        types_to_export.extend(["sprint_quali", "sprint_race"])

    for pred_type in types_to_export:
        try:
            path = export_features_for_race(race_id, pred_type, output_dir)
            exported.append(path)
            print(f"✓ Exported {pred_type}")
        except Exception as e:
            logger.error(f"Failed to export {pred_type}: {e}")
            print(f"✗ Failed {pred_type}: {e}")

    return exported


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export computed features to JSON for cross-platform inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export qualifying features for Abu Dhabi 2025
  uv run python -m src.cli.export_features --race-id 2025-24 --type qualifying

  # Export all prediction types for a race
  uv run python -m src.cli.export_features --race-id 2025-qatar --type all

Output:
  models/saved/onnx/features/{race_id}_{type}.json

The JSON contains:
  - race_info: Event details
  - feature_names: Ordered list of feature names
  - drivers: Array of {driver_code, team, features[]}

Use with PHP inference:
  $features = json_decode(file_get_contents('2025-24_qualifying.json'), true);
  $result = $predictor->predictQualifying(
      array_column($features['drivers'], 'features'),
      array_column($features['drivers'], 'driver_code')
  );
        """,
    )
    parser.add_argument(
        "--race-id",
        required=True,
        help="Race ID (e.g., 2025-24 or 2025-abu-dhabi)",
    )
    parser.add_argument(
        "--type",
        choices=PREDICTION_TYPES + ["all"],
        required=True,
        help="Prediction type to export (or 'all')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: models/saved/onnx/features/)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("F1 FEATURE EXPORT")
    print("=" * 60)
    print(f"Race: {args.race_id}")
    print(f"Type: {args.type}")
    print()

    if args.type == "all":
        exported = export_all_types_for_race(args.race_id, args.output_dir)
        print(f"\nExported {len(exported)} feature files")
    else:
        try:
            path = export_features_for_race(args.race_id, args.type, args.output_dir)
            print(f"\nExported: {path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\nError: {e}")
            return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
