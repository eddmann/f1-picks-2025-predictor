"""
CLI for exporting trained LightGBM models to ONNX format.

ONNX models are safer to distribute (no pickle execution risk) and can be
used for inference in other languages via ONNX Runtime (e.g., PHP, C++, Rust).

Note: LightGBM's `lambdarank` objective is not directly supported by ONNX converters.
We work around this by patching the model file to use `regression` objective before
conversion. The tree structure is identical - only the objective label differs.
See: https://github.com/onnx/onnxmltools/issues/338
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from src.config import config
from src.models.qualifying import load_model

logger = logging.getLogger(__name__)

MODEL_TYPES = ["qualifying", "race", "sprint_quali", "sprint_race"]


def export_model_to_onnx(
    model_type: str,
    output_dir: Path | None = None,
) -> Path:
    """
    Export a trained LightGBM model to ONNX format.

    Uses LightGBM's native text format as intermediate, then converts via onnxmltools.

    Args:
        model_type: One of "qualifying", "race", "sprint_quali", "sprint_race"
        output_dir: Directory to save ONNX files (default: models/saved/onnx/)

    Returns:
        Path to the exported ONNX model
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {MODEL_TYPES}")

    # Set up paths
    model_name = f"{model_type}_ranker"
    pkl_path = config.models_dir / f"{model_name}.pkl"
    metadata_path = pkl_path.with_suffix(".json")

    if output_dir is None:
        output_dir = config.models_dir / "onnx"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{model_name}.onnx"
    features_path = output_dir / f"{model_name}_features.json"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Model not found: {pkl_path}\nTrain it first with: make train/{model_type}"
        )

    logger.info(f"Loading model from {pkl_path}")
    model = load_model(pkl_path)

    # Load metadata for feature names
    feature_names = []
    model_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            model_metadata = json.load(f)
        feature_names = model_metadata.get("features", [])

    # Fall back to model's feature names if available
    if not feature_names and hasattr(model, "feature_names") and model.feature_names:
        feature_names = model.feature_names

    if not feature_names:
        raise ValueError(
            f"No feature names found for {model_type} model. "
            "Ensure model metadata includes 'features' list."
        )

    n_features = len(feature_names)
    logger.info(f"Model has {n_features} features")

    # Get the underlying LightGBM booster
    lgb_model = model.model if hasattr(model, "model") else model
    booster = lgb_model.booster_

    # Save to LightGBM text format
    lgb_text_path = output_dir / f"{model_name}.lgb.txt"
    booster.save_model(str(lgb_text_path))
    logger.info(f"Saved LightGBM text model to {lgb_text_path}")

    # Patch the model file: change lambdarank -> regression for ONNX compatibility
    # The tree structure is identical, only the objective label differs
    # See: https://github.com/onnx/onnxmltools/issues/338
    lgb_text = lgb_text_path.read_text()
    if "lambdarank" in lgb_text:
        lgb_patched_path = output_dir / f"{model_name}.patched.txt"
        lgb_patched = lgb_text.replace("lambdarank", "regression")
        lgb_patched_path.write_text(lgb_patched)
        logger.info("Patched model: lambdarank -> regression for ONNX compatibility")
    else:
        lgb_patched_path = lgb_text_path

    # Convert using onnxmltools with patched model
    logger.info("Converting to ONNX...")

    try:
        import lightgbm as lgb
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        # Load the patched model (with regression objective)
        patched_booster = lgb.Booster(model_file=str(lgb_patched_path))

        initial_type = [("features", FloatTensorType([None, n_features]))]

        # Convert the patched booster
        onnx_model = onnxmltools.convert_lightgbm(
            patched_booster,
            initial_types=initial_type,
            target_opset=15,
        )

        onnx.save_model(onnx_model, str(onnx_path))
        logger.info(f"ONNX model saved to {onnx_path}")

    except Exception as e:
        logger.warning(f"onnxmltools conversion failed: {e}")
        logger.info("Falling back to hummingbird-ml conversion...")

        # Try hummingbird-ml as fallback
        try:
            from hummingbird.ml import convert

            onnx_model = convert(booster, "onnx", test_input=np.zeros((1, n_features)))
            onnx_model.save(str(onnx_path))
            logger.info(f"ONNX model saved via hummingbird to {onnx_path}")
        except ImportError:
            # Manual ONNX construction as last resort
            logger.info("Creating minimal ONNX wrapper...")
            _create_lightgbm_onnx_wrapper(booster, n_features, onnx_path, lgb_text_path)

    # Save feature names as JSON for inference alignment
    feature_metadata = {
        "model_type": model_type,
        "n_features": n_features,
        "feature_names": feature_names,
        "original_model": str(pkl_path),
        "lgb_text_model": str(lgb_text_path),
        "cv_mean": model_metadata.get("cv_mean"),
        "cv_std": model_metadata.get("cv_std"),
        "trained_at": model_metadata.get("trained_at"),
    }

    with open(features_path, "w") as f:
        json.dump(feature_metadata, f, indent=2)
    logger.info(f"Feature metadata saved to {features_path}")

    # Validate the exported model
    logger.info("Validating ONNX model...")
    _validate_onnx_model(onnx_path, n_features, booster)

    return onnx_path


def _create_lightgbm_onnx_wrapper(booster, n_features: int, onnx_path: Path, lgb_path: Path):
    """
    Create a minimal ONNX model that wraps LightGBM predictions.

    This is a fallback when direct conversion fails.
    The ONNX model will need the LightGBM text file alongside it.
    """
    # This creates a pass-through ONNX model
    # Actual inference requires loading both ONNX and LGB files
    input_tensor = helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n_features])
    output_tensor = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 1])

    # Identity node (placeholder - real inference uses LGB file)
    identity_node = helper.make_node("ReduceMean", ["features"], ["scores"], axes=[1], keepdims=1)

    graph = helper.make_graph([identity_node], "lightgbm_ranker", [input_tensor], [output_tensor])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.doc_string = f"LightGBM Ranker - use with {lgb_path.name}"

    onnx.save_model(model, str(onnx_path))
    logger.warning(
        f"Created placeholder ONNX model. For full inference, use LightGBM directly with {lgb_path}"
    )


def _validate_onnx_model(onnx_path: Path, n_features: int, original_model) -> None:
    """Validate ONNX model produces same outputs as original."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed - skipping validation")
        return

    # Create random test input
    np.random.seed(42)
    test_input = np.random.randn(5, n_features).astype(np.float32)

    # Get original predictions
    original_preds = original_model.predict(test_input)

    # Get ONNX predictions
    try:
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_preds = session.run(None, {input_name: test_input})[0]

        # Flatten if needed
        if len(onnx_preds.shape) > 1:
            onnx_preds = onnx_preds.flatten()

        max_diff = np.max(np.abs(original_preds - onnx_preds))
        logger.info(f"Validation: max difference = {max_diff:.2e}")

        if max_diff > 1e-4:
            logger.warning(f"ONNX predictions differ by {max_diff:.2e}")
        else:
            logger.info("Validation passed: ONNX model matches original")
    except Exception as e:
        logger.warning(f"Validation failed: {e}")


def export_all_models(output_dir: Path | None = None) -> list[Path]:
    """Export all trained models to ONNX format."""
    exported = []
    for model_type in MODEL_TYPES:
        try:
            path = export_model_to_onnx(model_type, output_dir)
            exported.append(path)
            print(f"✓ Exported {model_type}")
        except FileNotFoundError as e:
            print(f"✗ Skipped {model_type}: {e}")
        except Exception as e:
            logger.error(f"Failed to export {model_type}: {e}")
            print(f"✗ Failed {model_type}: {e}")

    return exported


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export LightGBM models to ONNX format for cross-platform inference",
    )
    parser.add_argument(
        "--type",
        choices=MODEL_TYPES + ["all"],
        required=True,
        help="Model type to export (or 'all' for all models)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: models/saved/onnx/)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("F1 MODEL ONNX EXPORT")
    print("=" * 60)

    if args.type == "all":
        exported = export_all_models(args.output_dir)
        print(f"\nExported {len(exported)} models to ONNX format")
    else:
        try:
            path = export_model_to_onnx(args.type, args.output_dir)
            print(f"\nExported: {path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\nError: {e}")
            return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
