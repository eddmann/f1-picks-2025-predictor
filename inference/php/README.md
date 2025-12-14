# F1 Prediction - PHP ONNX Runtime

Run F1 model predictions in PHP using ONNX Runtime.

## Usage

```bash
# 1. Train models (if not already done)
make train

# 2. Export models to ONNX
make export/onnx

# 3. Export features for a race
make export/features RACE=2025-24 TYPE=qualifying

# 4. Run PHP prediction
make inference/php FEATURES=2025-24_qualifying.json
```

## Files

```
inference/php/
├── Dockerfile    # PHP 8.5 + ONNX Runtime
├── predict.php   # Single-file prediction script
└── README.md
```

## How It Works

1. **Python exports features** - `export_features.py` computes all 250+ features (ELO ratings, rolling averages, practice pace, etc.) and saves to JSON
2. **PHP loads features** - `predict.php` reads the pre-computed features
3. **ONNX Runtime runs inference** - The LightGBM model runs via ONNX Runtime
4. **Rankings output** - Drivers sorted by prediction score

## ONNX Conversion Note

LightGBM's `lambdarank` objective is not directly supported by ONNX converters.
The export script patches the model file to use `regression` objective before conversion.
The tree structure is identical - only the objective label differs, so predictions are the same.

See: https://github.com/onnx/onnxmltools/issues/338

## Feature Export Format

```json
{
  "race_info": {"year": 2025, "round": 24, "circuit": "Yas Marina", "event_name": "Abu Dhabi GP"},
  "prediction_type": "qualifying",
  "model_type": "qualifying_ranker",
  "n_features": 280,
  "drivers": [
    {"driver_code": "VER", "team": "Red Bull Racing", "features": [...]},
    {"driver_code": "NOR", "team": "McLaren", "features": [...]},
    ...
  ]
}
```

