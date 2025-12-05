"""
Model explanation using SHAP values.

Provides feature importance explanations for F1 prediction models:
- Global feature importance across all predictions
- Local explanations for individual predictions
- Visualization support for understanding model decisions
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    SHAP-based model explainer for F1 prediction models.

    Supports LightGBM ranking models and provides both global and local explanations.

    Example:
        explainer = ModelExplainer(model)
        explainer.fit(X_train)

        # Global importance
        importance = explainer.get_global_importance()

        # Explain a specific prediction
        local_exp = explainer.explain_prediction(X_single, driver_codes)
    """

    def __init__(self, model: Any, feature_names: list[str] | None = None):
        """
        Initialize explainer with a trained model.

        Args:
            model: Trained model (LGBMRanker or similar)
            feature_names: List of feature names (optional, extracted from model if available)
        """
        self.model = model
        self._shap_explainer = None
        self._background_data = None

        # Try to get feature names from model
        if feature_names:
            self.feature_names = feature_names
        elif hasattr(model, "feature_names_"):
            self.feature_names = model.feature_names_
        elif hasattr(model, "model") and hasattr(model.model, "feature_name_"):
            self.feature_names = model.model.feature_name_()
        else:
            self.feature_names = None

    def fit(self, X: pd.DataFrame | np.ndarray, sample_size: int = 100) -> "ModelExplainer":
        """
        Fit the SHAP explainer with background data.

        Args:
            X: Training data or representative sample
            sample_size: Number of samples for background (default: 100)

        Returns:
            self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = list(X.columns)
            X_values = X.values
        else:
            X_values = X

        # Sample background data if too large
        if len(X_values) > sample_size:
            indices = np.random.choice(len(X_values), sample_size, replace=False)
            self._background_data = X_values[indices]
        else:
            self._background_data = X_values

        # Create SHAP explainer
        # For LightGBM models, use TreeExplainer
        if hasattr(self.model, "model"):
            # Our wrapper classes store the actual model in .model
            lgbm_model = self.model.model
        else:
            lgbm_model = self.model

        try:
            self._shap_explainer = shap.TreeExplainer(lgbm_model)
            logger.info("Created TreeExplainer for LightGBM model")
        except Exception as e:
            logger.warning(f"TreeExplainer failed, falling back to KernelExplainer: {e}")
            # Fallback to KernelExplainer (slower but more general)

            def predict_fn(x):
                if hasattr(self.model, "predict"):
                    return self.model.predict(x)
                return lgbm_model.predict(x)

            self._shap_explainer = shap.KernelExplainer(predict_fn, self._background_data)

        return self

    def get_shap_values(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for given samples.

        Args:
            X: Feature matrix

        Returns:
            SHAP values array with shape (n_samples, n_features)
        """
        if self._shap_explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._shap_explainer.shap_values(X)

    def get_global_importance(
        self,
        X: pd.DataFrame | np.ndarray | None = None,
        top_k: int | None = None,
    ) -> dict[str, float]:
        """
        Get global feature importance based on mean absolute SHAP values.

        Args:
            X: Data to explain (uses background data if None)
            top_k: Return only top-k features (None for all)

        Returns:
            Dict mapping feature names to importance scores
        """
        if X is None:
            X = self._background_data

        shap_values = self.get_shap_values(X)

        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)

        # Create feature name mapping
        if self.feature_names:
            importance_dict = dict(zip(self.feature_names, importance, strict=False))
        else:
            importance_dict = {f"feature_{i}": v for i, v in enumerate(importance)}

        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        if top_k:
            sorted_importance = dict(list(sorted_importance.items())[:top_k])

        return sorted_importance

    def explain_prediction(
        self,
        X: pd.DataFrame | np.ndarray,
        driver_codes: list[str] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Explain a single prediction or set of predictions.

        Args:
            X: Feature matrix for prediction(s)
            driver_codes: Optional driver codes for labeling
            top_k: Number of top features to include per driver

        Returns:
            Dict with explanation details per driver
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X

        shap_values = self.get_shap_values(X_values)

        explanations = {}
        n_samples = len(X_values)

        for i in range(n_samples):
            driver = driver_codes[i] if driver_codes else f"sample_{i}"
            sample_shap = shap_values[i]

            # Get top positive and negative contributors
            feature_contributions = []
            for j, shap_val in enumerate(sample_shap):
                feature_name = self.feature_names[j] if self.feature_names else f"feature_{j}"
                feature_contributions.append(
                    {
                        "feature": feature_name,
                        "shap_value": float(shap_val),
                        "contribution": "positive" if shap_val > 0 else "negative",
                    }
                )

            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            explanations[driver] = {
                "total_shap": float(sum(sample_shap)),
                "top_features": feature_contributions[:top_k],
                "positive_contributors": [f for f in feature_contributions if f["shap_value"] > 0][
                    :5
                ],
                "negative_contributors": [f for f in feature_contributions if f["shap_value"] < 0][
                    :5
                ],
            }

        return explanations

    def explain_ranking(
        self,
        X: pd.DataFrame | np.ndarray,
        driver_codes: list[str],
        predicted_ranking: list[str],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Explain why drivers were ranked in a certain order.

        Args:
            X: Feature matrix
            driver_codes: Driver codes matching X rows
            predicted_ranking: Model's predicted ranking (driver codes)
            top_k: Number of top features to show

        Returns:
            Dict with ranking explanation
        """
        explanations = self.explain_prediction(X, driver_codes, top_k=top_k)

        # Add ranking context
        ranking_explanation = {
            "predicted_ranking": predicted_ranking[:3],
            "drivers": {},
        }

        for rank, driver in enumerate(predicted_ranking[:3], 1):
            if driver in explanations:
                driver_exp = explanations[driver]
                ranking_explanation["drivers"][driver] = {
                    "predicted_position": rank,
                    "total_score": driver_exp["total_shap"],
                    "key_factors": [
                        f"{f['feature']}: {f['shap_value']:+.3f}"
                        for f in driver_exp["top_features"][:top_k]
                    ],
                }

        # Add comparison between P1 and P2
        if len(predicted_ranking) >= 2:
            p1, p2 = predicted_ranking[0], predicted_ranking[1]
            if p1 in explanations and p2 in explanations:
                p1_shap = explanations[p1]["total_shap"]
                p2_shap = explanations[p2]["total_shap"]
                ranking_explanation["p1_vs_p2"] = {
                    "margin": p1_shap - p2_shap,
                    "explanation": f"{p1} ranked higher due to {p1_shap - p2_shap:+.3f} SHAP margin",
                }

        return ranking_explanation

    def format_explanation(self, explanation: dict, verbose: bool = False) -> str:
        """
        Format explanation as human-readable string.

        Args:
            explanation: Output from explain_prediction or explain_ranking
            verbose: Include all details if True

        Returns:
            Formatted string
        """
        lines = [
            "",
            "=" * 60,
            "PREDICTION EXPLANATION (SHAP)",
            "=" * 60,
        ]

        if "predicted_ranking" in explanation:
            # Ranking explanation
            lines.append(f"\nPredicted Top 3: {', '.join(explanation['predicted_ranking'])}")

            for driver, details in explanation.get("drivers", {}).items():
                lines.append(f"\n{driver} (P{details['predicted_position']}):")
                lines.append(f"  Total SHAP Score: {details['total_score']:.3f}")
                lines.append("  Key Factors:")
                for factor in details["key_factors"]:
                    lines.append(f"    - {factor}")

            if "p1_vs_p2" in explanation:
                lines.append(f"\n{explanation['p1_vs_p2']['explanation']}")

        else:
            # Per-driver explanation
            for driver, details in explanation.items():
                lines.append(f"\n{driver}:")
                lines.append(f"  Total SHAP: {details['total_shap']:.3f}")

                if verbose:
                    lines.append("  Top Contributing Features:")
                    for f in details["top_features"]:
                        lines.append(
                            f"    {f['feature']}: {f['shap_value']:+.4f} ({f['contribution']})"
                        )
                else:
                    # Compact view
                    pos = [f["feature"] for f in details["positive_contributors"][:3]]
                    neg = [f["feature"] for f in details["negative_contributors"][:3]]
                    if pos:
                        lines.append(f"  Boosted by: {', '.join(pos)}")
                    if neg:
                        lines.append(f"  Hurt by: {', '.join(neg)}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def save_explanation(
        self,
        explanation: dict,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """
        Save explanation to file.

        Args:
            explanation: Explanation dict
            output_path: Output file path
            format: Output format ('json' or 'txt')
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(explanation, f, indent=2, default=float)
        else:
            with open(output_path, "w") as f:
                f.write(self.format_explanation(explanation, verbose=True))

        logger.info(f"Explanation saved to {output_path}")


def explain_model_prediction(
    model: Any,
    X: pd.DataFrame,
    driver_codes: list[str],
    predicted_ranking: list[str],
    X_train: pd.DataFrame | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Convenience function to explain a model prediction.

    Args:
        model: Trained model
        X: Feature matrix for prediction
        driver_codes: Driver codes
        predicted_ranking: Model's predicted ranking
        X_train: Training data for background (optional)
        top_k: Number of features to show

    Returns:
        Explanation dict
    """
    explainer = ModelExplainer(model)

    # Fit with training data or prediction data
    if X_train is not None:
        explainer.fit(X_train)
    else:
        explainer.fit(X)

    return explainer.explain_ranking(X, driver_codes, predicted_ranking, top_k=top_k)
