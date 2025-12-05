"""
Relative and ranking feature extraction.

Computes comparative features: driver vs field average, driver vs teammate,
and position percentiles. These relative features capture how a driver
performs compared to others, which may be more predictive than absolute metrics.

All rolling features use .shift(1) for temporal safety.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RelativeFeatureExtractor:
    """Extracts relative performance features."""

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize relative feature extractor.

        Args:
            windows: Rolling window sizes for temporal features (default: [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10]
        logger.info(f"Initialized RelativeFeatureExtractor (windows={self.windows})")

    def extract_features(
        self,
        qualifying_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract relative features per driver per session.

        Args:
            qualifying_results: DataFrame with qualifying results

        Returns:
            DataFrame with relative features per driver per session
        """
        logger.info("Extracting relative features...")

        if qualifying_results.empty:
            logger.warning("No qualifying results provided")
            return pd.DataFrame()

        required_cols = ["session_key", "driver_code", "year", "round", "position"]
        if not all(c in qualifying_results.columns for c in required_cols):
            logger.warning("Missing required columns for relative features")
            return pd.DataFrame()

        # Start with base DataFrame
        features = qualifying_results[required_cols].copy()

        # Add team if available for teammate comparison
        if "team" in qualifying_results.columns:
            features["team"] = qualifying_results["team"]

        # Add position vs field features
        features = self._add_position_vs_field(features, qualifying_results)

        # Add teammate comparison features
        if "team" in features.columns:
            features = self._add_teammate_features(features, qualifying_results)

        # Add position percentile features
        features = self._add_percentile_features(features, qualifying_results)

        # Count features added
        n_features = len([c for c in features.columns if c not in required_cols + ["team"]])
        logger.info(f"Extracted {n_features} relative features")

        return features

    def _add_position_vs_field(
        self,
        features: pd.DataFrame,
        quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add position relative to field average features."""
        # Calculate field average per session
        field_avg = quali_results.groupby("session_key")["position"].mean().rename("field_avg_pos")

        features = features.merge(field_avg, left_on="session_key", right_index=True, how="left")

        # Current position vs field average
        features["current_pos_vs_field"] = features["position"] - features["field_avg_pos"]

        # Sort for rolling
        features = features.sort_values(["driver_code", "year", "round"])

        for window in self.windows:
            # Rolling average of position vs field
            features[f"avg_pos_vs_field_{window}"] = features.groupby("driver_code")[
                "current_pos_vs_field"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Drop helper columns (current_pos_vs_field is computed from target - would be leakage!)
        features = features.drop(columns=["field_avg_pos", "current_pos_vs_field"], errors="ignore")

        return features

    def _add_teammate_features(
        self,
        features: pd.DataFrame,
        quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add features comparing driver to teammate."""
        if "team" not in quali_results.columns:
            return features

        # Create teammate mapping per session
        quali_results.groupby(["session_key", "team", "driver_code"]).first()

        teammate_positions = []
        for _idx, row in features.iterrows():
            session = row["session_key"]
            driver = row["driver_code"]
            team = row.get("team")

            if pd.isna(team):
                teammate_positions.append(np.nan)
                continue

            # Find teammate
            team_drivers = quali_results[
                (quali_results["session_key"] == session)
                & (quali_results["team"] == team)
                & (quali_results["driver_code"] != driver)
            ]

            if team_drivers.empty:
                teammate_positions.append(np.nan)
            else:
                teammate_positions.append(team_drivers["position"].iloc[0])

        features["teammate_position"] = teammate_positions
        features["teammate_delta"] = features["position"] - features["teammate_position"]
        features["beats_teammate"] = (features["teammate_delta"] < 0).astype(int)

        # Sort for rolling
        features = features.sort_values(["driver_code", "year", "round"])

        for window in self.windows:
            # Rolling teammate delta
            features[f"avg_teammate_delta_{window}"] = features.groupby("driver_code")[
                "teammate_delta"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Rolling beats teammate rate
            features[f"beats_teammate_rate_{window}"] = features.groupby("driver_code")[
                "beats_teammate"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Drop helper columns (teammate_delta uses current position - would be leakage!)
        features = features.drop(
            columns=["teammate_position", "teammate_delta", "beats_teammate"], errors="ignore"
        )

        return features

    def _add_percentile_features(
        self,
        features: pd.DataFrame,
        quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add position percentile features."""
        # Calculate percentile rank within each session
        quali_with_pct = quali_results.copy()
        quali_with_pct["position_percentile"] = quali_with_pct.groupby("session_key")[
            "position"
        ].rank(pct=True)

        # Merge percentile
        pct_cols = quali_with_pct[["session_key", "driver_code", "position_percentile"]]
        features = features.merge(
            pct_cols,
            on=["session_key", "driver_code"],
            how="left",
        )

        # Sort for rolling
        features = features.sort_values(["driver_code", "year", "round"])

        for window in self.windows:
            # Rolling average percentile
            features[f"avg_percentile_{window}"] = features.groupby("driver_code")[
                "position_percentile"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Drop position_percentile (uses current position - would be leakage!)
        features = features.drop(columns=["position_percentile"], errors="ignore")

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        # Note: current_pos_vs_field, teammate_delta, beats_teammate, position_percentile
        # are helper columns that use current position (target) - they're dropped to prevent leakage
        names = []

        for window in self.windows:
            names.extend(
                [
                    f"avg_pos_vs_field_{window}",
                    f"avg_teammate_delta_{window}",
                    f"beats_teammate_rate_{window}",
                    f"avg_percentile_{window}",
                ]
            )

        return names
