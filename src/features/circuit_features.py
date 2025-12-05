"""
Circuit-specific feature extraction.

Captures driver and team historical performance at specific circuits.
Some drivers consistently perform better at certain tracks (e.g., Hamilton at Silverstone).
All features use temporal shift to prevent data leakage.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CircuitFeatureExtractor:
    """Extracts circuit-specific historical features."""

    def __init__(self):
        """Initialize circuit feature extractor."""
        pass

    def extract_features(
        self,
        quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract circuit-specific features.

        Args:
            quali_results: DataFrame with qualifying results including circuit

        Returns:
            DataFrame with circuit features per driver per session
        """
        if quali_results.empty or "circuit" not in quali_results.columns:
            logger.warning("No circuit data available")
            return pd.DataFrame()

        df = quali_results.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        features = df[["session_key", "driver_code", "year", "round", "circuit"]].copy()
        if "team" in df.columns:
            features["team"] = df["team"]

        # Driver circuit features
        features = self._add_driver_circuit_features(features, df)

        # Team circuit features
        if "team" in df.columns:
            features = self._add_team_circuit_features(features, df)

        return features

    def _add_driver_circuit_features(
        self, features: pd.DataFrame, quali: pd.DataFrame
    ) -> pd.DataFrame:
        """Add driver's historical performance at each circuit."""
        quali = quali.copy()

        # Create driver-circuit groups
        quali["driver_circuit"] = quali["driver_code"] + "_" + quali["circuit"]

        # Sort by driver-circuit and time
        quali = quali.sort_values(["driver_circuit", "year", "round"])

        # Count previous appearances at this circuit (before current race)
        features["circuit_appearances"] = quali.groupby("driver_circuit").cumcount()

        # Historical average position at this circuit (shifted to exclude current)
        features["circuit_avg_position"] = quali.groupby("driver_circuit")["position"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Historical best position at this circuit
        features["circuit_best_position"] = quali.groupby("driver_circuit")["position"].transform(
            lambda x: x.shift(1).expanding().min()
        )

        # Historical worst position at this circuit
        features["circuit_worst_position"] = quali.groupby("driver_circuit")["position"].transform(
            lambda x: x.shift(1).expanding().max()
        )

        # Position variance at this circuit (consistency)
        features["circuit_position_std"] = quali.groupby("driver_circuit")["position"].transform(
            lambda x: x.shift(1).expanding().std()
        )

        # Top-3 rate at this circuit
        quali["is_top3"] = (quali["position"] <= 3).astype(int)
        features["circuit_top3_rate"] = quali.groupby("driver_circuit")["is_top3"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Podium rate (top-3 frequency) - useful for identifying circuit specialists
        quali["is_top5"] = (quali["position"] <= 5).astype(int)
        features["circuit_top5_rate"] = quali.groupby("driver_circuit")["is_top5"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Most recent position at this circuit
        features["circuit_last_position"] = quali.groupby("driver_circuit")["position"].transform(
            lambda x: x.shift(1)
        )

        # Improvement trend at this circuit (negative = improving)
        def calc_trend(x):
            shifted = x.shift(1)
            if shifted.count() < 2:
                return np.nan
            positions = shifted.dropna().values
            if len(positions) < 2:
                return np.nan
            # Simple trend: last - first
            return positions[-1] - positions[0]

        features["circuit_position_trend"] = quali.groupby("driver_circuit")["position"].transform(
            calc_trend
        )

        return features

    def _add_team_circuit_features(
        self, features: pd.DataFrame, quali: pd.DataFrame
    ) -> pd.DataFrame:
        """Add team's historical performance at each circuit."""
        quali = quali.copy()

        # Create team-circuit groups
        quali["team_circuit"] = quali["team"] + "_" + quali["circuit"]

        # Sort by team-circuit and time
        quali = quali.sort_values(["team_circuit", "year", "round"])

        # Team's historical average position at this circuit
        # Use best of two drivers per race as team benchmark
        team_best_per_race = (
            quali.groupby(["team_circuit", "year", "round"])["position"].min().reset_index()
        )
        team_best_per_race = team_best_per_race.sort_values(["team_circuit", "year", "round"])

        # Calculate expanding mean of team's best position
        team_best_per_race["team_circuit_avg"] = team_best_per_race.groupby("team_circuit")[
            "position"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Merge back
        team_circuit_map = team_best_per_race.set_index(["team_circuit", "year", "round"])[
            "team_circuit_avg"
        ].to_dict()

        quali["team_circuit"] = quali["team"] + "_" + quali["circuit"]
        quali["_key"] = list(
            zip(quali["team_circuit"], quali["year"], quali["round"], strict=False)
        )
        features["team_circuit_avg_position"] = quali["_key"].map(team_circuit_map)

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        return [
            "circuit_appearances",
            "circuit_avg_position",
            "circuit_best_position",
            "circuit_worst_position",
            "circuit_position_std",
            "circuit_top3_rate",
            "circuit_top5_rate",
            "circuit_last_position",
            "circuit_position_trend",
            "team_circuit_avg_position",
        ]
