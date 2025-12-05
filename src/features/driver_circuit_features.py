"""
Driver-circuit interaction feature extraction.

Captures driver-specific performance patterns at circuit types and computes
circuit-specific teammate comparisons. These interaction features go beyond
simple driver or circuit features by capturing how specific drivers perform
on specific types of tracks.

Circuit types are classified by characteristics:
- Street circuits: tight, low-speed corners, walls close
- High-speed: long straights, high-speed corners, aero-dependent
- Technical: balance of slow/fast corners, rhythm important
- High-downforce: lots of corners, grip-limited

All features use temporal shift to prevent data leakage.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Circuit classification based on track characteristics
# Each circuit mapped to primary and secondary characteristics
CIRCUIT_TYPES: dict[str, dict[str, list[str]]] = {
    # Street circuits - walls close, low-speed corners, precision critical
    "Monaco": {"primary": "street", "characteristics": ["low_speed", "precision"]},
    "Monte Carlo": {"primary": "street", "characteristics": ["low_speed", "precision"]},
    "Marina Bay": {"primary": "street", "characteristics": ["street", "high_downforce"]},
    "Baku": {"primary": "street", "characteristics": ["street", "long_straight"]},
    "Las Vegas": {"primary": "street", "characteristics": ["street", "long_straight"]},
    "Jeddah": {"primary": "street", "characteristics": ["street", "high_speed"]},
    # High-speed circuits - long straights, aero efficiency matters
    "Monza": {"primary": "high_speed", "characteristics": ["long_straight", "low_downforce"]},
    "Spa-Francorchamps": {
        "primary": "high_speed",
        "characteristics": ["long_straight", "elevation"],
    },
    "Silverstone": {
        "primary": "high_speed",
        "characteristics": ["high_speed_corners", "aero_dependent"],
    },
    "Suzuka": {"primary": "technical", "characteristics": ["high_speed_corners", "rhythm"]},
    # Technical circuits - balance of corner types, driver skill important
    "Barcelona": {"primary": "technical", "characteristics": ["balanced", "tyre_degradation"]},
    "Budapest": {"primary": "technical", "characteristics": ["high_downforce", "low_speed"]},
    "Zandvoort": {"primary": "technical", "characteristics": ["high_downforce", "banking"]},
    "Imola": {"primary": "technical", "characteristics": ["old_school", "rhythm"]},
    "Mugello": {"primary": "technical", "characteristics": ["high_speed_corners", "elevation"]},
    "Portimão": {"primary": "technical", "characteristics": ["elevation", "blind_corners"]},
    "Nürburgring": {"primary": "technical", "characteristics": ["old_school", "elevation"]},
    "Istanbul": {"primary": "technical", "characteristics": ["multi_apex", "tyre_degradation"]},
    # High-downforce circuits - lots of corners, mechanical grip important
    "Melbourne": {"primary": "high_downforce", "characteristics": ["street_hybrid", "bumpy"]},
    "Montréal": {"primary": "high_downforce", "characteristics": ["stop_go", "walls"]},
    "Austin": {"primary": "technical", "characteristics": ["elevation", "high_speed_corners"]},
    "Mexico City": {
        "primary": "high_downforce",
        "characteristics": ["high_altitude", "long_straight"],
    },
    "São Paulo": {"primary": "technical", "characteristics": ["elevation", "anti_clockwise"]},
    "Spielberg": {"primary": "high_speed", "characteristics": ["short_lap", "elevation"]},
    "Le Castellet": {"primary": "technical", "characteristics": ["runoff", "tyre_degradation"]},
    "Sochi": {"primary": "technical", "characteristics": ["street_hybrid", "long_corners"]},
    "Shanghai": {"primary": "technical", "characteristics": ["long_straight", "back_straight"]},
    # Middle East circuits
    "Sakhir": {"primary": "technical", "characteristics": ["night_race", "sand"]},
    "Yas Island": {"primary": "technical", "characteristics": ["night_race", "street_hybrid"]},
    "Lusail": {"primary": "high_speed", "characteristics": ["night_race", "flowing"]},
    # Miami
    "Miami": {"primary": "street", "characteristics": ["street_hybrid", "long_straight"]},
    "Miami Gardens": {"primary": "street", "characteristics": ["street_hybrid", "long_straight"]},
}

# Fallback for unknown circuits
DEFAULT_CIRCUIT_TYPE = {"primary": "technical", "characteristics": ["unknown"]}


class DriverCircuitInteractionExtractor:
    """Extracts driver-circuit interaction features."""

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize driver-circuit interaction extractor.

        Args:
            windows: Rolling window sizes for temporal features (default: [3, 5])
        """
        self.windows = windows or [3, 5]
        self.circuit_types = CIRCUIT_TYPES
        logger.info(f"Initialized DriverCircuitInteractionExtractor (windows={self.windows})")

    def get_circuit_type(self, circuit: str) -> dict:
        """Get circuit type info for a circuit name."""
        return self.circuit_types.get(circuit, DEFAULT_CIRCUIT_TYPE)

    def get_similar_circuits(self, circuit: str) -> list[str]:
        """Get list of circuits with the same primary type."""
        target_type = self.get_circuit_type(circuit)["primary"]
        return [c for c, info in self.circuit_types.items() if info["primary"] == target_type]

    def extract_features(
        self,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract driver-circuit interaction features.

        Args:
            results: DataFrame with qualifying or race results including circuit and team

        Returns:
            DataFrame with driver-circuit interaction features per driver per session
        """
        logger.info("Extracting driver-circuit interaction features...")

        if results.empty:
            logger.warning("No results provided")
            return pd.DataFrame()

        required_cols = ["session_key", "driver_code", "year", "round", "position", "circuit"]
        if not all(c in results.columns for c in required_cols):
            missing = [c for c in required_cols if c not in results.columns]
            logger.warning(f"Missing required columns: {missing}")
            return pd.DataFrame()

        df = results.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        # Start with base columns
        features = df[["session_key", "driver_code", "year", "round", "circuit"]].copy()
        if "team" in df.columns:
            features["team"] = df["team"]

        # Add circuit type
        features["circuit_type"] = features["circuit"].apply(
            lambda c: self.get_circuit_type(c)["primary"]
        )

        # Add driver-circuit-type performance features
        features = self._add_circuit_type_features(features, df)

        # Add circuit-specific teammate delta features
        if "team" in df.columns:
            features = self._add_circuit_teammate_features(features, df)

        # Add driver affinity features (how much better/worse at this type vs overall)
        features = self._add_circuit_affinity_features(features, df)

        # Drop helper columns
        features = features.drop(columns=["circuit_type"], errors="ignore")

        n_features = len(
            [
                c
                for c in features.columns
                if c not in ["session_key", "driver_code", "year", "round", "circuit", "team"]
            ]
        )
        logger.info(f"Extracted {n_features} driver-circuit interaction features")

        return features

    def _add_circuit_type_features(
        self,
        features: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add driver performance features grouped by circuit type."""
        df = results.copy()
        df["circuit_type"] = df["circuit"].apply(lambda c: self.get_circuit_type(c)["primary"])

        # Create driver-circuit_type groups
        df["driver_circuit_type"] = df["driver_code"] + "_" + df["circuit_type"]
        df = df.sort_values(["driver_circuit_type", "year", "round"])

        # Count appearances at this circuit type
        features["circuit_type_appearances"] = df.groupby("driver_circuit_type").cumcount()

        # Historical average position at this circuit type
        features["circuit_type_avg_position"] = df.groupby("driver_circuit_type")[
            "position"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Best position at this circuit type
        features["circuit_type_best_position"] = df.groupby("driver_circuit_type")[
            "position"
        ].transform(lambda x: x.shift(1).expanding().min())

        # Top-3 rate at this circuit type
        df["is_top3"] = (df["position"] <= 3).astype(int)
        features["circuit_type_top3_rate"] = df.groupby("driver_circuit_type")["is_top3"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Position std at this circuit type (consistency)
        features["circuit_type_position_std"] = df.groupby("driver_circuit_type")[
            "position"
        ].transform(lambda x: x.shift(1).expanding().std())

        # Rolling windows for circuit type performance
        for window in self.windows:
            features[f"circuit_type_avg_pos_{window}"] = df.groupby("driver_circuit_type")[
                "position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        return features

    def _add_circuit_teammate_features(
        self,
        features: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add teammate comparison features specific to circuit types."""
        if "team" not in results.columns:
            return features

        df = results.copy()
        df["circuit_type"] = df["circuit"].apply(lambda c: self.get_circuit_type(c)["primary"])
        df = df.sort_values(["driver_code", "year", "round"])

        # Calculate teammate position for each session
        teammate_pos = []
        for _, row in df.iterrows():
            session = row["session_key"]
            driver = row["driver_code"]
            team = row["team"]

            teammate = results[
                (results["session_key"] == session)
                & (results["team"] == team)
                & (results["driver_code"] != driver)
            ]

            if teammate.empty:
                teammate_pos.append(np.nan)
            else:
                teammate_pos.append(teammate["position"].iloc[0])

        df["teammate_position"] = teammate_pos
        df["teammate_delta"] = df["position"] - df["teammate_position"]
        df["beats_teammate"] = (df["teammate_delta"] < 0).astype(int)

        # Circuit-specific teammate delta
        df["driver_circuit"] = df["driver_code"] + "_" + df["circuit"]
        df = df.sort_values(["driver_circuit", "year", "round"])

        # Average teammate delta at this specific circuit
        features["circuit_teammate_delta_avg"] = df.groupby("driver_circuit")[
            "teammate_delta"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Beats teammate rate at this specific circuit
        features["circuit_beats_teammate_rate"] = df.groupby("driver_circuit")[
            "beats_teammate"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Circuit TYPE teammate features
        df["driver_circuit_type"] = df["driver_code"] + "_" + df["circuit_type"]
        df = df.sort_values(["driver_circuit_type", "year", "round"])

        # Average teammate delta at this circuit type
        features["circuit_type_teammate_delta"] = df.groupby("driver_circuit_type")[
            "teammate_delta"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Beats teammate rate at circuit type
        features["circuit_type_beats_teammate_rate"] = df.groupby("driver_circuit_type")[
            "beats_teammate"
        ].transform(lambda x: x.shift(1).expanding().mean())

        return features

    def _add_circuit_affinity_features(
        self,
        features: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add features measuring driver's affinity for circuit types vs their overall average."""
        df = results.copy()
        df["circuit_type"] = df["circuit"].apply(lambda c: self.get_circuit_type(c)["primary"])
        df = df.sort_values(["driver_code", "year", "round"])

        # Calculate driver's overall average position (shifted)
        df["driver_overall_avg"] = df.groupby("driver_code")["position"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Calculate driver's circuit type average (shifted)
        df["driver_circuit_type"] = df["driver_code"] + "_" + df["circuit_type"]
        df = df.sort_values(["driver_circuit_type", "year", "round"])

        df["driver_type_avg"] = df.groupby("driver_circuit_type")["position"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Re-sort for proper assignment
        df = df.sort_values(["driver_code", "year", "round"])

        # Circuit type affinity = overall_avg - type_avg
        # Positive = driver performs BETTER at this type than their average
        # Negative = driver performs WORSE at this type than their average
        features["circuit_type_affinity"] = df["driver_overall_avg"] - df["driver_type_avg"]

        # Also add the raw averages for context
        features["driver_overall_avg_pos"] = df["driver_overall_avg"]

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = [
            # Circuit type features
            "circuit_type_appearances",
            "circuit_type_avg_position",
            "circuit_type_best_position",
            "circuit_type_top3_rate",
            "circuit_type_position_std",
            # Circuit-specific teammate features
            "circuit_teammate_delta_avg",
            "circuit_beats_teammate_rate",
            # Circuit type teammate features
            "circuit_type_teammate_delta",
            "circuit_type_beats_teammate_rate",
            # Affinity features
            "circuit_type_affinity",
            "driver_overall_avg_pos",
        ]

        # Add rolling window features
        for window in self.windows:
            names.append(f"circuit_type_avg_pos_{window}")

        return names
