"""
Weather feature extraction for F1 predictions.

Extracts features from session weather conditions including temperature,
humidity, and wet/dry indicators. Weather data is available in FastF1
parquet files but was not previously used.

All rolling features use .shift(1) to prevent data leakage.
Current session weather is available for prediction (known before quali).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WeatherFeatureExtractor:
    """Extracts weather-related features from session data."""

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize weather feature extractor.

        Args:
            windows: Rolling window sizes for temporal features (default: [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10]
        logger.info(f"Initialized WeatherFeatureExtractor (windows={self.windows})")

    def extract_features(
        self,
        qualifying_sessions: pd.DataFrame,
        practice_sessions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Extract weather features per driver per session.

        Args:
            qualifying_sessions: DataFrame with qualifying session data
            practice_sessions: Optional DataFrame with practice session data

        Returns:
            DataFrame with weather features per driver per session
        """
        logger.info("Extracting weather features...")

        if qualifying_sessions.empty:
            logger.warning("No qualifying sessions provided")
            return pd.DataFrame()

        # Check for weather columns
        weather_cols = ["avg_air_temp", "avg_track_temp", "avg_humidity", "rainfall"]
        available_cols = [c for c in weather_cols if c in qualifying_sessions.columns]

        if not available_cols:
            logger.warning("No weather columns found in data")
            return self._create_empty_features(qualifying_sessions)

        # Start with base DataFrame
        features = qualifying_sessions[
            ["session_key", "driver_code", "year", "round", "position"]
        ].copy()

        # Add current session weather (available before quali starts)
        features = self._add_current_weather(features, qualifying_sessions)

        # Add weather normalization features
        features = self._add_normalized_weather(features, qualifying_sessions)

        # Add practice-to-quali weather delta (same weekend)
        if practice_sessions is not None and not practice_sessions.empty:
            features = self._add_practice_weather_delta(
                features, qualifying_sessions, practice_sessions
            )

        # Add historical weather performance features
        features = self._add_historical_weather_features(features, qualifying_sessions)

        # Count features added
        n_features = len(
            [
                c
                for c in features.columns
                if c not in ["session_key", "driver_code", "year", "round", "position"]
            ]
        )
        logger.info(f"Extracted {n_features} weather features")

        return features

    def _create_empty_features(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """Create empty features DataFrame when no weather data available."""
        features = quali_sessions[["session_key", "driver_code", "year", "round"]].copy()

        # Add placeholder columns with NaN
        for col in self.get_feature_names():
            features[col] = np.nan

        return features

    def _add_current_weather(
        self,
        features: pd.DataFrame,
        quali_sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add current session weather (known before quali starts)."""
        # Get session-level weather
        weather_cols = ["session_key"]
        if "avg_track_temp" in quali_sessions.columns:
            weather_cols.append("avg_track_temp")
        if "avg_air_temp" in quali_sessions.columns:
            weather_cols.append("avg_air_temp")
        if "avg_humidity" in quali_sessions.columns:
            weather_cols.append("avg_humidity")
        if "rainfall" in quali_sessions.columns:
            weather_cols.append("rainfall")

        if len(weather_cols) <= 1:
            return features

        session_weather = quali_sessions[weather_cols].drop_duplicates(subset=["session_key"])

        # Rename for clarity
        rename_map = {}
        if "avg_track_temp" in session_weather.columns:
            rename_map["avg_track_temp"] = "current_track_temp"
        if "avg_air_temp" in session_weather.columns:
            rename_map["avg_air_temp"] = "current_air_temp"
        if "avg_humidity" in session_weather.columns:
            rename_map["avg_humidity"] = "current_humidity"
        if "rainfall" in session_weather.columns:
            rename_map["rainfall"] = "is_wet_session"

        session_weather = session_weather.rename(columns=rename_map)

        # Merge to features
        features = features.merge(session_weather, on="session_key", how="left")

        # Convert wet session to int
        if "is_wet_session" in features.columns:
            features["is_wet_session"] = features["is_wet_session"].astype(int)

        return features

    def _add_normalized_weather(
        self,
        features: pd.DataFrame,
        quali_sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add weather features normalized against season averages."""
        if "avg_track_temp" not in quali_sessions.columns:
            return features

        # Calculate season average track temp
        season_avg = (
            quali_sessions.groupby("year")["avg_track_temp"].mean().rename("season_avg_track_temp")
        )

        features = features.merge(season_avg, left_on="year", right_index=True, how="left")

        if "current_track_temp" in features.columns:
            features["track_temp_vs_season"] = (
                features["current_track_temp"] - features["season_avg_track_temp"]
            )

        # Drop helper column
        features = features.drop(columns=["season_avg_track_temp"], errors="ignore")

        return features

    def _add_practice_weather_delta(
        self,
        features: pd.DataFrame,
        quali_sessions: pd.DataFrame,
        practice_sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add weather change from FP3 to qualifying."""
        if "avg_track_temp" not in practice_sessions.columns:
            return features

        # Get FP3 weather per round
        fp3_weather = practice_sessions[practice_sessions["session_type"] == "FP3"].copy()

        if fp3_weather.empty:
            return features

        fp3_by_round = (
            fp3_weather.groupby(["year", "round"])["avg_track_temp"]
            .first()
            .rename("fp3_track_temp")
        )

        features = features.merge(
            fp3_by_round,
            left_on=["year", "round"],
            right_index=True,
            how="left",
        )

        if "current_track_temp" in features.columns and "fp3_track_temp" in features.columns:
            features["fp_to_q_temp_delta"] = (
                features["current_track_temp"] - features["fp3_track_temp"]
            )

        # Drop helper column
        features = features.drop(columns=["fp3_track_temp"], errors="ignore")

        return features

    def _add_historical_weather_features(
        self,
        features: pd.DataFrame,
        quali_sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add rolling historical weather performance features."""
        if "avg_track_temp" not in quali_sessions.columns:
            return features

        # Create temp for merging position data
        pos_data = quali_sessions[["session_key", "driver_code", "position"]].copy()

        # Merge position to features if not already there
        if "position" not in features.columns:
            features = features.merge(
                pos_data,
                on=["session_key", "driver_code"],
                how="left",
            )

        # Classify sessions as hot/cold
        median_temp = quali_sessions["avg_track_temp"].median()
        features["_is_hot_session"] = (features["current_track_temp"] > median_temp).astype(int)
        features["_is_cold_session"] = (features["current_track_temp"] <= median_temp).astype(int)

        # Sort for rolling
        features = features.sort_values(["driver_code", "year", "round"])

        for window in self.windows:
            # Rolling wet session rate
            if "is_wet_session" in features.columns:
                features[f"wet_session_rate_{window}"] = features.groupby("driver_code")[
                    "is_wet_session"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Hot track performance (avg position in hot sessions)
            features[f"hot_track_avg_pos_{window}"] = (
                features.groupby("driver_code")
                .apply(
                    lambda g: g["position"]
                    .where(g["_is_hot_session"] == 1)
                    .shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

            # Cold track performance
            features[f"cold_track_avg_pos_{window}"] = (
                features.groupby("driver_code")
                .apply(
                    lambda g: g["position"]
                    .where(g["_is_cold_session"] == 1)
                    .shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )

        # Drop helper columns
        features = features.drop(columns=["_is_hot_session", "_is_cold_session"], errors="ignore")

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = [
            "current_track_temp",
            "current_air_temp",
            "current_humidity",
            "is_wet_session",
            "track_temp_vs_season",
            "fp_to_q_temp_delta",
        ]

        for window in self.windows:
            names.extend(
                [
                    f"wet_session_rate_{window}",
                    f"hot_track_avg_pos_{window}",
                    f"cold_track_avg_pos_{window}",
                ]
            )

        return names
