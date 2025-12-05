"""
Race pace feature extraction.

These features capture WHO beats their grid position in races, which is
the key signal needed to outperform the trivial "just use grid position" baseline.

Key insights from analysis:
- 30% of podiums come from drivers starting outside grid top 3
- Some drivers consistently gain positions from their grid slot
- Some drivers consistently lose positions from their grid slot
- First lap performance is a major factor in position changes
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class RacePaceFeatureExtractor:
    """
    Extract features that predict who will beat their grid position.

    These features are specifically designed to help the race model
    outperform the trivial grid position baseline.
    """

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize race pace feature extractor.

        Args:
            windows: Rolling window sizes (default: [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10]

    def extract_features(
        self,
        race_results: pd.DataFrame,
        race_laps: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Extract race pace features from historical race results.

        Args:
            race_results: DataFrame with race results including grid_position and position
            race_laps: Optional DataFrame with lap-by-lap data for first lap analysis

        Returns:
            DataFrame with race pace features per driver per race
        """
        if race_results.empty:
            return pd.DataFrame()

        required_cols = ["year", "round", "driver_code", "position", "grid_position"]
        missing = [c for c in required_cols if c not in race_results.columns]
        if missing:
            logger.warning(f"Missing columns for race pace features: {missing}")
            return pd.DataFrame()

        df = race_results.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        # Calculate positions gained (positive = gained positions)
        df["positions_gained"] = df["grid_position"] - df["position"]

        # Track DNFs (position > 20 or null typically indicates DNF)
        df["is_dnf"] = (df["position"] > 20) | df["position"].isna()

        features = self._extract_position_change_features(df)

        # Add first lap features if lap data available
        if race_laps is not None and not race_laps.empty:
            lap1_features = self._extract_first_lap_features(race_laps, df)
            if not lap1_features.empty:
                features = features.merge(
                    lap1_features,
                    on=["year", "round", "driver_code"],
                    how="left",
                )

        # Add DNF features
        dnf_features = self._extract_dnf_features(df)
        if not dnf_features.empty:
            features = features.merge(
                dnf_features,
                on=["year", "round", "driver_code"],
                how="left",
            )

        logger.info(f"Extracted {len(features.columns) - 3} race pace features")
        return features

    def _extract_position_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features based on historical position changes.

        These identify "race specialists" who consistently gain positions
        and "quali specialists" who consistently lose positions.
        """
        features = df[["year", "round", "driver_code"]].copy()

        for window in self.windows:
            # Rolling average positions gained (SHIFTED to prevent leakage)
            features[f"avg_positions_gained_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Rolling std of positions gained (consistency)
            features[f"positions_gained_std_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())

            # Rate of gaining positions (% of races where gained)
            features[f"gain_rate_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).apply(lambda y: (y > 0).mean())
            )

            # Rate of losing positions
            features[f"loss_rate_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).apply(lambda y: (y < 0).mean())
            )

            # Max positions gained in window (best recovery drive)
            features[f"max_positions_gained_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())

            # Conversion rates (front row -> podium, pole -> win)
            # These require grid position context
            df_temp = df.copy()
            df_temp["front_row_podium"] = (
                (df_temp["grid_position"] <= 2) & (df_temp["position"] <= 3)
            ).astype(float)
            df_temp["pole_win"] = (
                (df_temp["grid_position"] == 1) & (df_temp["position"] == 1)
            ).astype(float)

            features[f"front_row_podium_rate_{window}"] = df_temp.groupby("driver_code")[
                "front_row_podium"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            features[f"pole_win_rate_{window}"] = df_temp.groupby("driver_code")[
                "pole_win"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Race specialist flags (based on longer-term average)
        # These are binary indicators of consistent patterns
        features["race_specialist"] = (features["avg_positions_gained_10"] > 0.5).astype(float)
        features["quali_specialist"] = (features["avg_positions_gained_10"] < -0.5).astype(float)
        features["consistent_gainer"] = (
            (features["avg_positions_gained_5"] > 0) & (features["gain_rate_5"] > 0.6)
        ).astype(float)
        features["consistent_loser"] = (
            (features["avg_positions_gained_5"] < 0) & (features["loss_rate_5"] > 0.6)
        ).astype(float)

        return features

    def _extract_first_lap_features(
        self,
        race_laps: pd.DataFrame,
        race_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract first lap position change features.

        First lap (especially Turn 1) is where many position changes happen.
        Some drivers consistently gain positions on lap 1.
        """
        if "lap_number" not in race_laps.columns:
            return pd.DataFrame()

        # Get lap 1 data
        lap1 = race_laps[race_laps["lap_number"] == 1].copy()

        if lap1.empty:
            return pd.DataFrame()

        # Need position at end of lap 1
        if "position" not in lap1.columns:
            return pd.DataFrame()

        # Merge with grid position from race results
        lap1 = lap1.merge(
            race_results[["year", "round", "driver_code", "grid_position"]],
            on=["year", "round", "driver_code"],
            how="left",
        )

        # Calculate lap 1 positions gained
        lap1["lap1_positions_gained"] = lap1["grid_position"] - lap1["position"]

        # Aggregate to driver level with rolling features
        features_list = []

        for driver in lap1["driver_code"].unique():
            driver_data = lap1[lap1["driver_code"] == driver].sort_values(["year", "round"])

            for window in self.windows:
                driver_data[f"lap1_avg_gain_{window}"] = (
                    driver_data["lap1_positions_gained"]
                    .shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )

            # T1 specialist flag
            driver_data["t1_specialist"] = (driver_data["lap1_avg_gain_5"] > 1.0).astype(float)

            features_list.append(driver_data)

        if not features_list:
            return pd.DataFrame()

        features = pd.concat(features_list, ignore_index=True)

        # Select relevant columns
        feature_cols = ["year", "round", "driver_code"]
        feature_cols += [c for c in features.columns if "lap1_" in c or "t1_" in c]

        return features[feature_cols].drop_duplicates()

    def _extract_dnf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract DNF/reliability features.

        DNF rate affects who finishes in the top 3.
        """
        features = df[["year", "round", "driver_code"]].copy()

        for window in self.windows:
            # DNF rate
            features[f"dnf_rate_{window}"] = df.groupby("driver_code")["is_dnf"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Finish rate (inverse of DNF)
            features[f"finish_rate_{window}"] = 1 - features[f"dnf_rate_{window}"]

        # Add team DNF rate if team column exists
        if "team" in df.columns:
            for window in self.windows:
                features[f"team_dnf_rate_{window}"] = df.groupby("team")["is_dnf"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Reliability flag
        features["reliable_driver"] = (features["dnf_rate_10"] < 0.1).astype(float)
        features["unreliable_driver"] = (features["dnf_rate_10"] > 0.2).astype(float)

        return features


def extract_race_pace_features(
    race_results: pd.DataFrame,
    race_laps: pd.DataFrame | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Convenience function to extract race pace features.

    Args:
        race_results: DataFrame with race results
        race_laps: Optional DataFrame with lap data
        windows: Rolling window sizes

    Returns:
        DataFrame with race pace features
    """
    extractor = RacePaceFeatureExtractor(windows=windows)
    return extractor.extract_features(race_results, race_laps)
