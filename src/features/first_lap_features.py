"""
First lap position change feature extraction for F1 predictions.

Captures driver performance on lap 1 (especially Turn 1), which is where
many position changes happen. Some drivers consistently gain positions
on the first lap (good starts, aggressive driving), while others lose.

Research insights:
- Some drivers consistently gain positions on the first lap (good starts, aggressive driving)
- Some drivers excel at defending their position on lap 1
- First lap skill is partially driver-specific, partially car-dependent

All features use temporal shift to prevent data leakage.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FirstLapFeatureExtractor:
    """
    Extract first lap position change features for F1 prediction.

    Features capture driver-specific patterns in lap 1 position changes,
    identifying "start specialists" who consistently gain positions and
    "conservative starters" who tend to lose positions early.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        data_dir: Path | str = "data/fastf1",
    ):
        """
        Initialize first lap feature extractor.

        Args:
            windows: Rolling window sizes for temporal features
            data_dir: Directory containing FastF1 parquet files
        """
        self.windows = windows or [3, 5, 10]
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized FirstLapFeatureExtractor (windows={self.windows})")

    def _load_race_lap_data(self, min_year: int) -> pd.DataFrame:
        """
        Load lap-by-lap race data from parquet files.

        Args:
            min_year: Minimum year to load

        Returns:
            DataFrame with race lap data including lap 1 positions
        """
        sessions_dir = self.data_dir / "sessions"
        if not sessions_dir.exists():
            logger.warning(f"Sessions directory not found: {sessions_dir}")
            return pd.DataFrame()

        # Load all race sessions
        race_files = list(sessions_dir.glob("*_R.parquet"))
        if not race_files:
            logger.warning("No race session files found")
            return pd.DataFrame()

        dfs = []
        for file_path in race_files:
            # Parse filename: YYYY_RR_R.parquet
            parts = file_path.stem.split("_")
            if len(parts) >= 2:
                year = int(parts[0])
                if year >= min_year:
                    try:
                        df = pd.read_parquet(file_path)
                        dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        all_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(all_data)} laps from {len(dfs)} race sessions")
        return all_data

    def extract_features(
        self,
        target_results: pd.DataFrame,
        min_year: int = 2020,
    ) -> pd.DataFrame:
        """
        Extract first lap performance features.

        Args:
            target_results: Target session results (for merging context)
            min_year: Minimum year for loading race data

        Returns:
            DataFrame with first lap features per driver per session
        """
        logger.info("Extracting first lap features...")

        # Load race lap data
        race_laps = self._load_race_lap_data(min_year)

        if race_laps.empty:
            logger.warning("No race lap data available for first lap features")
            return pd.DataFrame()

        # Required columns for first lap analysis
        required_cols = [
            "session_key",
            "year",
            "round",
            "driver_code",
            "lap_number",
            "position",
            "grid_position",
        ]
        missing = [c for c in required_cols if c not in race_laps.columns]
        if missing:
            logger.warning(f"Missing columns for first lap features: {missing}")
            return pd.DataFrame()

        # Extract lap 1 data
        lap1_data = self._extract_lap1_positions(race_laps)

        if lap1_data.empty:
            return pd.DataFrame()

        # Calculate first lap performance features
        features = self._calculate_first_lap_features(lap1_data)

        if features.empty:
            return pd.DataFrame()

        n_features = len(
            [
                c
                for c in features.columns
                if c not in ["session_key", "driver_code", "year", "round"]
            ]
        )
        logger.info(f"Extracted {n_features} first lap features for {len(features)} samples")

        return features

    def _extract_lap1_positions(self, race_laps: pd.DataFrame) -> pd.DataFrame:
        """
        Extract lap 1 position data from race laps.

        Args:
            race_laps: Full lap data from races

        Returns:
            DataFrame with lap 1 position changes per driver per race
        """
        # Get lap 1 data only
        lap1 = race_laps[race_laps["lap_number"] == 1].copy()

        if lap1.empty:
            logger.warning("No lap 1 data found")
            return pd.DataFrame()

        # Keep only needed columns
        lap1 = lap1[
            ["session_key", "year", "round", "driver_code", "position", "grid_position", "team"]
        ].copy()

        # Calculate first lap positions gained (positive = gained positions)
        lap1["first_lap_positions_gained"] = lap1["grid_position"] - lap1["position"]

        # Calculate binary indicators
        lap1["first_lap_gained"] = (lap1["first_lap_positions_gained"] > 0).astype(int)
        lap1["first_lap_lost"] = (lap1["first_lap_positions_gained"] < 0).astype(int)
        lap1["first_lap_held"] = (lap1["first_lap_positions_gained"] == 0).astype(int)

        # Major gain (3+ positions)
        lap1["first_lap_major_gain"] = (lap1["first_lap_positions_gained"] >= 3).astype(int)
        lap1["first_lap_major_loss"] = (lap1["first_lap_positions_gained"] <= -3).astype(int)

        # Sort chronologically
        lap1 = lap1.sort_values(["driver_code", "year", "round"])

        logger.info(f"Extracted lap 1 data for {len(lap1)} driver-race combinations")
        return lap1

    def _calculate_first_lap_features(
        self,
        lap1_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate rolling first lap performance features.

        Args:
            lap1_data: Lap 1 position change data

        Returns:
            DataFrame with rolling features per driver per race
        """
        df = lap1_data.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        # Start with base columns
        features = df[["session_key", "driver_code", "year", "round"]].copy()
        if "team" in df.columns:
            features["team"] = df["team"]

        # Add driver first lap features (rolling, shifted to prevent leakage)
        for window in self.windows:
            # Average positions gained on lap 1
            features[f"first_lap_avg_gain_{window}"] = df.groupby("driver_code")[
                "first_lap_positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Std of lap 1 positions gained (consistency)
            features[f"first_lap_gain_std_{window}"] = df.groupby("driver_code")[
                "first_lap_positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())

            # Rate of gaining positions on lap 1
            features[f"first_lap_gain_rate_{window}"] = df.groupby("driver_code")[
                "first_lap_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Rate of losing positions on lap 1
            features[f"first_lap_loss_rate_{window}"] = df.groupby("driver_code")[
                "first_lap_lost"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Rate of holding position (stable starters)
            features[f"first_lap_hold_rate_{window}"] = df.groupby("driver_code")[
                "first_lap_held"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Max positions gained on lap 1 (best start)
            features[f"first_lap_max_gain_{window}"] = df.groupby("driver_code")[
                "first_lap_positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())

            # Min positions gained (worst start, will be negative for losses)
            features[f"first_lap_min_gain_{window}"] = df.groupby("driver_code")[
                "first_lap_positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())

        # Career first lap stats (expanding window)
        features["first_lap_career_avg_gain"] = df.groupby("driver_code")[
            "first_lap_positions_gained"
        ].transform(lambda x: x.shift(1).expanding().mean())

        features["first_lap_career_gain_rate"] = df.groupby("driver_code")[
            "first_lap_gained"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Season first lap stats
        features["first_lap_season_avg_gain"] = df.groupby(["driver_code", "year"])[
            "first_lap_positions_gained"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Binary specialist flags (based on longer-term average)
        # Start specialist: consistently gains 1+ positions on lap 1
        features["start_specialist"] = (features["first_lap_avg_gain_10"] >= 1.0).astype(float)

        # Start conservative: consistently loses positions on lap 1
        features["start_conservative"] = (features["first_lap_avg_gain_10"] <= -1.0).astype(float)

        # Start consistent: rarely changes position
        features["start_consistent"] = (features["first_lap_gain_std_10"] < 1.0).astype(float)

        # Start aggressive: high variance in lap 1 results
        features["start_aggressive"] = (features["first_lap_gain_std_10"] >= 2.0).astype(float)

        # Add team first lap features (team's average first lap performance)
        if "team" in df.columns:
            for window in self.windows:
                features[f"team_first_lap_avg_gain_{window}"] = df.groupby("team")[
                    "first_lap_positions_gained"
                ].transform(lambda x: x.shift(1).rolling(window * 2, min_periods=2).mean())

        # Relative to grid position expectations
        # From front row (grid <= 2), harder to gain positions
        df["from_front_row"] = (df["grid_position"] <= 2).astype(int)
        features["first_lap_front_row_hold_rate"] = (
            df.groupby("driver_code")
            .apply(lambda x: _front_row_hold_rate(x), include_groups=False)
            .reset_index(level=0, drop=True)
        )

        # From back of grid (grid >= 15), easier to gain positions
        df["from_back_grid"] = (df["grid_position"] >= 15).astype(int)
        features["first_lap_back_grid_gain_rate"] = (
            df.groupby("driver_code")
            .apply(lambda x: _back_grid_gain_rate(x), include_groups=False)
            .reset_index(level=0, drop=True)
        )

        # Last race first lap result (recent form)
        features["first_lap_last_race_gain"] = df.groupby("driver_code")[
            "first_lap_positions_gained"
        ].transform(lambda x: x.shift(1))

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = []

        # Rolling window features
        for window in self.windows:
            names.extend(
                [
                    f"first_lap_avg_gain_{window}",
                    f"first_lap_gain_std_{window}",
                    f"first_lap_gain_rate_{window}",
                    f"first_lap_loss_rate_{window}",
                    f"first_lap_hold_rate_{window}",
                    f"first_lap_max_gain_{window}",
                    f"first_lap_min_gain_{window}",
                    f"team_first_lap_avg_gain_{window}",
                ]
            )

        # Career/expanding features
        names.extend(
            [
                "first_lap_career_avg_gain",
                "first_lap_career_gain_rate",
                "first_lap_season_avg_gain",
                "start_specialist",
                "start_conservative",
                "start_consistent",
                "start_aggressive",
                "first_lap_front_row_hold_rate",
                "first_lap_back_grid_gain_rate",
                "first_lap_last_race_gain",
            ]
        )

        return names


def _front_row_hold_rate(driver_data: pd.DataFrame) -> pd.Series:
    """Calculate rate of holding position when starting from front row."""
    result = pd.Series(index=driver_data.index, dtype=float)

    front_row_mask = driver_data["from_front_row"] == 1
    held_mask = driver_data["first_lap_positions_gained"] >= 0  # Held or gained

    # Use expanding window with shift
    if front_row_mask.any():
        # Calculate on front row starts only
        fr_held = (front_row_mask & held_mask).astype(float)
        fr_starts = front_row_mask.astype(float)

        # Avoid division by zero
        cumsum_held = fr_held.shift(1).expanding().sum().fillna(0)
        cumsum_starts = fr_starts.shift(1).expanding().sum().fillna(0)

        result = cumsum_held / cumsum_starts.replace(0, np.nan)

    return result.fillna(0.5)  # Default to 50% if no front row starts


def _back_grid_gain_rate(driver_data: pd.DataFrame) -> pd.Series:
    """Calculate rate of gaining positions when starting from back of grid."""
    result = pd.Series(index=driver_data.index, dtype=float)

    back_grid_mask = driver_data["from_back_grid"] == 1
    gained_mask = driver_data["first_lap_positions_gained"] > 0

    if back_grid_mask.any():
        bg_gained = (back_grid_mask & gained_mask).astype(float)
        bg_starts = back_grid_mask.astype(float)

        cumsum_gained = bg_gained.shift(1).expanding().sum().fillna(0)
        cumsum_starts = bg_starts.shift(1).expanding().sum().fillna(0)

        result = cumsum_gained / cumsum_starts.replace(0, np.nan)

    return result.fillna(0.5)  # Default to 50% if no back grid starts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.loaders import F1DataLoader

    loader = F1DataLoader()

    # Load race results for context
    race_results = loader.load_race_results(min_year=2022)
    print(f"Loaded {len(race_results)} race results")

    # Extract first lap features
    extractor = FirstLapFeatureExtractor()
    features = extractor.extract_features(race_results, min_year=2022)

    print(f"\nFeature columns: {features.columns.tolist()}")
    print(f"\nFeature shape: {features.shape}")
    print("\nSample features:")
    print(features.head(20))

    # Check first lap gain distribution
    print("\nFirst lap avg gain (last 5) stats:")
    print(features["first_lap_avg_gain_5"].describe())

    # Show specialists
    print("\nStart specialists (last 10 races avg >= 1.0):")
    specialists = features[features["start_specialist"] == 1]["driver_code"].unique()
    print(specialists[:10])
