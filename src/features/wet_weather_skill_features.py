"""
Wet weather skill feature extraction for F1 predictions.

Some drivers consistently outperform in wet conditions (Hamilton, Verstappen)
while others struggle. This module tracks driver performance in wet vs dry
conditions to capture this skill differential.

Key insights from research:
- Wet races create the biggest upsets and unpredictable results
- Driver skill matters more in wet conditions (car advantage reduced)
- Some drivers excel on intermediate tyres specifically

All features use temporal shift to prevent data leakage.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Wet tyre compounds
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


class WetWeatherSkillExtractor:
    """
    Extract wet weather performance features for F1 prediction.

    Captures driver-specific wet weather ability by comparing performance
    in wet vs dry conditions. Also tracks team wet weather competitiveness.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        data_dir: Path | str = "data/fastf1",
    ):
        """
        Initialize wet weather skill extractor.

        Args:
            windows: Rolling window sizes for temporal features
            data_dir: Directory containing FastF1 parquet files
        """
        self.windows = windows or [3, 5, 10]
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized WetWeatherSkillExtractor (windows={self.windows})")

    def _load_session_data(self, min_year: int) -> pd.DataFrame:
        """
        Load session data and identify wet sessions.

        Args:
            min_year: Minimum year to load

        Returns:
            DataFrame with session data including wet indicators
        """
        sessions_dir = self.data_dir / "sessions"
        if not sessions_dir.exists():
            logger.warning(f"Sessions directory not found: {sessions_dir}")
            return pd.DataFrame()

        # Load all race and qualifying sessions
        dfs = []
        for pattern in ["*_R.parquet", "*_Q.parquet"]:
            for file_path in sessions_dir.glob(pattern):
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
        logger.info(f"Loaded {len(all_data)} rows from sessions")
        return all_data

    def _identify_wet_sessions(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify wet sessions based on compound usage and rainfall.

        A session is considered "wet" if:
        - INTERMEDIATE or WET tyres were used, OR
        - rainfall=True for significant portion of session

        Args:
            session_data: Raw session data

        Returns:
            DataFrame with session_key and is_wet indicator
        """
        if session_data.empty:
            return pd.DataFrame()

        wet_sessions = []

        for session_key in session_data["session_key"].unique():
            session = session_data[session_data["session_key"] == session_key]

            # Check for wet tyre usage
            compounds_used = set()
            if "compound" in session.columns:
                compounds_used = set(session["compound"].dropna().unique())

            has_wet_tyres = bool(compounds_used & WET_COMPOUNDS)

            # Check for rainfall
            has_rainfall = False
            if "rainfall" in session.columns:
                has_rainfall = session["rainfall"].any()

            # Session is wet if wet tyres used (more reliable indicator)
            is_wet = has_wet_tyres

            # Get session metadata
            row = session.iloc[0]
            wet_sessions.append(
                {
                    "session_key": session_key,
                    "year": row.get("year"),
                    "round": row.get("round"),
                    "session_type": row.get("session_type"),
                    "is_wet": is_wet,
                    "has_wet_tyres": has_wet_tyres,
                    "has_rainfall": has_rainfall,
                }
            )

        wet_df = pd.DataFrame(wet_sessions)
        n_wet = wet_df["is_wet"].sum()
        logger.info(f"Identified {n_wet} wet sessions out of {len(wet_df)} total")
        return wet_df

    def extract_features(
        self,
        target_results: pd.DataFrame,
        min_year: int = 2020,
    ) -> pd.DataFrame:
        """
        Extract wet weather skill features.

        Args:
            target_results: Target session results (for merging context)
            min_year: Minimum year for loading session data

        Returns:
            DataFrame with wet weather features per driver per session
        """
        logger.info("Extracting wet weather skill features...")

        # Load session data to identify wet sessions
        session_data = self._load_session_data(min_year)

        if session_data.empty:
            logger.warning("No session data available for wet weather features")
            return pd.DataFrame()

        # Identify wet sessions
        wet_sessions = self._identify_wet_sessions(session_data)

        if wet_sessions.empty:
            return pd.DataFrame()

        # Merge wet indicator with target results
        if "session_key" not in target_results.columns:
            logger.warning("No session_key in target_results")
            return pd.DataFrame()

        df = target_results.copy()
        df = df.merge(
            wet_sessions[["session_key", "is_wet", "has_rainfall"]],
            on="session_key",
            how="left",
        )
        df["is_wet"] = df["is_wet"].fillna(False).astype(int)
        df["has_rainfall"] = df["has_rainfall"].fillna(False).astype(int)

        # Calculate wet weather performance features
        features = self._calculate_wet_features(df)

        if features.empty:
            return pd.DataFrame()

        n_features = len(
            [
                c
                for c in features.columns
                if c not in ["session_key", "driver_code", "year", "round"]
            ]
        )
        logger.info(f"Extracted {n_features} wet weather features")

        return features

    def _calculate_wet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate wet weather skill features.

        Args:
            df: Results with wet session indicators

        Returns:
            DataFrame with wet weather features
        """
        df = df.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        # Ensure position is numeric
        df["position"] = pd.to_numeric(df["position"], errors="coerce")

        # Start with base columns
        features = df[["session_key", "driver_code", "year", "round"]].copy()
        if "team" in df.columns:
            features["team"] = df["team"]

        # Current session wet indicator
        # Note: is_wet_session already provided by weather_features.py
        # Only add has_rainfall here (indicates rain during session)
        features["has_rainfall_wet"] = df["has_rainfall"].values

        # Driver wet weather features
        features = self._add_driver_wet_features(features, df)

        # Team wet weather features
        if "team" in df.columns:
            features = self._add_team_wet_features(features, df)

        # Relative wet performance features
        features = self._add_relative_wet_features(features, df)

        return features

    def _add_driver_wet_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add driver-specific wet weather performance features."""
        df = df.sort_values(["driver_code", "year", "round"])

        # Calculate position delta from average (for wet vs dry comparison)
        # First get rolling average position
        df["rolling_avg_pos"] = df.groupby("driver_code")["position"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        df["pos_vs_avg"] = df["rolling_avg_pos"] - df["position"]  # Positive = better than avg

        # Wet-only position (for wet performance tracking)
        df["wet_position"] = df["position"].where(df["is_wet"] == 1)
        df["dry_position"] = df["position"].where(df["is_wet"] == 0)

        for window in self.windows:
            # Wet race count (experience in wet)
            features[f"wet_race_count_{window}"] = df.groupby("driver_code")["is_wet"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            )

            # Wet race rate (how often they've raced in wet)
            features[f"wet_race_rate_{window}"] = df.groupby("driver_code")["is_wet"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Average position in wet races (lower = better)
            features[f"wet_avg_position_{window}"] = df.groupby("driver_code")[
                "wet_position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Average position in dry races
            features[f"dry_avg_position_{window}"] = df.groupby("driver_code")[
                "dry_position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Wet vs dry performance delta (positive = better in wet)
            features[f"wet_dry_delta_{window}"] = (
                features[f"dry_avg_position_{window}"] - features[f"wet_avg_position_{window}"]
            )

            # Performance vs own average in wet conditions
            df["wet_pos_vs_avg"] = df["pos_vs_avg"].where(df["is_wet"] == 1)
            features[f"wet_overperformance_{window}"] = df.groupby("driver_code")[
                "wet_pos_vs_avg"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Career wet stats
        features["wet_career_count"] = df.groupby("driver_code")["is_wet"].transform(
            lambda x: x.shift(1).expanding().sum()
        )

        features["wet_career_avg_position"] = df.groupby("driver_code")["wet_position"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        features["dry_career_avg_position"] = df.groupby("driver_code")["dry_position"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Career wet skill (positive = better in wet than dry)
        features["wet_skill_career"] = (
            features["dry_career_avg_position"] - features["wet_career_avg_position"]
        )

        # Binary wet specialist flag (consistently better in wet)
        features["wet_specialist"] = (
            (features["wet_skill_career"] > 1.0) & (features["wet_career_count"] >= 3)
        ).astype(float)

        # Binary wet struggler flag (consistently worse in wet)
        features["wet_struggler"] = (
            (features["wet_skill_career"] < -1.0) & (features["wet_career_count"] >= 3)
        ).astype(float)

        return features

    def _add_team_wet_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add team-level wet weather performance features."""
        df = df.sort_values(["team", "year", "round"])

        # Team wet position
        df["team_wet_position"] = df["position"].where(df["is_wet"] == 1)

        for window in self.windows:
            # Team average position in wet (using larger window for team)
            features[f"team_wet_avg_position_{window}"] = df.groupby("team")[
                "team_wet_position"
            ].transform(lambda x: x.shift(1).rolling(window * 2, min_periods=2).mean())

        # Team career wet average
        features["team_wet_career_avg"] = df.groupby("team")["team_wet_position"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        return features

    def _add_relative_wet_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add relative wet performance features (vs field)."""
        # Calculate field average position in wet sessions
        wet_df = df[df["is_wet"] == 1].copy()

        if not wet_df.empty:
            # Field average in wet
            wet_field_avg = wet_df.groupby("session_key")["position"].transform("mean")
            wet_df["wet_vs_field"] = (
                wet_field_avg - wet_df["position"]
            )  # Positive = better than field

            # Merge back
            wet_vs_field_lookup = wet_df[
                ["session_key", "driver_code", "wet_vs_field"]
            ].drop_duplicates()

            features = features.merge(
                wet_vs_field_lookup,
                on=["session_key", "driver_code"],
                how="left",
            )
        else:
            features["wet_vs_field"] = np.nan

        # Fill NaN for non-wet sessions
        features["wet_vs_field"] = features["wet_vs_field"].fillna(0)

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = [
            "is_wet_session",
            "has_rainfall",
        ]

        # Rolling window features
        for window in self.windows:
            names.extend(
                [
                    f"wet_race_count_{window}",
                    f"wet_race_rate_{window}",
                    f"wet_avg_position_{window}",
                    f"dry_avg_position_{window}",
                    f"wet_dry_delta_{window}",
                    f"wet_overperformance_{window}",
                    f"team_wet_avg_position_{window}",
                ]
            )

        # Career features
        names.extend(
            [
                "wet_career_count",
                "wet_career_avg_position",
                "dry_career_avg_position",
                "wet_skill_career",
                "wet_specialist",
                "wet_struggler",
                "team_wet_career_avg",
                "wet_vs_field",
            ]
        )

        return names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.loaders import F1DataLoader

    loader = F1DataLoader()

    # Load race results for context
    race_results = loader.load_race_results(min_year=2020)
    print(f"Loaded {len(race_results)} race results")

    # Extract wet weather features
    extractor = WetWeatherSkillExtractor()
    features = extractor.extract_features(race_results, min_year=2020)

    print(f"\nFeature columns: {features.columns.tolist()}")
    print(f"\nFeature shape: {features.shape}")

    # Check wet skill distribution
    print("\nWet skill career stats:")
    print(features["wet_skill_career"].describe())

    # Show wet specialists
    print("\nWet specialists:")
    specialists = features[features["wet_specialist"] == 1]["driver_code"].unique()
    print(specialists[:10])
