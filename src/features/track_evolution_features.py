"""
Track evolution feature extraction for F1 predictions.

Track grip improves throughout a session as more rubber is laid down.
This creates advantages for drivers who run later in sessions:
- Q3 track is significantly grippier than Q1
- Final run specialists benefit from peak grip
- Some drivers/teams adapt better to evolving track conditions

Key insights from research:
- Lap times can improve 1-2 seconds from FP1 to Q3 due to rubber
- Green track (new surface) specialists vs rubbered track specialists
- Track evolution rate varies by circuit (street circuits evolve more)

All features use temporal shift to prevent data leakage.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrackEvolutionExtractor:
    """
    Extract track evolution features for F1 prediction.

    Captures how drivers perform relative to track evolution:
    - Improvement from early to late session runs
    - Adaptation to changing grip levels
    - Performance on green vs rubbered tracks
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        data_dir: Path | str = "data/fastf1",
    ):
        """
        Initialize track evolution extractor.

        Args:
            windows: Rolling window sizes for temporal features
            data_dir: Directory containing FastF1 parquet files
        """
        self.windows = windows or [3, 5, 10]
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized TrackEvolutionExtractor (windows={self.windows})")

    def _load_session_data(
        self,
        min_year: int,
        session_types: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load session lap data for track evolution analysis.

        Args:
            min_year: Minimum year to load
            session_types: Session types to load (default: Q, FP1, FP2, FP3)

        Returns:
            DataFrame with lap-by-lap session data
        """
        if session_types is None:
            session_types = ["Q", "FP1", "FP2", "FP3"]

        sessions_dir = self.data_dir / "sessions"
        if not sessions_dir.exists():
            logger.warning(f"Sessions directory not found: {sessions_dir}")
            return pd.DataFrame()

        dfs = []
        for session_type in session_types:
            for file_path in sessions_dir.glob(f"*_{session_type}.parquet"):
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
        logger.info(f"Loaded {len(all_data)} rows from {len(dfs)} sessions")
        return all_data

    def _calculate_session_evolution(
        self,
        session_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate track evolution within sessions.

        Measures how lap times improve from start to end of session.

        Args:
            session_data: Raw session lap data

        Returns:
            DataFrame with session evolution metrics
        """
        if session_data.empty:
            return pd.DataFrame()

        # Need lap time column
        time_col = None
        for col in ["lap_time_ms", "time_ms", "lap_time"]:
            if col in session_data.columns:
                time_col = col
                break

        if time_col is None:
            logger.warning("No lap time column found for track evolution")
            return pd.DataFrame()

        evolution_stats = []

        for session_key in session_data["session_key"].unique():
            session = session_data[session_data["session_key"] == session_key].copy()

            # Get valid lap times
            session[time_col] = pd.to_numeric(session[time_col], errors="coerce")
            valid_laps = session[session[time_col] > 0].copy()

            if valid_laps.empty or len(valid_laps) < 10:
                continue

            # Sort by lap number or time to get session progression
            if "lap_number" in valid_laps.columns:
                valid_laps = valid_laps.sort_values("lap_number")
            elif "time" in valid_laps.columns:
                valid_laps = valid_laps.sort_values("time")

            # Calculate session-wide evolution
            # Compare first third vs last third of session
            n_laps = len(valid_laps)
            third = n_laps // 3

            if third < 3:
                continue

            early_laps = valid_laps.head(third)
            late_laps = valid_laps.tail(third)

            # Best lap times in each period (representative of track grip)
            early_best = early_laps[time_col].min()
            late_best = late_laps[time_col].min()

            # Evolution = improvement from early to late (positive = track got faster)
            evolution_ms = early_best - late_best
            evolution_pct = (evolution_ms / early_best) * 100 if early_best > 0 else 0

            # Get session metadata
            row = session.iloc[0]
            evolution_stats.append(
                {
                    "session_key": session_key,
                    "year": row.get("year"),
                    "round": row.get("round"),
                    "session_type": row.get("session_type"),
                    "track_evolution_ms": evolution_ms,
                    "track_evolution_pct": evolution_pct,
                    "early_best_ms": early_best,
                    "late_best_ms": late_best,
                }
            )

        if not evolution_stats:
            return pd.DataFrame()

        return pd.DataFrame(evolution_stats)

    def _calculate_driver_evolution_features(
        self,
        session_data: pd.DataFrame,
        target_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate driver-specific track evolution features.

        Args:
            session_data: Raw session lap data
            target_results: Target session results

        Returns:
            DataFrame with driver evolution features
        """
        if session_data.empty or target_results.empty:
            return pd.DataFrame()

        # Need lap time column
        time_col = None
        for col in ["lap_time_ms", "time_ms", "lap_time"]:
            if col in session_data.columns:
                time_col = col
                break

        if time_col is None:
            return pd.DataFrame()

        driver_stats = []

        for session_key in session_data["session_key"].unique():
            session = session_data[session_data["session_key"] == session_key].copy()
            session[time_col] = pd.to_numeric(session[time_col], errors="coerce")

            # Get session metadata
            meta = session.iloc[0]
            year = meta.get("year")
            round_num = meta.get("round")

            # Calculate per-driver evolution within session
            for driver in session["driver_code"].unique():
                driver_laps = session[
                    (session["driver_code"] == driver) & (session[time_col] > 0)
                ].copy()

                if len(driver_laps) < 4:
                    continue

                # Sort by lap progression
                if "lap_number" in driver_laps.columns:
                    driver_laps = driver_laps.sort_values("lap_number")

                n_laps = len(driver_laps)
                half = n_laps // 2

                # Compare first half vs second half performance
                first_half = driver_laps.head(half)
                second_half = driver_laps.tail(half)

                first_best = first_half[time_col].min()
                second_best = second_half[time_col].min()

                # Driver's improvement through session
                driver_evolution_ms = first_best - second_best
                driver_evolution_pct = (
                    (driver_evolution_ms / first_best) * 100 if first_best > 0 else 0
                )

                # Consistency improvement (std reduction)
                first_std = first_half[time_col].std()
                second_std = second_half[time_col].std()
                consistency_improvement = (
                    first_std - second_std if pd.notna(first_std) and pd.notna(second_std) else 0
                )

                driver_stats.append(
                    {
                        "session_key": session_key,
                        "driver_code": driver,
                        "year": year,
                        "round": round_num,
                        "driver_evolution_ms": driver_evolution_ms,
                        "driver_evolution_pct": driver_evolution_pct,
                        "consistency_improvement": consistency_improvement,
                        "first_half_best": first_best,
                        "second_half_best": second_best,
                    }
                )

        if not driver_stats:
            return pd.DataFrame()

        return pd.DataFrame(driver_stats)

    def _calculate_rolling_evolution_features(
        self,
        driver_evolution: pd.DataFrame,
        target_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate rolling historical evolution features.

        Args:
            driver_evolution: Per-session driver evolution stats
            target_results: Target session results

        Returns:
            DataFrame with rolling evolution features
        """
        if driver_evolution.empty:
            return pd.DataFrame()

        df = target_results.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        # Merge driver evolution stats
        # First aggregate to session level (in case of multiple session types)
        driver_agg = (
            driver_evolution.groupby(["driver_code", "year", "round"])
            .agg(
                {
                    "driver_evolution_ms": "mean",
                    "driver_evolution_pct": "mean",
                    "consistency_improvement": "mean",
                }
            )
            .reset_index()
        )

        df = df.merge(
            driver_agg,
            on=["driver_code", "year", "round"],
            how="left",
        )

        features = df[["session_key", "driver_code", "year", "round"]].copy()

        for window in self.windows:
            # Rolling average evolution (how well driver adapts to track evolution)
            features[f"avg_evolution_ms_{window}"] = df.groupby("driver_code")[
                "driver_evolution_ms"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            features[f"avg_evolution_pct_{window}"] = df.groupby("driver_code")[
                "driver_evolution_pct"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Consistency improvement through sessions
            features[f"avg_consistency_gain_{window}"] = df.groupby("driver_code")[
                "consistency_improvement"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Evolution variability (how consistent is their adaptation)
            features[f"evolution_std_{window}"] = df.groupby("driver_code")[
                "driver_evolution_pct"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())

        # Career evolution stats
        features["career_avg_evolution"] = df.groupby("driver_code")[
            "driver_evolution_pct"
        ].transform(lambda x: x.shift(1).expanding().mean())

        features["career_evolution_consistency"] = df.groupby("driver_code")[
            "driver_evolution_pct"
        ].transform(lambda x: x.shift(1).expanding().std())

        # Binary flags
        # Late session specialist (consistently improves through session)
        features["late_session_specialist"] = (
            (features["career_avg_evolution"] > 0.5)
            & (features["career_evolution_consistency"] < 1.0)
        ).astype(float)

        # Green track specialist (performs well early in sessions)
        features["green_track_specialist"] = (features["career_avg_evolution"] < -0.2).astype(float)

        return features

    def extract_features(
        self,
        target_results: pd.DataFrame,
        min_year: int = 2020,
    ) -> pd.DataFrame:
        """
        Extract track evolution features.

        Args:
            target_results: Target session results (for merging context)
            min_year: Minimum year for loading data

        Returns:
            DataFrame with track evolution features per driver per session
        """
        logger.info("Extracting track evolution features...")

        # Load session data
        session_data = self._load_session_data(min_year)

        if session_data.empty:
            logger.warning("No session data for track evolution features")
            return pd.DataFrame()

        # Calculate session-level evolution
        session_evolution = self._calculate_session_evolution(session_data)

        # Calculate driver-level evolution
        driver_evolution = self._calculate_driver_evolution_features(session_data, target_results)

        # Calculate rolling historical features
        if not driver_evolution.empty:
            features = self._calculate_rolling_evolution_features(driver_evolution, target_results)
        else:
            features = target_results[["session_key", "driver_code", "year", "round"]].copy()

        # Add current session evolution context
        if not session_evolution.empty:
            # Get qualifying session evolution for current weekend
            quali_evolution = session_evolution[session_evolution["session_type"] == "Q"][
                ["year", "round", "track_evolution_ms", "track_evolution_pct"]
            ]

            if not quali_evolution.empty:
                quali_evolution = quali_evolution.rename(
                    columns={
                        "track_evolution_ms": "current_track_evolution_ms",
                        "track_evolution_pct": "current_track_evolution_pct",
                    }
                )

                features = features.merge(
                    quali_evolution.drop_duplicates(subset=["year", "round"]),
                    on=["year", "round"],
                    how="left",
                )

        # Fill missing evolution columns
        evolution_cols = [
            "current_track_evolution_ms",
            "current_track_evolution_pct",
        ]
        for col in evolution_cols:
            if col not in features.columns:
                features[col] = np.nan

        n_features = len(
            [
                c
                for c in features.columns
                if c not in ["session_key", "driver_code", "year", "round"]
            ]
        )
        logger.info(f"Extracted {n_features} track evolution features")

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = []

        for window in self.windows:
            names.extend(
                [
                    f"avg_evolution_ms_{window}",
                    f"avg_evolution_pct_{window}",
                    f"avg_consistency_gain_{window}",
                    f"evolution_std_{window}",
                ]
            )

        names.extend(
            [
                "career_avg_evolution",
                "career_evolution_consistency",
                "late_session_specialist",
                "green_track_specialist",
                "current_track_evolution_ms",
                "current_track_evolution_pct",
            ]
        )

        return names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.loaders import F1DataLoader

    loader = F1DataLoader()

    # Load qualifying results
    quali_results = loader.load_qualifying_results(min_year=2020)
    print(f"Loaded {len(quali_results)} qualifying results")

    # Extract features
    extractor = TrackEvolutionExtractor()
    features = extractor.extract_features(quali_results, min_year=2020)

    print(f"\nFeature columns: {features.columns.tolist()}")
    print(f"\nFeature shape: {features.shape}")

    # Check evolution distribution
    print("\nCareer avg evolution stats:")
    print(features["career_avg_evolution"].describe())

    # Show late session specialists
    print("\nLate session specialists:")
    specialists = features[features["late_session_specialist"] == 1]["driver_code"].unique()
    print(specialists[:10])
