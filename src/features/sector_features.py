"""
Sector time feature extraction.

Extracts features based on sector times (S1, S2, S3) for qualifying prediction.
All features use temporal safety patterns (shift before rolling) to prevent data leakage.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class SectorFeatureExtractor:
    """Extracts sector time-based features from qualifying lap data."""

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize sector feature extractor.

        Args:
            windows: Rolling window sizes for aggregations (default: [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10]

    def extract_features(self, quali_sessions: pd.DataFrame, min_laps: int = 3) -> pd.DataFrame:
        """
        Extract sector-based features from qualifying session data.

        Args:
            quali_sessions: DataFrame with qualifying lap data including sector times
            min_laps: Minimum laps required for feature calculation

        Returns:
            DataFrame with sector features per driver per session
        """
        if quali_sessions.empty:
            return pd.DataFrame()

        # First, compute per-session best sectors for each driver
        session_features = self._compute_session_sector_stats(quali_sessions)

        if session_features.empty:
            return pd.DataFrame()

        # Then compute rolling features across sessions (with temporal shift)
        rolling_features = self._compute_rolling_sector_features(session_features)

        return rolling_features

    def _compute_session_sector_stats(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """Compute sector statistics for each driver per qualifying session."""
        required_cols = [
            "session_key",
            "driver_code",
            "sector1_time_ms",
            "sector2_time_ms",
            "sector3_time_ms",
            "lap_time_ms",
        ]

        # Check for required columns
        missing = [c for c in required_cols if c not in quali_sessions.columns]
        if missing:
            logger.warning(f"Missing columns for sector features: {missing}")
            return pd.DataFrame()

        # Filter to valid laps with sector times
        valid_laps = quali_sessions[
            quali_sessions["sector1_time_ms"].notna()
            & quali_sessions["sector2_time_ms"].notna()
            & quali_sessions["sector3_time_ms"].notna()
            & quali_sessions["lap_time_ms"].notna()
            & ~quali_sessions.get("deleted", False)
        ].copy()

        if valid_laps.empty:
            return pd.DataFrame()

        # Group by session and driver
        grouped = valid_laps.groupby(["session_key", "driver_code"])

        session_stats = grouped.agg(
            {
                "year": "first",
                "round": "first",
                "circuit": "first",
                "team": "first",
                "position": "first",
                # Best times
                "sector1_time_ms": "min",
                "sector2_time_ms": "min",
                "sector3_time_ms": "min",
                "lap_time_ms": "min",
            }
        ).reset_index()

        session_stats = session_stats.rename(
            columns={
                "sector1_time_ms": "best_s1_ms",
                "sector2_time_ms": "best_s2_ms",
                "sector3_time_ms": "best_s3_ms",
                "lap_time_ms": "best_lap_ms",
            }
        )

        # Compute theoretical best lap (sum of best sectors)
        session_stats["theoretical_best_ms"] = (
            session_stats["best_s1_ms"] + session_stats["best_s2_ms"] + session_stats["best_s3_ms"]
        )

        # Gap between actual best and theoretical best
        session_stats["theoretical_gap_ms"] = (
            session_stats["best_lap_ms"] - session_stats["theoretical_best_ms"]
        )

        # Compute sector proportions (for balance analysis)
        total_sector_time = (
            session_stats["best_s1_ms"] + session_stats["best_s2_ms"] + session_stats["best_s3_ms"]
        )
        session_stats["s1_proportion"] = session_stats["best_s1_ms"] / total_sector_time
        session_stats["s2_proportion"] = session_stats["best_s2_ms"] / total_sector_time
        session_stats["s3_proportion"] = session_stats["best_s3_ms"] / total_sector_time

        # Add session-level rankings for each sector
        for sector in ["best_s1_ms", "best_s2_ms", "best_s3_ms"]:
            rank_col = sector.replace("best_", "").replace("_ms", "_rank")
            session_stats[rank_col] = session_stats.groupby("session_key")[sector].rank()

        # Calculate sector vs field
        for sector in ["best_s1_ms", "best_s2_ms", "best_s3_ms"]:
            col_name = sector.replace("best_", "").replace("_ms", "_vs_field")
            session_median = session_stats.groupby("session_key")[sector].transform("median")
            session_stats[col_name] = session_median - session_stats[sector]

        return session_stats

    def _compute_rolling_sector_features(self, session_stats: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling features across sessions with temporal shift."""
        if session_stats.empty:
            return pd.DataFrame()

        # Sort by driver and session for proper rolling
        df = session_stats.sort_values(["driver_code", "year", "round"]).copy()

        # Columns to compute rolling stats for
        sector_cols = ["best_s1_ms", "best_s2_ms", "best_s3_ms"]
        rank_cols = ["s1_rank", "s2_rank", "s3_rank"]
        vs_field_cols = ["s1_vs_field", "s2_vs_field", "s3_vs_field"]

        features = df[
            ["session_key", "driver_code", "year", "round", "circuit", "team", "position"]
        ].copy()

        for window in self.windows:
            for col in sector_cols + rank_cols + vs_field_cols + ["theoretical_gap_ms"]:
                if col not in df.columns:
                    continue

                feature_name = f"rolling_{col}_{window}"

                # CRITICAL: shift(1) before rolling to exclude current session
                features[feature_name] = df.groupby("driver_code")[col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Compute sector balance score (std of sector proportions)
        # Lower = more balanced across sectors
        for window in self.windows:
            s1_prop = df.groupby("driver_code")["s1_proportion"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            s2_prop = df.groupby("driver_code")["s2_proportion"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            s3_prop = df.groupby("driver_code")["s3_proportion"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Standard deviation of proportions (0.33 each = perfectly balanced)
            features[f"sector_balance_{window}"] = pd.concat(
                [s1_prop, s2_prop, s3_prop], axis=1
            ).std(axis=1)

        # Identify driver's strongest sector (historical)
        features["strongest_sector"] = self._identify_strongest_sector(df)

        return features

    def _identify_strongest_sector(self, df: pd.DataFrame) -> pd.Series:
        """Identify each driver's historically strongest sector."""
        # Use last 10 sessions to determine strongest sector
        result = pd.Series(index=df.index, dtype="object")

        for driver in df["driver_code"].unique():
            driver_mask = df["driver_code"] == driver
            driver_data = df[driver_mask].copy()

            # Use shifted data to avoid leakage
            s1_avg = driver_data["s1_rank"].shift(1).rolling(10, min_periods=1).mean()
            s2_avg = driver_data["s2_rank"].shift(1).rolling(10, min_periods=1).mean()
            s3_avg = driver_data["s3_rank"].shift(1).rolling(10, min_periods=1).mean()

            # Best (lowest) rank = strongest sector
            strongest = pd.concat(
                [s1_avg.rename("S1"), s2_avg.rename("S2"), s3_avg.rename("S3")], axis=1
            ).idxmin(axis=1)

            result.loc[driver_mask] = strongest.values

        return result

    def extract_circuit_sector_features(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract circuit-specific sector features.

        Args:
            quali_sessions: DataFrame with qualifying lap data

        Returns:
            DataFrame with circuit-specific sector features
        """
        session_stats = self._compute_session_sector_stats(quali_sessions)

        if session_stats.empty:
            return pd.DataFrame()

        # Group by driver and circuit
        circuit_features = (
            session_stats.groupby(["driver_code", "circuit"])
            .agg(
                {
                    "best_s1_ms": "mean",
                    "best_s2_ms": "mean",
                    "best_s3_ms": "mean",
                    "s1_rank": "mean",
                    "s2_rank": "mean",
                    "s3_rank": "mean",
                    "theoretical_gap_ms": "mean",
                    "session_key": "count",
                }
            )
            .reset_index()
        )

        circuit_features = circuit_features.rename(
            columns={
                "session_key": "circuit_appearances",
                "best_s1_ms": "circuit_avg_s1_ms",
                "best_s2_ms": "circuit_avg_s2_ms",
                "best_s3_ms": "circuit_avg_s3_ms",
                "s1_rank": "circuit_avg_s1_rank",
                "s2_rank": "circuit_avg_s2_rank",
                "s3_rank": "circuit_avg_s3_rank",
                "theoretical_gap_ms": "circuit_avg_theoretical_gap",
            }
        )

        return circuit_features
