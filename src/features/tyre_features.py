"""
Tyre compound feature extraction.

Extracts features based on tyre compound preferences and performance.
All features use temporal safety patterns to prevent data leakage.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TyreFeatureExtractor:
    """Extracts tyre-related features from session data."""

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize tyre feature extractor.

        Args:
            windows: Rolling window sizes for aggregations (default: [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10]

    def extract_features(
        self, quali_sessions: pd.DataFrame, practice_sessions: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Extract tyre-based features.

        Args:
            quali_sessions: DataFrame with qualifying lap data
            practice_sessions: Optional DataFrame with practice lap data

        Returns:
            DataFrame with tyre features per driver per session
        """
        if quali_sessions.empty:
            return pd.DataFrame()

        # Compute per-session tyre stats
        session_features = self._compute_session_tyre_stats(quali_sessions)

        if session_features.empty:
            return pd.DataFrame()

        # Add practice tyre data if available
        if practice_sessions is not None and not practice_sessions.empty:
            session_features = self._add_practice_tyre_data(session_features, practice_sessions)

        # Compute rolling features
        rolling_features = self._compute_rolling_tyre_features(session_features)

        return rolling_features

    def _compute_session_tyre_stats(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """Compute tyre statistics for each driver per qualifying session."""
        required_cols = ["session_key", "driver_code"]
        missing = [c for c in required_cols if c not in quali_sessions.columns]
        if missing:
            logger.warning(f"Missing columns for tyre features: {missing}")
            return pd.DataFrame()

        has_compound = "compound" in quali_sessions.columns
        has_tyre_life = "tyre_life" in quali_sessions.columns
        has_fresh = "fresh_tyre" in quali_sessions.columns

        if not has_compound:
            logger.warning("No compound data available for tyre features")
            return pd.DataFrame()

        # Filter to valid laps
        valid_laps = quali_sessions[
            quali_sessions["compound"].notna()
            & (quali_sessions["compound"] != "")
            & quali_sessions["lap_time_ms"].notna()
        ].copy()

        if valid_laps.empty:
            return pd.DataFrame()

        session_stats = []

        for (session_key, driver), group in valid_laps.groupby(["session_key", "driver_code"]):
            stats = {
                "session_key": session_key,
                "driver_code": driver,
                "year": group["year"].iloc[0],
                "round": group["round"].iloc[0],
                "circuit": group.get("circuit", pd.Series([""])).iloc[0],
                "team": group.get("team", pd.Series([""])).iloc[0],
                "position": group["position"].iloc[0] if "position" in group.columns else None,
            }

            # Compound usage
            compounds_used = group["compound"].unique()
            stats["compounds_used"] = len(compounds_used)
            stats["used_soft"] = "SOFT" in compounds_used
            stats["used_medium"] = "MEDIUM" in compounds_used
            stats["used_hard"] = "HARD" in compounds_used

            # Best lap per compound
            for compound in ["SOFT", "MEDIUM", "HARD"]:
                compound_laps = group[group["compound"] == compound]
                if not compound_laps.empty:
                    stats[f"best_{compound.lower()}_ms"] = compound_laps["lap_time_ms"].min()
                    stats[f"{compound.lower()}_laps"] = len(compound_laps)
                else:
                    stats[f"best_{compound.lower()}_ms"] = None
                    stats[f"{compound.lower()}_laps"] = 0

            # Compound performance delta (soft vs medium advantage)
            if stats.get("best_soft_ms") and stats.get("best_medium_ms"):
                stats["soft_vs_medium_ms"] = stats["best_medium_ms"] - stats["best_soft_ms"]

            # Fresh tyre advantage
            if has_fresh and has_tyre_life:
                fresh_laps = group[group["fresh_tyre"] == True]  # noqa: E712
                used_laps = group[group["fresh_tyre"] == False]  # noqa: E712

                if not fresh_laps.empty:
                    stats["best_fresh_ms"] = fresh_laps["lap_time_ms"].min()
                if not used_laps.empty:
                    stats["best_used_ms"] = used_laps["lap_time_ms"].min()

                if stats.get("best_fresh_ms") and stats.get("best_used_ms"):
                    stats["fresh_vs_used_ms"] = stats["best_used_ms"] - stats["best_fresh_ms"]

            # Average tyre life on best laps
            if has_tyre_life:
                # Tyre life on best lap
                best_lap_idx = group["lap_time_ms"].idxmin()
                stats["best_lap_tyre_life"] = group.loc[best_lap_idx, "tyre_life"]

            session_stats.append(stats)

        return pd.DataFrame(session_stats)

    def _add_practice_tyre_data(
        self, session_features: pd.DataFrame, practice_sessions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add practice tyre data to session features."""
        if "compound" not in practice_sessions.columns:
            return session_features

        # Get practice tyre usage per weekend
        practice_tyres = (
            practice_sessions.groupby(["year", "round", "driver_code"])
            .agg(
                {
                    "compound": lambda x: x.value_counts().to_dict(),
                }
            )
            .reset_index()
        )

        # Count laps per compound
        for idx, row in practice_tyres.iterrows():
            compound_counts = row["compound"]
            practice_tyres.loc[idx, "practice_soft_laps"] = compound_counts.get("SOFT", 0)
            practice_tyres.loc[idx, "practice_medium_laps"] = compound_counts.get("MEDIUM", 0)
            practice_tyres.loc[idx, "practice_hard_laps"] = compound_counts.get("HARD", 0)

        practice_tyres = practice_tyres.drop(columns=["compound"])

        # Merge with session features
        session_features = session_features.merge(
            practice_tyres,
            on=["year", "round", "driver_code"],
            how="left",
        )

        return session_features

    def _compute_rolling_tyre_features(self, session_stats: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling tyre features with temporal shift."""
        if session_stats.empty:
            return pd.DataFrame()

        df = session_stats.sort_values(["driver_code", "year", "round"]).copy()

        features = df[
            ["session_key", "driver_code", "year", "round", "circuit", "team", "position"]
        ].copy()

        # Current session tyre data (OK to use)
        for col in ["used_soft", "used_medium", "compounds_used"]:
            if col in df.columns:
                features[f"current_{col}"] = df[col]

        # Rolling features (from PREVIOUS sessions - shift(1))
        for window in self.windows:
            # Soft compound affinity (rate of using soft)
            if "used_soft" in df.columns:
                features[f"soft_usage_rate_{window}"] = df.groupby("driver_code")[
                    "used_soft"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Soft vs medium advantage
            if "soft_vs_medium_ms" in df.columns:
                features[f"avg_soft_advantage_{window}"] = df.groupby("driver_code")[
                    "soft_vs_medium_ms"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Fresh tyre advantage
            if "fresh_vs_used_ms" in df.columns:
                features[f"avg_fresh_advantage_{window}"] = df.groupby("driver_code")[
                    "fresh_vs_used_ms"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Typical tyre life on best laps
            if "best_lap_tyre_life" in df.columns:
                features[f"avg_best_lap_tyre_life_{window}"] = df.groupby("driver_code")[
                    "best_lap_tyre_life"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        return features

    def extract_team_tyre_features(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract team-level tyre features.

        Args:
            quali_sessions: DataFrame with qualifying data

        Returns:
            DataFrame with team tyre features
        """
        session_stats = self._compute_session_tyre_stats(quali_sessions)

        if session_stats.empty:
            return pd.DataFrame()

        # Group by session and team
        team_stats = (
            session_stats.groupby(["session_key", "team"])
            .agg(
                {
                    "year": "first",
                    "round": "first",
                    "soft_vs_medium_ms": "mean",
                    "fresh_vs_used_ms": "mean",
                    "used_soft": "mean",
                }
            )
            .reset_index()
        )

        return team_stats

    def extract_circuit_tyre_features(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract circuit-specific tyre features.

        Args:
            quali_sessions: DataFrame with qualifying data

        Returns:
            DataFrame with circuit tyre features
        """
        session_stats = self._compute_session_tyre_stats(quali_sessions)

        if session_stats.empty:
            return pd.DataFrame()

        # Group by driver and circuit
        circuit_features = (
            session_stats.groupby(["driver_code", "circuit"])
            .agg(
                {
                    "soft_vs_medium_ms": "mean",
                    "used_soft": "mean",
                    "session_key": "count",
                }
            )
            .reset_index()
        )

        circuit_features = circuit_features.rename(
            columns={
                "session_key": "circuit_appearances",
                "soft_vs_medium_ms": "circuit_soft_advantage",
                "used_soft": "circuit_soft_rate",
            }
        )

        return circuit_features
