"""
Base feature pipeline for F1 predictions.

Provides abstract base class for session-type-specific feature pipelines.
Each subclass handles temporal constraints specific to its prediction target.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.data.loaders import F1DataLoader
from src.features.circuit_features import CircuitFeatureExtractor
from src.features.circuit_overtaking_features import CircuitOvertakingExtractor
from src.features.driver_circuit_features import DriverCircuitInteractionExtractor
from src.features.elo_features import ELOFeatureExtractor
from src.features.first_lap_features import FirstLapFeatureExtractor
from src.features.momentum_features import MomentumFeatureExtractor
from src.features.relative_features import RelativeFeatureExtractor
from src.features.reliability_features import ReliabilityFeatureExtractor
from src.features.sector_features import SectorFeatureExtractor
from src.features.sprint_features import SprintFeatureExtractor
from src.features.track_evolution_features import TrackEvolutionExtractor
from src.features.tyre_features import TyreFeatureExtractor
from src.features.weather_features import WeatherFeatureExtractor
from src.features.wet_weather_skill_features import WetWeatherSkillExtractor

logger = logging.getLogger(__name__)


class BaseFeaturePipeline(ABC):
    """
    Abstract base class for session-type-specific feature pipelines.

    Each subclass implements temporal constraints specific to its prediction target:
    - Qualifying: FP1, FP2, FP3 available
    - Sprint Qualifying: FP1 only (FP2/FP3 happen after SQ)
    - Sprint Race: FP1 + SQ grid (FP2/FP3/Q happen after S)
    - Race: All sessions available (FP1-3, Q, SQ, S)
    """

    # Session type identifier (Q, SQ, S, R)
    session_type: str = ""

    def __init__(self, data_dir: Path | str = "data/fastf1"):
        """
        Initialize the feature pipeline.

        Args:
            data_dir: Directory containing FastF1 parquet files
        """
        self.data_dir = Path(data_dir)
        self.loader = F1DataLoader(self.data_dir)

        # Initialize shared extractors (used by all pipelines)
        self.momentum_extractor = MomentumFeatureExtractor()
        self.relative_extractor = RelativeFeatureExtractor()
        self.weather_extractor = WeatherFeatureExtractor()
        self.circuit_extractor = CircuitFeatureExtractor()
        self.sprint_extractor = SprintFeatureExtractor()
        self.sector_extractor = SectorFeatureExtractor()
        self.tyre_extractor = TyreFeatureExtractor()
        self.driver_circuit_extractor = DriverCircuitInteractionExtractor()
        self.elo_extractor = ELOFeatureExtractor()
        self.reliability_extractor = ReliabilityFeatureExtractor()
        self.first_lap_extractor = FirstLapFeatureExtractor(data_dir=data_dir)
        self.wet_weather_extractor = WetWeatherSkillExtractor(data_dir=data_dir)
        self.circuit_overtaking_extractor = CircuitOvertakingExtractor(data_dir=data_dir)
        self.track_evolution_extractor = TrackEvolutionExtractor(data_dir=data_dir)

        logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(session_type={self.session_type}, data_dir={data_dir})"
        )

    @abstractmethod
    def get_available_practice_sessions(self) -> list[str]:
        """
        Return which FP sessions are available for this prediction type.

        Returns:
            List of session type strings, e.g., ["FP1", "FP2", "FP3"]
        """
        pass

    @abstractmethod
    def get_available_current_weekend_sessions(self) -> list[str]:
        """
        Return which session results from current weekend can be used as features.

        For example, race prediction can use qualifying grid position.

        Returns:
            List of session type strings, e.g., ["Q", "SQ", "S"]
        """
        pass

    @abstractmethod
    def _load_target_results(self, min_year: int) -> pd.DataFrame:
        """
        Load the results we are trying to predict.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with target session results
        """
        pass

    @abstractmethod
    def _get_session_specific_features(
        self,
        df: pd.DataFrame,
        min_year: int,
        windows: list[int],
    ) -> pd.DataFrame:
        """
        Add session-specific features.

        Each subclass implements features unique to its prediction target,
        e.g., Q1/Q2/Q3 progression for qualifying, grid position for race.

        Args:
            df: Base dataframe to add features to
            min_year: Minimum year for data loading
            windows: Rolling window sizes

        Returns:
            DataFrame with session-specific features added
        """
        pass

    def build_features(
        self,
        min_year: int = 2020,
        up_to_race: tuple[int, int] | None = None,
        windows: list[int] | None = None,
        for_ranking: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Build comprehensive feature set for model training.

        Respects temporal constraints defined by subclass methods.

        Args:
            min_year: Minimum year to include
            up_to_race: Optional (year, round) tuple to filter data up to but not including
            windows: Rolling window sizes for temporal features (default: [3, 5, 10])
            for_ranking: If True, returns position as target; if False, returns is_top3

        Returns:
            Tuple of (features DataFrame, target Series, metadata DataFrame)
        """
        if windows is None:
            windows = [3, 5, 10]

        logger.info(
            f"Building {self.session_type} features (min_year={min_year}, windows={windows})..."
        )

        # Load target session results
        target_results = self._load_target_results(min_year=min_year)
        if target_results.empty:
            logger.warning(f"No {self.session_type} results found")
            return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

        # Filter to sessions before target race if specified
        if up_to_race:
            target_year, target_round = up_to_race
            original_count = len(target_results)
            target_results = target_results[
                (target_results["year"] < target_year)
                | ((target_results["year"] == target_year) & (target_results["round"] < target_round))
            ]
            logger.info(
                f"Filtered to {len(target_results)} sessions before "
                f"{target_year} R{target_round} (from {original_count})"
            )

        # Load common data
        events = self.loader.load_events(min_year=min_year)

        # Filter events if up_to_race specified
        if up_to_race:
            target_year, target_round = up_to_race
            events = events[
                (events["year"] < target_year)
                | ((events["year"] == target_year) & (events["round"] < target_round))
            ]

        # Build base dataframe from target results
        df = self._build_base_dataframe(target_results, events)

        # Add shared temporal features (position-based rolling averages)
        df = self._add_temporal_features(df, windows)

        # Add team performance features
        df = self._add_team_features(df, target_results)

        # Add circuit-specific features
        df = self._add_circuit_features(df, target_results)

        # Add driver-circuit interaction features
        df = self._add_driver_circuit_features(df, target_results)

        # Add historical momentum features
        df = self._add_momentum_features(df, target_results)

        # Add historical relative features
        df = self._add_relative_features(df, target_results)

        # Add ELO rating features
        df = self._add_elo_features(df, target_results)

        # Add reliability/DNF features (from race history)
        df = self._add_reliability_features(df, min_year)

        # Add first lap performance features (from race history)
        df = self._add_first_lap_features(df, target_results, min_year)

        # Add wet weather skill features (from race history)
        df = self._add_wet_weather_features(df, target_results, min_year)

        # Add circuit overtaking difficulty features (from race history)
        df = self._add_circuit_overtaking_features(df, target_results, min_year)

        # Add track evolution features (grip improvement through sessions)
        df = self._add_track_evolution_features(df, target_results, min_year)

        # Add session-specific features (implemented by subclass)
        df = self._get_session_specific_features(df, min_year, windows)

        # Prepare final feature matrix
        df = df.sort_values(["year", "round", "position"])
        df = self._convert_to_numeric(df)
        df = df.fillna(0)

        # Get feature columns
        feature_cols = self._get_feature_columns(windows)
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].copy()

        # Target variable
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
        if for_ranking:
            y = df["position"].copy()
        else:
            y = (df["position"] <= 3).astype(int)

        # Metadata for analysis
        meta_cols = ["session_key", "driver_code", "team", "year", "round", "circuit", "position"]
        meta_cols = [c for c in meta_cols if c in df.columns]
        meta = df[meta_cols].copy()

        logger.info(f"Built {len(X)} samples with {len(feature_cols)} features")
        return X, y, meta

    def _build_base_dataframe(
        self,
        target_results: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build base dataframe from target results."""
        df = target_results.copy()

        # Add event date if available
        if not events.empty and "event_date" in events.columns:
            events_lookup = events[["year", "round", "event_date"]].drop_duplicates()
            df = df.merge(events_lookup, on=["year", "round"], how="left")
            df["event_date"] = pd.to_datetime(df["event_date"])

        # Ensure position is numeric
        df["position"] = pd.to_numeric(df["position"], errors="coerce")

        # Add binary indicators
        df["is_top3"] = (df["position"] <= 3).astype(int)
        df["is_pole"] = (df["position"] == 1).astype(int)
        df["is_front_row"] = (df["position"] <= 2).astype(int)

        return df

    def _add_temporal_features(
        self,
        df: pd.DataFrame,
        windows: list[int],
    ) -> pd.DataFrame:
        """Add multi-window rolling features with temporal safety."""
        logger.info("Adding core temporal features...")

        df = df.sort_values(["driver_code", "year", "round"])

        for window in windows:
            # Rolling average position (shifted to prevent leakage)
            df[f"rolling_avg_pos_{window}"] = df.groupby("driver_code")["position"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Rolling top-3 rate
            df[f"rolling_top3_rate_{window}"] = df.groupby("driver_code")["is_top3"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Rolling pole rate
            df[f"rolling_pole_rate_{window}"] = df.groupby("driver_code")["is_pole"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Rolling front row rate
            df[f"rolling_front_row_rate_{window}"] = df.groupby("driver_code")[
                "is_front_row"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Position std (consistency)
            df[f"rolling_pos_std_{window}"] = df.groupby("driver_code")["position"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).std()
            )

        # Momentum: short-term vs long-term form
        if 3 in windows and 10 in windows:
            df["momentum"] = df["rolling_avg_pos_10"] - df["rolling_avg_pos_3"]

        # Season race count
        df["season_race_count"] = df.groupby(["driver_code", "year"]).cumcount()

        # Career race count
        df["career_race_count"] = df.groupby("driver_code").cumcount()

        return df

    def _add_team_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add team/constructor performance features."""
        logger.info("Adding team features...")

        results_with_date = target_results.copy()
        results_with_date["position"] = pd.to_numeric(
            results_with_date["position"], errors="coerce"
        )
        results_with_date = results_with_date.sort_values(["team", "year", "round"])

        # Team average position (rolling, shifted)
        team_avg = results_with_date.groupby("team")["position"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        results_with_date["team_rolling_avg_pos"] = team_avg

        # Team best position (rolling)
        team_best = results_with_date.groupby("team")["position"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).min()
        )
        results_with_date["team_rolling_best_pos"] = team_best

        # Merge back
        team_features = results_with_date[
            ["session_key", "driver_code", "team_rolling_avg_pos", "team_rolling_best_pos"]
        ].drop_duplicates(subset=["session_key", "driver_code"])
        df = df.merge(team_features, on=["session_key", "driver_code"], how="left")

        return df

    def _add_circuit_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add circuit-specific performance features."""
        logger.info("Adding circuit features...")

        circuit_features = self.circuit_extractor.extract_features(target_results)

        if circuit_features.empty:
            return df

        merge_keys = ["session_key", "driver_code"]
        exclude_cols = merge_keys + ["year", "round", "circuit", "team"]
        feature_cols = [c for c in circuit_features.columns if c not in exclude_cols]

        if feature_cols:
            merge_df = circuit_features[merge_keys + feature_cols].drop_duplicates(
                subset=merge_keys
            )
            df = df.merge(merge_df, on=merge_keys, how="left")

        return df

    def _add_driver_circuit_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add driver-circuit interaction features."""
        logger.info("Adding driver-circuit interaction features...")

        driver_circuit_features = self.driver_circuit_extractor.extract_features(target_results)

        if driver_circuit_features.empty:
            return df

        merge_keys = ["session_key", "driver_code"]
        exclude_cols = merge_keys + ["year", "round", "circuit", "team"]
        feature_cols = [c for c in driver_circuit_features.columns if c not in exclude_cols]

        if feature_cols:
            merge_df = driver_circuit_features[merge_keys + feature_cols].drop_duplicates(
                subset=merge_keys
            )
            df = df.merge(merge_df, on=merge_keys, how="left")

        return df

    def _add_momentum_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add momentum features from historical performance."""
        logger.info("Adding momentum features...")

        momentum_features = self.momentum_extractor.extract_features(target_results)

        if momentum_features.empty:
            return df

        return self._merge_features(df, momentum_features, "momentum")

    def _add_relative_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add relative performance features."""
        logger.info("Adding relative features...")

        relative_features = self.relative_extractor.extract_features(target_results)

        if relative_features.empty:
            return df

        return self._merge_features(df, relative_features, "relative")

    def _add_elo_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add ELO rating features for drivers and constructors."""
        logger.info("Adding ELO rating features...")

        elo_features = self.elo_extractor.extract_features(
            target_results, session_type=self.session_type
        )

        if elo_features.empty:
            return df

        return self._merge_features(df, elo_features, "elo")

    def _add_reliability_features(
        self,
        df: pd.DataFrame,
        min_year: int,
    ) -> pd.DataFrame:
        """Add DNF rate and reliability features from race history."""
        logger.info("Adding reliability/DNF features...")

        # Load race results for reliability calculation
        race_results = self.loader.load_race_results(min_year=min_year)

        if race_results.empty:
            logger.warning("No race results for reliability features")
            return df

        reliability_features = self.reliability_extractor.extract_features(race_results)

        if reliability_features.empty:
            return df

        return self._merge_features(df, reliability_features, "reliability")

    def _add_first_lap_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
        min_year: int,
    ) -> pd.DataFrame:
        """Add first lap position change features from race history."""
        logger.info("Adding first lap performance features...")

        first_lap_features = self.first_lap_extractor.extract_features(
            target_results, min_year=min_year
        )

        if first_lap_features.empty:
            logger.warning("No first lap features extracted")
            return df

        return self._merge_features(df, first_lap_features, "first_lap")

    def _add_wet_weather_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
        min_year: int,
    ) -> pd.DataFrame:
        """Add wet weather skill features from race history."""
        logger.info("Adding wet weather skill features...")

        wet_features = self.wet_weather_extractor.extract_features(
            target_results, min_year=min_year
        )

        if wet_features.empty:
            logger.warning("No wet weather features extracted")
            return df

        return self._merge_features(df, wet_features, "wet_weather")

    def _add_circuit_overtaking_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
        min_year: int,
    ) -> pd.DataFrame:
        """Add circuit overtaking difficulty features from race history."""
        logger.info("Adding circuit overtaking features...")

        overtaking_features = self.circuit_overtaking_extractor.extract_features(
            target_results, min_year=min_year
        )

        if overtaking_features.empty:
            logger.warning("No circuit overtaking features extracted")
            return df

        return self._merge_features(df, overtaking_features, "overtaking")

    def _add_track_evolution_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
        min_year: int,
    ) -> pd.DataFrame:
        """Add track evolution features (grip improvement through sessions)."""
        logger.info("Adding track evolution features...")

        evolution_features = self.track_evolution_extractor.extract_features(
            target_results, min_year=min_year
        )

        if evolution_features.empty:
            logger.warning("No track evolution features extracted")
            return df

        return self._merge_features(df, evolution_features, "evolution")

    def _merge_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """Merge extracted features with base dataframe."""
        if "session_key" in features.columns and "session_key" in df.columns:
            merge_keys = ["session_key", "driver_code"]
        elif "weekend_key" in features.columns:
            merge_keys = ["year", "round", "driver_code"]
        else:
            merge_keys = ["year", "round", "driver_code"]

        merge_keys = [k for k in merge_keys if k in features.columns and k in df.columns]

        if not merge_keys:
            logger.warning(f"No valid merge keys for {prefix} features")
            return df

        exclude_cols = merge_keys + [
            "year",
            "round",
            "circuit",
            "team",
            "position",
            "session_key",
            "weekend_key",
        ]
        feature_cols = [c for c in features.columns if c not in exclude_cols]

        if not feature_cols:
            return df

        merge_df = features[merge_keys + feature_cols].drop_duplicates(subset=merge_keys)
        df = df.merge(merge_df, on=merge_keys, how="left", suffixes=("", f"_{prefix}"))

        return df

    def _get_feature_columns(self, windows: list[int]) -> list[str]:
        """Get list of base feature columns. Subclasses extend this."""
        feature_cols = []

        # Multi-window temporal features
        for w in windows:
            feature_cols.extend(
                [
                    f"rolling_avg_pos_{w}",
                    f"rolling_top3_rate_{w}",
                    f"rolling_pole_rate_{w}",
                    f"rolling_front_row_rate_{w}",
                    f"rolling_pos_std_{w}",
                ]
            )

        # Momentum
        feature_cols.append("momentum")

        # Experience
        feature_cols.extend(
            [
                "season_race_count",
                "career_race_count",
            ]
        )

        # Team features
        feature_cols.extend(
            [
                "team_rolling_avg_pos",
                "team_rolling_best_pos",
            ]
        )

        # Circuit features
        feature_cols.extend(
            [
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
        )

        # Driver-circuit interaction features
        feature_cols.extend(
            [
                "circuit_type_appearances",
                "circuit_type_avg_position",
                "circuit_type_best_position",
                "circuit_type_top3_rate",
                "circuit_type_position_std",
                "circuit_teammate_delta_avg",
                "circuit_beats_teammate_rate",
                "circuit_type_teammate_delta",
                "circuit_type_beats_teammate_rate",
                "circuit_type_affinity",
                "driver_overall_avg_pos",
            ]
        )
        # Rolling window features for circuit type
        for w in [3, 5]:
            feature_cols.append(f"circuit_type_avg_pos_{w}")

        # Momentum features
        feature_cols.extend(
            [
                "form_acceleration",
                "momentum_reversal",
                "recent_vs_longterm",
                "form_improving",
            ]
        )
        for span in [3, 5]:
            feature_cols.extend(
                [
                    f"ewm_position_{span}",
                    f"ewm_top3_rate_{span}",
                ]
            )
        for w in [5, 10]:
            feature_cols.append(f"position_trend_{w}")

        # Relative features
        for w in windows:
            feature_cols.extend(
                [
                    f"avg_pos_vs_field_{w}",
                    f"avg_teammate_delta_{w}",
                    f"beats_teammate_rate_{w}",
                    f"avg_percentile_{w}",
                ]
            )

        # ELO rating features
        feature_cols.extend(
            [
                "driver_elo",
                "constructor_elo",
                "ga_driver_elo",
                "driver_elo_vs_initial",
                "constructor_elo_vs_initial",
                "driver_elo_percentile",
                "constructor_elo_percentile",
                "driver_elo_vs_avg",
                "constructor_elo_vs_avg",
                "combined_elo",
            ]
        )

        # Reliability/DNF features
        for w in windows:
            feature_cols.extend(
                [
                    f"driver_dnf_rate_{w}",
                    f"driver_mechanical_dnf_rate_{w}",
                    f"driver_incident_dnf_rate_{w}",
                    f"driver_finish_rate_{w}",
                    # team_dnf_rate is already in race_pace_features
                    f"team_mechanical_dnf_rate_{w}",
                ]
            )
        feature_cols.extend(
            [
                "driver_career_dnf_rate",
                "consecutive_finishes",
                "dnf_last_race",
                "team_career_dnf_rate",
                "driver_dnf_vs_field",
                "driver_reliability_percentile",
                "driver_confidence",
                "driver_season_confidence",
                "races_since_dnf",
            ]
        )

        # First lap performance features
        for w in windows:
            feature_cols.extend(
                [
                    f"first_lap_avg_gain_{w}",
                    f"first_lap_gain_std_{w}",
                    f"first_lap_gain_rate_{w}",
                    f"first_lap_loss_rate_{w}",
                    f"first_lap_hold_rate_{w}",
                    f"first_lap_max_gain_{w}",
                    f"first_lap_min_gain_{w}",
                    f"team_first_lap_avg_gain_{w}",
                ]
            )
        feature_cols.extend(
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

        # Wet weather skill features
        # Note: is_wet_session is already in weather_features, so not duplicated here
        feature_cols.extend(
            [
                "has_rainfall_wet",
            ]
        )
        for w in windows:
            feature_cols.extend(
                [
                    f"wet_race_count_{w}",
                    f"wet_race_rate_{w}",
                    f"wet_avg_position_{w}",
                    f"dry_avg_position_{w}",
                    f"wet_dry_delta_{w}",
                    f"wet_overperformance_{w}",
                    f"team_wet_avg_position_{w}",
                ]
            )
        feature_cols.extend(
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

        # Circuit overtaking difficulty features
        feature_cols.extend(
            [
                "circuit_avg_positions_changed",
                "circuit_max_positions_changed",
                "circuit_positions_changed_std",
                "circuit_overtaking_rate",
                "circuit_difficulty",
                "grid_importance",
                "circuit_race_count",
                "low_overtaking_circuit",
                "high_overtaking_circuit",
            ]
        )
        for w in windows:
            feature_cols.extend(
                [
                    f"driver_avg_overtakes_{w}",
                    f"driver_overtakes_std_{w}",
                    f"driver_gain_rate_{w}",
                    f"driver_max_overtakes_{w}",
                    f"overtake_potential_{w}",
                ]
            )
        feature_cols.extend(
            [
                "driver_career_avg_overtakes",
                "driver_career_overtake_rate",
                "overtaking_specialist",
                "position_defender",
            ]
        )

        # Track evolution features (grip improvement through sessions)
        for w in windows:
            feature_cols.extend(
                [
                    f"avg_evolution_ms_{w}",
                    f"avg_evolution_pct_{w}",
                    f"avg_consistency_gain_{w}",
                    f"evolution_std_{w}",
                ]
            )
        feature_cols.extend(
            [
                "career_avg_evolution",
                "career_evolution_consistency",
                # Note: late_session_specialist, green_track_specialist, and
                # current_track_evolution_* removed due to 0% feature importance
            ]
        )

        return feature_cols

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert object columns to numeric for model compatibility."""
        if "strongest_sector" in df.columns:
            sector_map = {"S1": 1, "S2": 2, "S3": 3}
            df["strongest_sector"] = df["strongest_sector"].map(sector_map).fillna(0).astype(int)

        bool_like_cols = ["current_used_soft", "current_used_medium", "final_run_specialist"]
        for col in bool_like_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .map(lambda x: 1 if x is True or x == "True" or x == 1 else 0)
                    .astype(int)
                )

        return df
