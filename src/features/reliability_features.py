"""
DNF rate and reliability feature extraction for F1 predictions.

Captures driver and constructor reliability patterns:
- DNF (Did Not Finish) rates
- Driver confidence (% races completed)
- Mechanical vs incident-related retirements
- Team reliability trends

Inspired by F1-Predictor project (JaideepGuntupalli) which found DNF rate
to be a significant predictor of race outcomes.

All features use temporal shift to prevent data leakage.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Classification of status values
FINISHED_STATUSES = {
    "Finished",
    "Lapped",
    "+1 Lap",
    "+2 Laps",
    "+3 Laps",
    "+4 Laps",
    "+5 Laps",
    "+6 Laps",
}

MECHANICAL_DNF_STATUSES = {
    "Engine",
    "Gearbox",
    "Transmission",
    "Hydraulics",
    "Brakes",
    "Suspension",
    "Power Unit",
    "ERS",
    "Turbo",
    "Fuel pressure",
    "Fuel leak",
    "Cooling system",
    "Oil leak",
    "Oil pressure",
    "Water leak",
    "Electrical",
    "Battery",
    "Clutch",
    "Throttle",
    "Steering",
    "Driveshaft",
    "Wheel",
    "Wheel nut",
    "Puncture",
    "Tyre",
    "Front wing",
    "Rear wing",
    "Undertray",
    "Floor",
    "Power loss",
    "Overheating",
    "Fire",
    "Exhaust",
    "Retired",  # Often mechanical
}

INCIDENT_DNF_STATUSES = {
    "Accident",
    "Collision",
    "Collision damage",
    "Spun off",
    "Crash",
    "Damage",
    "Barrier",
    "Wall",
}

OTHER_DNF_STATUSES = {
    "Disqualified",
    "Did not start",
    "Withdrew",
    "Excluded",
    "Not classified",
    "DNS",
    "DSQ",
    "Illness",
}


def classify_status(status: str) -> str:
    """Classify race status into categories."""
    if pd.isna(status):
        return "unknown"
    status = str(status).strip()

    if status in FINISHED_STATUSES:
        return "finished"
    elif status in MECHANICAL_DNF_STATUSES:
        return "mechanical_dnf"
    elif status in INCIDENT_DNF_STATUSES:
        return "incident_dnf"
    elif status in OTHER_DNF_STATUSES:
        return "other_dnf"
    else:
        # Check for partial matches
        status_lower = status.lower()
        if any(x in status_lower for x in ["engine", "gearbox", "power", "fuel", "oil"]):
            return "mechanical_dnf"
        elif any(x in status_lower for x in ["accident", "collision", "spun", "crash"]):
            return "incident_dnf"
        elif "lap" in status_lower or "finish" in status_lower:
            return "finished"
        else:
            return "other_dnf"


class ReliabilityFeatureExtractor:
    """
    Extract DNF rate and reliability features for F1 prediction.

    Features capture:
    - Driver DNF rates (overall, mechanical, incident)
    - Constructor/team reliability
    - Driver confidence (completion rate)
    - Recent reliability trends
    """

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize reliability feature extractor.

        Args:
            windows: Rolling window sizes for temporal features
        """
        self.windows = windows or [3, 5, 10]
        logger.info(f"Initialized ReliabilityFeatureExtractor (windows={self.windows})")

    def extract_features(self, race_results: pd.DataFrame) -> pd.DataFrame:
        """
        Extract reliability features from race results.

        Args:
            race_results: Race results with status column

        Returns:
            DataFrame with reliability features per driver per session
        """
        logger.info("Extracting reliability features...")

        if race_results.empty:
            logger.warning("No race results provided")
            return pd.DataFrame()

        required_cols = ["session_key", "driver_code", "year", "round", "status"]
        if not all(c in race_results.columns for c in required_cols):
            missing = [c for c in required_cols if c not in race_results.columns]
            logger.warning(f"Missing required columns: {missing}")
            return pd.DataFrame()

        df = race_results.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        # Classify status
        df["status_class"] = df["status"].apply(classify_status)
        df["is_dnf"] = (df["status_class"] != "finished").astype(int)
        df["is_mechanical_dnf"] = (df["status_class"] == "mechanical_dnf").astype(int)
        df["is_incident_dnf"] = (df["status_class"] == "incident_dnf").astype(int)
        df["is_finished"] = (df["status_class"] == "finished").astype(int)

        # Start with base columns
        features = df[["session_key", "driver_code", "year", "round"]].copy()
        if "team" in df.columns:
            features["team"] = df["team"]

        # Add driver reliability features
        features = self._add_driver_reliability_features(features, df)

        # Add constructor reliability features
        if "team" in df.columns:
            features = self._add_constructor_reliability_features(features, df)

        # Add relative reliability features
        features = self._add_relative_reliability_features(features, df)

        # Add driver confidence (career completion rate)
        features = self._add_driver_confidence(features, df)

        n_features = len(
            [
                c
                for c in features.columns
                if c not in ["session_key", "driver_code", "year", "round", "team"]
            ]
        )
        logger.info(f"Extracted {n_features} reliability features")

        return features

    def _add_driver_reliability_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add driver-specific reliability features."""
        df = df.sort_values(["driver_code", "year", "round"])

        for window in self.windows:
            # Overall DNF rate
            features[f"driver_dnf_rate_{window}"] = df.groupby("driver_code")["is_dnf"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Mechanical DNF rate (car reliability indicator)
            features[f"driver_mechanical_dnf_rate_{window}"] = df.groupby("driver_code")[
                "is_mechanical_dnf"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Incident DNF rate (driver error/risk indicator)
            features[f"driver_incident_dnf_rate_{window}"] = df.groupby("driver_code")[
                "is_incident_dnf"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Finish rate (driver confidence)
            features[f"driver_finish_rate_{window}"] = df.groupby("driver_code")[
                "is_finished"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Career DNF rate (expanding window)
        features["driver_career_dnf_rate"] = df.groupby("driver_code")["is_dnf"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Consecutive finishes streak
        features["consecutive_finishes"] = df.groupby("driver_code")["is_finished"].transform(
            lambda x: x.shift(1).groupby((x.shift(1) != x.shift(2)).cumsum()).cumsum()
        )

        # Recent DNF (did driver DNF in last race)
        features["dnf_last_race"] = df.groupby("driver_code")["is_dnf"].transform(
            lambda x: x.shift(1)
        )

        return features

    def _add_constructor_reliability_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add constructor/team reliability features."""
        if "team" not in df.columns:
            return features

        df = df.sort_values(["team", "year", "round"])

        for window in self.windows:
            # Note: team_dnf_rate is already computed in race_pace_features.py
            # Only add team_mechanical_dnf_rate (car reliability breakdown)
            features[f"team_mechanical_dnf_rate_{window}"] = df.groupby("team")[
                "is_mechanical_dnf"
            ].transform(lambda x: x.shift(1).rolling(window * 2, min_periods=2).mean())

        # Team career reliability
        features["team_career_dnf_rate"] = df.groupby("team")["is_dnf"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        return features

    def _add_relative_reliability_features(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add features comparing driver/team reliability to field average."""
        # Calculate field average DNF rate per session
        session_dnf_rate = df.groupby("session_key")["is_dnf"].transform("mean")

        # Driver's DNF rate vs field (using career rate)
        driver_career_dnf = df.groupby("driver_code")["is_dnf"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # We need to align indices properly
        df_aligned = df.copy()
        df_aligned["driver_career_dnf"] = driver_career_dnf
        df_aligned["session_dnf_rate"] = session_dnf_rate

        features["driver_dnf_vs_field"] = (
            df_aligned["driver_career_dnf"] - df_aligned["session_dnf_rate"]
        ).values

        # Reliability percentile within field (rank driver's career finish rate vs others in session)
        driver_career_finish_rate = df.groupby("driver_code")["is_finished"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df_aligned["driver_career_finish_rate"] = driver_career_finish_rate
        features["driver_reliability_percentile"] = df_aligned.groupby("session_key")[
            "driver_career_finish_rate"
        ].rank(pct=True)

        return features

    def _add_driver_confidence(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add driver confidence feature.

        Driver confidence = percentage of career races completed without DNF.
        Higher confidence indicates more reliable/cautious driver.
        """
        df = df.sort_values(["driver_code", "year", "round"])

        # Career completion rate (expanding)
        features["driver_confidence"] = df.groupby("driver_code")["is_finished"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # Season completion rate
        features["driver_season_confidence"] = df.groupby(["driver_code", "year"])[
            "is_finished"
        ].transform(lambda x: x.shift(1).expanding().mean())

        # Races since last DNF
        def races_since_dnf(series):
            result = []
            count = 0
            for val in series.shift(1):
                if pd.isna(val):
                    result.append(0)
                elif val == 1:  # finished
                    count += 1
                    result.append(count)
                else:  # DNF
                    count = 0
                    result.append(0)
            return pd.Series(result, index=series.index)

        features["races_since_dnf"] = df.groupby("driver_code")["is_finished"].transform(
            races_since_dnf
        )

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = []

        # Driver reliability features
        for window in self.windows:
            names.extend(
                [
                    f"driver_dnf_rate_{window}",
                    f"driver_mechanical_dnf_rate_{window}",
                    f"driver_incident_dnf_rate_{window}",
                    f"driver_finish_rate_{window}",
                    # team_dnf_rate is in race_pace_features
                    f"team_mechanical_dnf_rate_{window}",
                ]
            )

        # Career/expanding features
        names.extend(
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

        return names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.loaders import F1DataLoader

    loader = F1DataLoader()

    # Load race results
    race_results = loader.load_race_results(min_year=2022)
    print(f"Loaded {len(race_results)} race results")

    # Extract reliability features
    extractor = ReliabilityFeatureExtractor()
    features = extractor.extract_features(race_results)

    print(f"\nFeature columns: {features.columns.tolist()}")
    print(f"\nFeature shape: {features.shape}")
    print("\nSample features:")
    print(features.head(20))

    # Check DNF rate distribution
    print("\nDriver DNF rate (last 5) stats:")
    print(features["driver_dnf_rate_5"].describe())
    print("\nDriver confidence stats:")
    print(features["driver_confidence"].describe())
