"""
Data loaders for F1 FastF1 datasets.

Loads session parquet files with lap times, sector times, and weather data.
Prevents future data leakage through temporal filtering.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class F1DataLoader:
    """Loads F1 data from FastF1 parquet files."""

    def __init__(self, data_dir: Path = Path("data/fastf1")):
        """
        Initialize data loader.

        Args:
            data_dir: Path to data/fastf1/ directory
        """
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "sessions"
        self.metadata_dir = self.data_dir / "metadata"
        logger.info(f"Initialized F1DataLoader with data_dir: {self.data_dir}")

    def load_session(self, year: int, round_num: int, session_type: str) -> pd.DataFrame:
        """
        Load a single session's data.

        Args:
            year: Season year
            round_num: Round number
            session_type: Session type (FP1, FP2, FP3, Q, R, S, SQ)

        Returns:
            DataFrame with session lap data
        """
        session_key = f"{year}_{round_num:02d}_{session_type}"
        parquet_path = self.sessions_dir / f"{session_key}.parquet"

        if not parquet_path.exists():
            logger.warning(f"Session file not found: {parquet_path}")
            return pd.DataFrame()

        df = pd.read_parquet(parquet_path)
        logger.debug(f"Loaded session {session_key}: {len(df)} rows")
        return df

    def load_sessions(
        self,
        min_year: int = 2020,
        max_year: int | None = None,
        session_types: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load multiple sessions across seasons.

        Args:
            min_year: Minimum year to include
            max_year: Maximum year to include (None = no limit)
            session_types: List of session types to include (None = all)

        Returns:
            DataFrame with all matching session data
        """
        logger.info(f"Loading sessions from {min_year}...")

        all_sessions = []

        for parquet_path in sorted(self.sessions_dir.glob("*.parquet")):
            # Parse session key from filename
            parts = parquet_path.stem.split("_")
            if len(parts) < 3:
                continue

            year = int(parts[0])
            session_type = parts[2]

            # Filter by year
            if year < min_year:
                continue
            if max_year and year > max_year:
                continue

            # Filter by session type
            if session_types and session_type not in session_types:
                continue

            try:
                df = pd.read_parquet(parquet_path)
                all_sessions.append(df)
            except FileNotFoundError:
                logger.warning(f"Session file not found: {parquet_path}")
            except pd.errors.EmptyDataError:
                logger.warning(f"Empty parquet file: {parquet_path}")
            except OSError as e:
                logger.warning(f"IO error reading {parquet_path}: {e}")
            except Exception as e:
                # Log unexpected errors but continue processing other files
                logger.error(f"Unexpected error loading {parquet_path}: {type(e).__name__}: {e}")

        if not all_sessions:
            logger.warning("No sessions found")
            return pd.DataFrame()

        combined = pd.concat(all_sessions, ignore_index=True)
        logger.info(f"Loaded {len(combined)} rows from {len(all_sessions)} sessions")
        return combined

    def load_qualifying_sessions(self, min_year: int = 2020) -> pd.DataFrame:
        """
        Load all qualifying session data.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with qualifying lap data including sector times
        """
        return self.load_sessions(min_year=min_year, session_types=["Q"])

    def load_race_sessions(self, min_year: int = 2020) -> pd.DataFrame:
        """
        Load all race session data.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with race lap data
        """
        return self.load_sessions(min_year=min_year, session_types=["R"])

    def load_practice_sessions(self, min_year: int = 2020) -> pd.DataFrame:
        """
        Load all practice session data.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with practice lap data
        """
        return self.load_sessions(min_year=min_year, session_types=["FP1", "FP2", "FP3"])

    def load_qualifying_results(self, min_year: int = 2020) -> pd.DataFrame:
        """
        Load qualifying results (final positions) for all sessions.

        This extracts the final position for each driver from qualifying sessions,
        similar to the old qualifying.csv format but with richer data.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with one row per driver per qualifying session
        """
        logger.info(f"Loading qualifying results from {min_year}...")

        quali_sessions = self.load_qualifying_sessions(min_year=min_year)

        if quali_sessions.empty:
            return pd.DataFrame()

        # Get unique driver results per session
        # Group by session and driver, take the row with best information

        # Build aggregation dict with only columns that exist
        agg_dict = {
            "year": "first",
            "round": "first",
            "circuit": "first",
            "team": "first",
            "position": "first",  # Final position
        }

        # Add optional columns if they exist
        optional_first = ["event_name", "driver_number", "q1_time_ms", "q2_time_ms", "q3_time_ms"]
        for col in optional_first:
            if col in quali_sessions.columns:
                agg_dict[col] = "first"

        # Add min aggregations for timing columns
        optional_min = ["lap_time_ms", "sector1_time_ms", "sector2_time_ms", "sector3_time_ms"]
        for col in optional_min:
            if col in quali_sessions.columns:
                agg_dict[col] = "min"

        results = quali_sessions.groupby(["session_key", "driver_code"]).agg(agg_dict).reset_index()

        # Rename timing columns for clarity
        rename_map = {
            "lap_time_ms": "best_lap_ms",
            "sector1_time_ms": "best_s1_ms",
            "sector2_time_ms": "best_s2_ms",
            "sector3_time_ms": "best_s3_ms",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in results.columns}
        results = results.rename(columns=rename_map)

        logger.info(f"Loaded {len(results)} qualifying results")
        return results

    def load_race_results(self, min_year: int = 2020) -> pd.DataFrame:
        """
        Load race results (final positions) for all sessions.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with one row per driver per race
        """
        logger.info(f"Loading race results from {min_year}...")

        race_sessions = self.load_race_sessions(min_year=min_year)

        if race_sessions.empty:
            return pd.DataFrame()

        # Get unique driver results per session
        results = (
            race_sessions.groupby(["session_key", "driver_code"])
            .agg(
                {
                    "year": "first",
                    "round": "first",
                    "circuit": "first",
                    "event_name": "first",
                    "team": "first",
                    "driver_number": "first",
                    "position": "first",
                    "grid_position": "first",
                    "points": "first",
                    "status": "first",
                    # Aggregate lap stats
                    "lap_time_ms": "min",  # Best lap
                    "lap_number": "max",  # Laps completed
                }
            )
            .reset_index()
        )

        results = results.rename(
            columns={
                "lap_time_ms": "best_lap_ms",
                "lap_number": "laps_completed",
            }
        )

        logger.info(f"Loaded {len(results)} race results")
        return results

    def load_events(self, min_year: int = 2020) -> pd.DataFrame:
        """
        Load events metadata.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with event information
        """
        events_path = self.metadata_dir / "events.parquet"

        if not events_path.exists():
            logger.warning(f"Events file not found: {events_path}")
            return pd.DataFrame()

        df = pd.read_parquet(events_path)

        if "year" in df.columns:
            df = df[df["year"] >= min_year].copy()

        logger.info(f"Loaded {len(df)} events")
        return df

    def load_drivers(self) -> pd.DataFrame:
        """
        Load driver metadata.

        Returns:
            DataFrame with driver information
        """
        drivers_path = self.metadata_dir / "drivers.parquet"

        if not drivers_path.exists():
            logger.warning(f"Drivers file not found: {drivers_path}")
            return pd.DataFrame()

        df = pd.read_parquet(drivers_path)
        logger.info(f"Loaded {len(df)} drivers")
        return df

    def load_teams(self) -> pd.DataFrame:
        """
        Load team metadata.

        Returns:
            DataFrame with team information
        """
        teams_path = self.metadata_dir / "teams.parquet"

        if not teams_path.exists():
            logger.warning(f"Teams file not found: {teams_path}")
            return pd.DataFrame()

        df = pd.read_parquet(teams_path)
        logger.info(f"Loaded {len(df)} teams")
        return df

    def load_session_laps(
        self,
        year: int,
        round_num: int,
        session_type: str,
        driver_code: str | None = None,
    ) -> pd.DataFrame:
        """
        Load lap-by-lap data for a specific session.

        Args:
            year: Season year
            round_num: Round number
            session_type: Session type
            driver_code: Filter to specific driver (None = all)

        Returns:
            DataFrame with lap data
        """
        df = self.load_session(year, round_num, session_type)

        if df.empty:
            return df

        if driver_code:
            df = df[df["driver_code"] == driver_code].copy()

        return df.sort_values(["driver_code", "lap_number"])

    def get_sessions_before(
        self,
        target_year: int,
        target_round: int,
        session_types: list[str] | None = None,
        min_year: int = 2020,
    ) -> pd.DataFrame:
        """
        Load all sessions BEFORE a target race (for temporal safety).

        This is critical for feature engineering - ensures we only use
        historical data when predicting a specific race.

        Args:
            target_year: Target year
            target_round: Target round
            session_types: Session types to include
            min_year: Minimum year to include

        Returns:
            DataFrame with sessions strictly before target
        """
        logger.info(f"Loading sessions before {target_year} R{target_round}...")

        all_sessions = []

        for parquet_path in sorted(self.sessions_dir.glob("*.parquet")):
            parts = parquet_path.stem.split("_")
            if len(parts) < 3:
                continue

            year = int(parts[0])
            round_num = int(parts[1])
            session_type = parts[2]

            # Skip if after or at target
            if year > target_year:
                continue
            if year == target_year and round_num >= target_round:
                continue

            # Filter by year
            if year < min_year:
                continue

            # Filter by session type
            if session_types and session_type not in session_types:
                continue

            try:
                df = pd.read_parquet(parquet_path)
                all_sessions.append(df)
            except FileNotFoundError:
                logger.warning(f"Session file not found: {parquet_path}")
            except pd.errors.EmptyDataError:
                logger.warning(f"Empty parquet file: {parquet_path}")
            except OSError as e:
                logger.warning(f"IO error reading {parquet_path}: {e}")
            except Exception as e:
                # Log unexpected errors but continue processing other files
                logger.error(f"Unexpected error loading {parquet_path}: {type(e).__name__}: {e}")

        if not all_sessions:
            return pd.DataFrame()

        combined = pd.concat(all_sessions, ignore_index=True)
        logger.info(f"Loaded {len(combined)} rows from {len(all_sessions)} sessions before target")
        return combined

    def enforce_temporal_validation(
        self, df: pd.DataFrame, cutoff_date: date | None = None
    ) -> pd.DataFrame:
        """
        Enforce temporal constraint: only data before cutoff_date.

        Args:
            df: DataFrame to filter
            cutoff_date: Maximum date to include (default: today)

        Returns:
            Filtered DataFrame
        """
        if cutoff_date is None:
            cutoff_date = date.today()

        # Try to filter by event_date if available in events metadata
        events = self.load_events()

        if events.empty or "year" not in df.columns or "round" not in df.columns:
            logger.warning("Cannot enforce temporal validation without year/round")
            return df

        # Merge to get event dates
        events_lookup = events[["year", "round", "event_date"]].copy()
        original_len = len(df)

        df = df.merge(events_lookup, on=["year", "round"], how="left")
        df = df[pd.to_datetime(df["event_date"]).dt.date < cutoff_date].copy()
        df = df.drop(columns=["event_date"], errors="ignore")

        filtered_len = len(df)

        if filtered_len < original_len:
            logger.info(
                f"Temporal validation: filtered {original_len - filtered_len} future records"
            )

        return df

    def get_available_sessions(self) -> list[str]:
        """
        List all available session keys.

        Returns:
            List of session keys (e.g., ['2024_01_Q', '2024_01_R', ...])
        """
        sessions = []
        for parquet_path in sorted(self.sessions_dir.glob("*.parquet")):
            sessions.append(parquet_path.stem)
        return sessions

    def get_available_years(self) -> list[int]:
        """
        Get list of years with available data.

        Returns:
            Sorted list of years
        """
        years = set()
        for parquet_path in self.sessions_dir.glob("*.parquet"):
            parts = parquet_path.stem.split("_")
            if parts:
                try:
                    years.add(int(parts[0]))
                except ValueError:
                    pass
        return sorted(years)

    def load_sprint_qualifying_sessions(self, min_year: int = 2021) -> pd.DataFrame:
        """
        Load all Sprint Qualifying (SQ) session data.

        Sprint Qualifying was introduced in 2021 and sets the grid for the Sprint race.

        Args:
            min_year: Minimum year to include (default 2021, first year of sprints)

        Returns:
            DataFrame with sprint qualifying lap data
        """
        return self.load_sessions(min_year=min_year, session_types=["SQ"])

    def load_sprint_race_sessions(self, min_year: int = 2021) -> pd.DataFrame:
        """
        Load all Sprint Race (S) session data.

        Sprint races are shorter races that award points and set grid positions.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with sprint race lap data
        """
        return self.load_sessions(min_year=min_year, session_types=["S"])

    def load_sprint_results(self, min_year: int = 2021) -> pd.DataFrame:
        """
        Load Sprint Race results (final positions) for all sessions.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with one row per driver per sprint race
        """
        logger.info(f"Loading sprint results from {min_year}...")

        sprint_sessions = self.load_sprint_race_sessions(min_year=min_year)

        if sprint_sessions.empty:
            return pd.DataFrame()

        # Build aggregation dict with only columns that exist
        agg_dict = {
            "year": "first",
            "round": "first",
            "circuit": "first",
            "team": "first",
            "position": "first",
        }

        # Add optional columns if they exist
        optional_first = ["event_name", "driver_number", "grid_position", "points", "status"]
        for col in optional_first:
            if col in sprint_sessions.columns:
                agg_dict[col] = "first"

        # Add min aggregations for timing columns
        optional_min = ["lap_time_ms"]
        for col in optional_min:
            if col in sprint_sessions.columns:
                agg_dict[col] = "min"

        # Add max for lap count
        if "lap_number" in sprint_sessions.columns:
            agg_dict["lap_number"] = "max"

        results = (
            sprint_sessions.groupby(["session_key", "driver_code"]).agg(agg_dict).reset_index()
        )

        # Rename columns for clarity
        rename_map = {
            "lap_time_ms": "best_lap_ms",
            "lap_number": "laps_completed",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in results.columns}
        results = results.rename(columns=rename_map)

        logger.info(f"Loaded {len(results)} sprint results")
        return results

    def load_sprint_qualifying_results(self, min_year: int = 2021) -> pd.DataFrame:
        """
        Load Sprint Qualifying results (final positions) for all sessions.

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with one row per driver per sprint qualifying session
        """
        logger.info(f"Loading sprint qualifying results from {min_year}...")

        sq_sessions = self.load_sprint_qualifying_sessions(min_year=min_year)

        if sq_sessions.empty:
            return pd.DataFrame()

        # Build aggregation dict with only columns that exist
        agg_dict = {
            "year": "first",
            "round": "first",
            "circuit": "first",
            "team": "first",
            "position": "first",
        }

        # Add optional columns if they exist
        optional_first = ["event_name", "driver_number"]
        for col in optional_first:
            if col in sq_sessions.columns:
                agg_dict[col] = "first"

        # Add min aggregations for timing columns
        optional_min = ["lap_time_ms", "sector1_time_ms", "sector2_time_ms", "sector3_time_ms"]
        for col in optional_min:
            if col in sq_sessions.columns:
                agg_dict[col] = "min"

        results = sq_sessions.groupby(["session_key", "driver_code"]).agg(agg_dict).reset_index()

        # Rename timing columns for clarity
        rename_map = {
            "lap_time_ms": "best_lap_ms",
            "sector1_time_ms": "best_s1_ms",
            "sector2_time_ms": "best_s2_ms",
            "sector3_time_ms": "best_s3_ms",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in results.columns}
        results = results.rename(columns=rename_map)

        logger.info(f"Loaded {len(results)} sprint qualifying results")
        return results

    def is_sprint_weekend(self, year: int, round_num: int) -> bool:
        """
        Check if a given round is a sprint weekend.

        Args:
            year: Season year
            round_num: Round number

        Returns:
            True if sprint sessions exist for this round
        """
        sprint_path = self.sessions_dir / f"{year}_{round_num:02d}_S.parquet"
        sq_path = self.sessions_dir / f"{year}_{round_num:02d}_SQ.parquet"
        return sprint_path.exists() or sq_path.exists()


def main():
    """Command-line interface for data loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Load F1 FastF1 data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fastf1",
        help="Data directory (default: data/fastf1)",
    )
    parser.add_argument("--min-year", type=int, default=2020, help="Minimum year to load")
    parser.add_argument("--list-sessions", action="store_true", help="List available sessions")
    parser.add_argument("--summary", action="store_true", help="Show data summary")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Initialize loader
    loader = F1DataLoader(Path(args.data_dir))

    if args.list_sessions:
        sessions = loader.get_available_sessions()
        print(f"\nAvailable sessions ({len(sessions)}):")
        for session in sessions:
            print(f"  {session}")
        return

    if args.summary:
        years = loader.get_available_years()
        print("\nData Summary:")
        print(f"  Years: {years}")
        print(f"  Sessions: {len(loader.get_available_sessions())}")

        events = loader.load_events(min_year=args.min_year)
        if not events.empty:
            print(f"  Events: {len(events)}")

        drivers = loader.load_drivers()
        if not drivers.empty:
            print(f"  Drivers: {len(drivers)}")

        teams = loader.load_teams()
        if not teams.empty:
            print(f"  Teams: {len(teams)}")

        return

    # Default: load and show stats
    print("\nLoading data...")

    quali = loader.load_qualifying_results(min_year=args.min_year)
    print(f"Qualifying results: {len(quali)} rows")

    races = loader.load_race_results(min_year=args.min_year)
    print(f"Race results: {len(races)} rows")

    practice = loader.load_practice_sessions(min_year=args.min_year)
    print(f"Practice sessions: {len(practice)} rows")


if __name__ == "__main__":
    main()
