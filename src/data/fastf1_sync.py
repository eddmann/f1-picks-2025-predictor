"""
FastF1 data synchronization script.

Fetches F1 data from FastF1 and stores as parquet files with rich session data
including lap times, sector times, tyre compounds, and weather conditions.

Usage:
    python -m src.data.fastf1_sync --season 2025 --up-to-date
    python -m src.data.fastf1_sync --season 2024 --all
    python -m src.data.fastf1_sync --season 2025 --round 1
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import fastf1
import pandas as pd
from requests.exceptions import ConnectionError, HTTPError, Timeout

from src.config import config

logger = logging.getLogger(__name__)


class FastF1SyncError(Exception):
    """Base exception for FastF1 sync errors."""

    pass


class FastF1APIError(FastF1SyncError):
    """Raised when FastF1 API call fails."""

    def __init__(self, message: str, season: int | None = None, round_num: int | None = None):
        self.season = season
        self.round_num = round_num
        super().__init__(message)


class FastF1DataNotAvailableError(FastF1SyncError):
    """Raised when requested data is not yet available."""

    pass


# Session types to sync
PRACTICE_SESSIONS = ["FP1", "FP2", "FP3"]
QUALIFYING_SESSIONS = ["Q", "SQ"]  # SQ = Sprint Qualifying
RACE_SESSIONS = ["R", "S"]  # S = Sprint Race


class FastF1Sync:
    """Synchronizes F1 data from FastF1 to parquet format."""

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize sync with data directory.

        Args:
            data_dir: Path to data/fastf1/ directory (default: from config)
        """
        self.data_dir = Path(data_dir) if data_dir else config.data.data_dir
        self.sessions_dir = self.data_dir / "sessions"
        self.metadata_dir = self.data_dir / "metadata"

        # Create directories
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FastF1 cache
        cache_dir = config.data.cache_dir
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))

        logger.info(f"FastF1Sync initialized with data_dir: {self.data_dir}")

    def sync_season(
        self,
        season: int,
        round_num: int | None = None,
        up_to_date: bool = False,
        sync_all: bool = False,
        force: bool = False,
    ) -> dict:
        """
        Sync data for a season.

        Args:
            season: Season year (e.g., 2025)
            round_num: Specific round to sync, or None for all
            up_to_date: If True, only sync completed races up to today
            sync_all: If True, sync all historical races (ignore skip logic)
            force: If True, re-sync even if data exists

        Returns:
            Summary of synced data

        Raises:
            FastF1APIError: If unable to fetch schedule from FastF1 API
        """
        logger.info(f"Syncing season {season}...")

        # Get schedule with error handling
        try:
            schedule = fastf1.get_event_schedule(season)
        except (ConnectionError, Timeout) as e:
            raise FastF1APIError(
                f"Network error fetching {season} schedule. Check your internet connection.",
                season=season,
            ) from e
        except HTTPError as e:
            raise FastF1APIError(
                f"HTTP error fetching {season} schedule: {e}",
                season=season,
            ) from e
        except Exception as e:
            raise FastF1APIError(
                f"Failed to fetch {season} schedule: {e}. "
                "The season may not exist or FastF1 API may be unavailable.",
                season=season,
            ) from e

        schedule = schedule[schedule["RoundNumber"] > 0]  # Filter out testing

        if up_to_date:
            today = datetime.now()
            schedule = schedule[pd.to_datetime(schedule["EventDate"]) <= today]
            logger.info(f"Filtering to races up to {today.date()}")

        if round_num is not None:
            schedule = schedule[schedule["RoundNumber"] == round_num]

        if schedule.empty:
            logger.warning("No races to sync")
            return {"sessions_synced": 0, "sessions_skipped": 0}

        summary = {
            "sessions_synced": 0,
            "sessions_skipped": 0,
            "sessions_failed": 0,
        }

        # Update events metadata
        self._update_events_metadata(season, schedule)

        for _, event in schedule.iterrows():
            round_n = int(event["RoundNumber"])
            event_name = event["EventName"]

            # Determine which sessions to sync based on event format
            sessions_to_sync = self._get_sessions_for_event(event)

            for session_type in sessions_to_sync:
                session_key = f"{season}_{round_n:02d}_{session_type}"
                parquet_path = self.sessions_dir / f"{session_key}.parquet"

                # Skip if exists (unless force)
                if parquet_path.exists() and not force:
                    logger.debug(f"Skipping {session_key} - already exists")
                    summary["sessions_skipped"] += 1
                    continue

                try:
                    self._sync_session(season, round_n, session_type, event)
                    summary["sessions_synced"] += 1
                    logger.info(f"Synced {session_key} ({event_name})")
                except Exception as e:
                    summary["sessions_failed"] += 1
                    logger.warning(f"Failed to sync {session_key}: {e}")

        # Update driver and team metadata
        self._update_driver_metadata(season)
        self._update_team_metadata(season)

        logger.info(
            f"Sync complete: {summary['sessions_synced']} synced, "
            f"{summary['sessions_skipped']} skipped, {summary['sessions_failed']} failed"
        )
        return summary

    def _get_sessions_for_event(self, event: pd.Series) -> list[str]:
        """Determine which sessions to sync for an event."""
        sessions = []

        # Always try practice sessions (may not exist for all formats)
        sessions.extend(PRACTICE_SESSIONS)

        # Qualifying
        sessions.append("Q")

        # Sprint events have additional sessions
        event_format = event.get("EventFormat", "conventional")
        if event_format in ["sprint", "sprint_shootout", "sprint_qualifying"]:
            sessions.append("SQ")  # Sprint Qualifying
            sessions.append("S")  # Sprint Race

        # Race
        sessions.append("R")

        return sessions

    def _sync_session(
        self, season: int, round_num: int, session_type: str, event: pd.Series
    ) -> None:
        """
        Sync a single session to parquet.

        Args:
            season: Season year
            round_num: Round number
            session_type: Session type (FP1, FP2, FP3, Q, SQ, S, R)
            event: Event metadata

        Raises:
            FastF1APIError: If session cannot be fetched
            FastF1DataNotAvailable: If session data not yet available
        """
        session_key = f"{season}_{round_num:02d}_{session_type}"

        # Load session from FastF1 with error handling
        try:
            session = fastf1.get_session(season, round_num, session_type)
        except (ConnectionError, Timeout) as e:
            raise FastF1APIError(
                f"Network error fetching {session_key}. Check your internet connection.",
                season=season,
                round_num=round_num,
            ) from e
        except ValueError as e:
            # FastF1 raises ValueError for invalid session types
            raise FastF1DataNotAvailableError(
                f"Session {session_key} does not exist or is not available yet."
            ) from e
        except Exception as e:
            raise FastF1APIError(
                f"Failed to get session {session_key}: {e}",
                season=season,
                round_num=round_num,
            ) from e

        try:
            session.load(telemetry=False, weather=True, messages=False)
        except Exception as e:
            raise FastF1APIError(
                f"Failed to load data for {session_key}: {e}. Session may not have completed yet.",
                season=season,
                round_num=round_num,
            ) from e

        # Extract data
        session_data = self._extract_session_data(session, event, season, round_num)

        if session_data.empty:
            raise FastF1DataNotAvailableError(f"No data available for {session_key}")

        # Save to parquet
        parquet_path = self.sessions_dir / f"{session_key}.parquet"
        session_data.to_parquet(parquet_path, index=False)

    def _extract_session_data(
        self, session, event: pd.Series, season: int, round_num: int
    ) -> pd.DataFrame:
        """Extract comprehensive session data including laps."""
        session_key = f"{season}_{round_num:02d}_{session.name}"

        # Get lap data
        laps = session.laps

        if laps.empty:
            # No lap data - create minimal results-only DataFrame
            return self._extract_results_only(session, event, season, round_num)

        # Build comprehensive lap DataFrame
        lap_data = []

        for _, lap in laps.iterrows():
            lap_record = {
                # Session identifiers
                "session_key": session_key,
                "year": season,
                "round": round_num,
                "session_type": session.name,
                "circuit": event.get("Location", ""),
                "event_name": event.get("EventName", ""),
                # Driver info
                "driver_code": lap.get("Driver", ""),
                "driver_number": self._safe_int(lap.get("DriverNumber")),
                "team": lap.get("Team", ""),
                # Lap identifiers
                "lap_number": self._safe_int(lap.get("LapNumber")),
                "stint": self._safe_int(lap.get("Stint")),
                # Timing - convert to milliseconds for precision
                "lap_time_ms": self._timedelta_to_ms(lap.get("LapTime")),
                "sector1_time_ms": self._timedelta_to_ms(lap.get("Sector1Time")),
                "sector2_time_ms": self._timedelta_to_ms(lap.get("Sector2Time")),
                "sector3_time_ms": self._timedelta_to_ms(lap.get("Sector3Time")),
                # Speed traps
                "speed_i1": self._safe_float(lap.get("SpeedI1")),
                "speed_i2": self._safe_float(lap.get("SpeedI2")),
                "speed_fl": self._safe_float(lap.get("SpeedFL")),
                "speed_st": self._safe_float(lap.get("SpeedST")),
                # Tyre info
                "compound": lap.get("Compound", ""),
                "tyre_life": self._safe_int(lap.get("TyreLife")),
                "fresh_tyre": bool(lap.get("FreshTyre", False)),
                # Lap flags
                "is_personal_best": bool(lap.get("IsPersonalBest", False)),
                "deleted": bool(lap.get("Deleted", False)),
                "is_accurate": bool(lap.get("IsAccurate", True)),
                # Pit info
                "pit_in_time_ms": self._timedelta_to_ms(lap.get("PitInTime")),
                "pit_out_time_ms": self._timedelta_to_ms(lap.get("PitOutTime")),
            }
            lap_data.append(lap_record)

        df = pd.DataFrame(lap_data)

        # Add qualifying session split (Q1/Q2/Q3) for qualifying sessions
        if session.name == "Q":
            df = self._add_qualifying_session_split(df, session)

        # Add weather summary
        df = self._add_weather_data(df, session)

        # Add final position from results
        df = self._add_results_data(df, session)

        return df

    def _extract_results_only(
        self, session, event: pd.Series, season: int, round_num: int
    ) -> pd.DataFrame:
        """Extract results when no lap data available."""
        session_key = f"{season}_{round_num:02d}_{session.name}"
        results = session.results

        if results.empty:
            return pd.DataFrame()

        records = []
        for _, row in results.iterrows():
            record = {
                "session_key": session_key,
                "year": season,
                "round": round_num,
                "session_type": session.name,
                "circuit": event.get("Location", ""),
                "event_name": event.get("EventName", ""),
                "driver_code": row.get("Abbreviation", ""),
                "driver_number": self._safe_int(row.get("DriverNumber")),
                "team": row.get("TeamName", ""),
                "position": self._safe_int(row.get("Position")),
                "grid_position": self._safe_int(row.get("GridPosition")),
                "points": self._safe_float(row.get("Points")),
                "status": row.get("Status", ""),
                "q1_time_ms": self._timedelta_to_ms(row.get("Q1")),
                "q2_time_ms": self._timedelta_to_ms(row.get("Q2")),
                "q3_time_ms": self._timedelta_to_ms(row.get("Q3")),
            }
            records.append(record)

        return pd.DataFrame(records)

    def _add_qualifying_session_split(self, df: pd.DataFrame, session) -> pd.DataFrame:
        """Add Q1/Q2/Q3 session identifier to qualifying laps."""
        try:
            q1, q2, q3 = session.laps.split_qualifying_sessions()

            # Create mapping of lap index to quali session
            set(q1.index) if not q1.empty else set()
            set(q2.index) if not q2.empty else set()
            set(q3.index) if not q3.empty else set()

            # Map by approximate lap timing
            # Since we rebuilt the DataFrame, use lap numbers per driver
            df["quali_session"] = ""

            for driver in df["driver_code"].unique():
                driver_mask = df["driver_code"] == driver
                driver_laps = df[driver_mask].sort_values("lap_number")

                # Get driver's laps from each session
                if not q1.empty:
                    q1_driver = q1[q1["Driver"] == driver]
                    q1_lap_nums = set(q1_driver["LapNumber"].values)
                else:
                    q1_lap_nums = set()

                if not q2.empty:
                    q2_driver = q2[q2["Driver"] == driver]
                    q2_lap_nums = set(q2_driver["LapNumber"].values)
                else:
                    q2_lap_nums = set()

                if not q3.empty:
                    q3_driver = q3[q3["Driver"] == driver]
                    q3_lap_nums = set(q3_driver["LapNumber"].values)
                else:
                    q3_lap_nums = set()

                for idx, row in driver_laps.iterrows():
                    lap_num = row["lap_number"]
                    if lap_num in q1_lap_nums:
                        df.loc[idx, "quali_session"] = "Q1"
                    elif lap_num in q2_lap_nums:
                        df.loc[idx, "quali_session"] = "Q2"
                    elif lap_num in q3_lap_nums:
                        df.loc[idx, "quali_session"] = "Q3"

        except Exception as e:
            logger.warning(f"Could not split qualifying sessions: {e}")
            df["quali_session"] = ""

        return df

    def _add_weather_data(self, df: pd.DataFrame, session) -> pd.DataFrame:
        """Add weather summary to session data."""
        try:
            weather = session.weather_data
            if weather is not None and not weather.empty:
                # Calculate session averages
                df["avg_air_temp"] = weather["AirTemp"].mean()
                df["avg_track_temp"] = weather["TrackTemp"].mean()
                df["rainfall"] = weather["Rainfall"].any()
                df["avg_humidity"] = weather["Humidity"].mean()
            else:
                df["avg_air_temp"] = None
                df["avg_track_temp"] = None
                df["rainfall"] = False
                df["avg_humidity"] = None
        except Exception as e:
            logger.debug(f"Could not extract weather data: {e}")
            df["avg_air_temp"] = None
            df["avg_track_temp"] = None
            df["rainfall"] = False
            df["avg_humidity"] = None

        return df

    def _add_results_data(self, df: pd.DataFrame, session) -> pd.DataFrame:
        """Add final position and points from results."""
        results = session.results

        if results.empty:
            df["position"] = None
            df["grid_position"] = None
            df["points"] = None
            df["status"] = ""
            return df

        # Create driver -> result mapping
        result_map = {}
        for _, row in results.iterrows():
            driver = row.get("Abbreviation", "")
            result_map[driver] = {
                "position": self._safe_int(row.get("Position")),
                "grid_position": self._safe_int(row.get("GridPosition")),
                "points": self._safe_float(row.get("Points")),
                "status": row.get("Status", ""),
                "q1_time_ms": self._timedelta_to_ms(row.get("Q1")),
                "q2_time_ms": self._timedelta_to_ms(row.get("Q2")),
                "q3_time_ms": self._timedelta_to_ms(row.get("Q3")),
            }

        # Apply to DataFrame
        df["position"] = df["driver_code"].map(lambda x: result_map.get(x, {}).get("position"))
        df["grid_position"] = df["driver_code"].map(
            lambda x: result_map.get(x, {}).get("grid_position")
        )
        df["points"] = df["driver_code"].map(lambda x: result_map.get(x, {}).get("points"))
        df["status"] = df["driver_code"].map(lambda x: result_map.get(x, {}).get("status", ""))

        # Add Q times for qualifying sessions
        if session.name == "Q":
            df["q1_time_ms"] = df["driver_code"].map(
                lambda x: result_map.get(x, {}).get("q1_time_ms")
            )
            df["q2_time_ms"] = df["driver_code"].map(
                lambda x: result_map.get(x, {}).get("q2_time_ms")
            )
            df["q3_time_ms"] = df["driver_code"].map(
                lambda x: result_map.get(x, {}).get("q3_time_ms")
            )

        return df

    def _update_events_metadata(self, season: int, schedule: pd.DataFrame) -> None:
        """Update events metadata file."""
        events_path = self.metadata_dir / "events.parquet"

        # Load existing or create new
        if events_path.exists():
            existing = pd.read_parquet(events_path)
        else:
            existing = pd.DataFrame()

        # Build new events data
        new_events = []
        for _, event in schedule.iterrows():
            new_events.append(
                {
                    "year": season,
                    "round": int(event["RoundNumber"]),
                    "event_name": event["EventName"],
                    "circuit": event.get("Location", ""),
                    "country": event.get("Country", ""),
                    "event_date": pd.to_datetime(event["EventDate"]),
                    "event_format": event.get("EventFormat", "conventional"),
                }
            )

        new_df = pd.DataFrame(new_events)

        # Merge with existing (update if exists)
        if not existing.empty:
            # Remove existing entries for this season
            existing = existing[existing["year"] != season]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined = combined.sort_values(["year", "round"]).reset_index(drop=True)
        combined.to_parquet(events_path, index=False)

    def _update_driver_metadata(self, season: int) -> None:
        """Update drivers metadata from synced sessions."""
        drivers_path = self.metadata_dir / "drivers.parquet"

        # Collect all unique drivers from this season's sessions
        drivers = {}
        for parquet_file in self.sessions_dir.glob(f"{season}_*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                for _, row in df.drop_duplicates("driver_code").iterrows():
                    code = row.get("driver_code", "")
                    if code and code not in drivers:
                        drivers[code] = {
                            "driver_code": code,
                            "driver_number": row.get("driver_number"),
                            "team": row.get("team", ""),
                            "first_seen_year": season,
                        }
            except Exception:
                continue

        if not drivers:
            return

        # Load existing
        if drivers_path.exists():
            existing = pd.read_parquet(drivers_path)
            existing_codes = set(existing["driver_code"])

            # Only add new drivers
            new_drivers = [d for c, d in drivers.items() if c not in existing_codes]
            if new_drivers:
                combined = pd.concat([existing, pd.DataFrame(new_drivers)], ignore_index=True)
                combined.to_parquet(drivers_path, index=False)
        else:
            pd.DataFrame(list(drivers.values())).to_parquet(drivers_path, index=False)

    def _update_team_metadata(self, season: int) -> None:
        """Update teams metadata from synced sessions."""
        teams_path = self.metadata_dir / "teams.parquet"

        # Collect all unique teams from this season's sessions
        teams = set()
        for parquet_file in self.sessions_dir.glob(f"{season}_*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                teams.update(df["team"].dropna().unique())
            except Exception:
                continue

        if not teams:
            return

        # Load existing
        if teams_path.exists():
            existing = pd.read_parquet(teams_path)
            existing_teams = set(existing["team"])
            new_teams = teams - existing_teams
            if new_teams:
                new_df = pd.DataFrame({"team": list(new_teams)})
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined.to_parquet(teams_path, index=False)
        else:
            pd.DataFrame({"team": list(teams)}).to_parquet(teams_path, index=False)

    @staticmethod
    def _timedelta_to_ms(td) -> int | None:
        """Convert timedelta to milliseconds."""
        if pd.isna(td):
            return None
        try:
            return int(td.total_seconds() * 1000)
        except Exception:
            return None

    @staticmethod
    def _safe_int(val) -> int | None:
        """Safely convert to int."""
        if pd.isna(val):
            return None
        try:
            return int(val)
        except Exception:
            return None

    def _safe_float(self, val) -> float | None:
        """Safely convert to float."""
        if pd.isna(val):
            return None
        try:
            return float(val)
        except Exception:
            return None

    def session_exists(self, year: int, round_num: int, session_type: str) -> bool:
        """Check if a session has already been synced."""
        session_key = f"{year}_{round_num:02d}_{session_type}"
        return (self.sessions_dir / f"{session_key}.parquet").exists()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sync F1 data from FastF1 to parquet format")
    parser.add_argument("--season", type=int, required=True, help="Season year (e.g., 2025)")
    parser.add_argument("--round", type=int, default=None, help="Specific round number to sync")
    parser.add_argument(
        "--up-to-date",
        action="store_true",
        help="Only sync completed races up to today",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="sync_all",
        help="Sync all races in the season",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-sync even if data exists",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Data directory (default: {config.data.data_dir})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Suppress FastF1 info messages unless verbose
    if not args.verbose:
        logging.getLogger("fastf1").setLevel(logging.WARNING)

    # Run sync
    data_dir = Path(args.data_dir) if args.data_dir else None
    syncer = FastF1Sync(data_dir)
    summary = syncer.sync_season(
        season=args.season,
        round_num=args.round,
        up_to_date=args.up_to_date,
        sync_all=args.sync_all,
        force=args.force,
    )

    print("\n" + "=" * 50)
    print("SYNC COMPLETE")
    print("=" * 50)
    print(f"Sessions synced: {summary['sessions_synced']}")
    print(f"Sessions skipped: {summary['sessions_skipped']}")
    print(f"Sessions failed: {summary['sessions_failed']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
