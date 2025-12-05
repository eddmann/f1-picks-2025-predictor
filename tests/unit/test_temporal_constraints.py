"""
Tests for temporal constraints in feature pipelines.

Each session type has different data availability based on the weekend format:
- 2024 format: FP1 -> SQ -> Sprint Race -> FP2 -> FP3 -> Qualifying -> Race

Constraints:
- Q: FP1, FP2, FP3 available (all happen before Q)
- SQ: FP1 only (FP2, FP3, Q happen after SQ)
- S: FP1, SQ grid (FP2, FP3, Q happen after S)
- R: All sessions available (FP1-3, Q, SQ, S)
"""

from src.features.qualifying_pipeline import QualifyingFeaturePipeline
from src.features.race_pipeline import RaceFeaturePipeline
from src.features.sprint_quali_pipeline import SprintQualiFeaturePipeline
from src.features.sprint_race_pipeline import SprintRaceFeaturePipeline


class TestQualifyingPipelineTemporal:
    """Test QualifyingFeaturePipeline uses correct sessions."""

    def test_available_practice_sessions(self):
        """Qualifying can use all practice sessions (FP1, FP2, FP3)."""
        pipeline = QualifyingFeaturePipeline()
        sessions = pipeline.get_available_practice_sessions()

        assert "FP1" in sessions
        assert "FP2" in sessions
        assert "FP3" in sessions
        assert len(sessions) == 3

    def test_session_type(self):
        """Qualifying pipeline targets Q sessions."""
        pipeline = QualifyingFeaturePipeline()
        assert pipeline.session_type == "Q"


class TestSprintQualiPipelineTemporal:
    """Test SprintQualiFeaturePipeline only uses FP1."""

    def test_only_fp1_available(self):
        """Sprint qualifying can only use FP1 (FP2/FP3 happen after SQ)."""
        pipeline = SprintQualiFeaturePipeline()
        sessions = pipeline.get_available_practice_sessions()

        assert sessions == ["FP1"], "SQ should only have FP1 available"

    def test_no_fp2_fp3_available(self):
        """Sprint qualifying must NOT use FP2 or FP3 data."""
        pipeline = SprintQualiFeaturePipeline()
        sessions = pipeline.get_available_practice_sessions()

        assert "FP2" not in sessions, "FP2 happens AFTER SQ - cannot use"
        assert "FP3" not in sessions, "FP3 happens AFTER SQ - cannot use"

    def test_no_qualifying_available(self):
        """Sprint qualifying must NOT use main qualifying data."""
        pipeline = SprintQualiFeaturePipeline()
        weekend_sessions = pipeline.get_available_current_weekend_sessions()

        # SQ happens BEFORE main Q, so Q data is not available
        assert "Q" not in weekend_sessions, "Main Q happens AFTER SQ - cannot use"

    def test_session_type(self):
        """Sprint qualifying pipeline targets SQ sessions."""
        pipeline = SprintQualiFeaturePipeline()
        assert pipeline.session_type == "SQ"


class TestSprintRacePipelineTemporal:
    """Test SprintRaceFeaturePipeline uses FP1 + SQ grid."""

    def test_only_fp1_available(self):
        """Sprint race can only use FP1 (FP2/FP3 happen after sprint race)."""
        pipeline = SprintRaceFeaturePipeline()
        sessions = pipeline.get_available_practice_sessions()

        assert sessions == ["FP1"], "Sprint race should only have FP1 available"

    def test_no_fp2_fp3_available(self):
        """Sprint race must NOT use FP2 or FP3 data."""
        pipeline = SprintRaceFeaturePipeline()
        sessions = pipeline.get_available_practice_sessions()

        assert "FP2" not in sessions, "FP2 happens AFTER sprint race - cannot use"
        assert "FP3" not in sessions, "FP3 happens AFTER sprint race - cannot use"

    def test_sq_grid_available(self):
        """Sprint race can use SQ results (sets sprint race grid)."""
        pipeline = SprintRaceFeaturePipeline()
        weekend_sessions = pipeline.get_available_current_weekend_sessions()

        assert "SQ" in weekend_sessions, "SQ results (sprint grid) should be available"

    def test_no_qualifying_available(self):
        """Sprint race must NOT use main qualifying data."""
        pipeline = SprintRaceFeaturePipeline()
        weekend_sessions = pipeline.get_available_current_weekend_sessions()

        # Sprint race happens BEFORE main Q
        assert "Q" not in weekend_sessions, "Main Q happens AFTER sprint race - cannot use"

    def test_session_type(self):
        """Sprint race pipeline targets S sessions."""
        pipeline = SprintRaceFeaturePipeline()
        assert pipeline.session_type == "S"


class TestRacePipelineTemporal:
    """Test RaceFeaturePipeline uses all available sessions."""

    def test_all_practice_sessions_available(self):
        """Race can use all practice sessions (all happen before race)."""
        pipeline = RaceFeaturePipeline()
        sessions = pipeline.get_available_practice_sessions()

        assert "FP1" in sessions
        assert "FP2" in sessions
        assert "FP3" in sessions

    def test_qualifying_grid_available(self):
        """Race can use qualifying results (sets race grid)."""
        pipeline = RaceFeaturePipeline()
        weekend_sessions = pipeline.get_available_current_weekend_sessions()

        assert "Q" in weekend_sessions, "Qualifying grid should be available for race"

    def test_sprint_data_available(self):
        """Race can use sprint weekend data if available."""
        pipeline = RaceFeaturePipeline()
        weekend_sessions = pipeline.get_available_current_weekend_sessions()

        # Both SQ and S happen before main race
        assert "SQ" in weekend_sessions or "S" in weekend_sessions, (
            "Sprint data should be available for race (on sprint weekends)"
        )

    def test_session_type(self):
        """Race pipeline targets R sessions."""
        pipeline = RaceFeaturePipeline()
        assert pipeline.session_type == "R"


class TestTemporalConsistency:
    """Test consistency across all pipelines."""

    def test_only_race_has_all_practice(self):
        """Only Race pipeline should have FP2/FP3 available."""
        q_pipeline = QualifyingFeaturePipeline()
        sq_pipeline = SprintQualiFeaturePipeline()
        s_pipeline = SprintRaceFeaturePipeline()
        r_pipeline = RaceFeaturePipeline()

        # Qualifying also has all practice sessions (happens after FP3)
        assert "FP2" in q_pipeline.get_available_practice_sessions()
        assert "FP3" in q_pipeline.get_available_practice_sessions()

        # Sprint sessions only have FP1
        assert "FP2" not in sq_pipeline.get_available_practice_sessions()
        assert "FP3" not in sq_pipeline.get_available_practice_sessions()
        assert "FP2" not in s_pipeline.get_available_practice_sessions()
        assert "FP3" not in s_pipeline.get_available_practice_sessions()

        # Race has all
        assert "FP2" in r_pipeline.get_available_practice_sessions()
        assert "FP3" in r_pipeline.get_available_practice_sessions()

    def test_only_race_has_qualifying_grid(self):
        """Only Race pipeline should have qualifying grid available."""
        sq_pipeline = SprintQualiFeaturePipeline()
        s_pipeline = SprintRaceFeaturePipeline()
        r_pipeline = RaceFeaturePipeline()

        # SQ and S don't have Q grid
        assert "Q" not in sq_pipeline.get_available_current_weekend_sessions()
        assert "Q" not in s_pipeline.get_available_current_weekend_sessions()

        # Race has Q grid
        assert "Q" in r_pipeline.get_available_current_weekend_sessions()

    def test_sprint_race_has_sq_grid(self):
        """Sprint race should have SQ grid, SQ should not."""
        sq_pipeline = SprintQualiFeaturePipeline()
        s_pipeline = SprintRaceFeaturePipeline()

        # SQ doesn't have its own results yet
        assert "SQ" not in sq_pipeline.get_available_current_weekend_sessions()

        # Sprint race has SQ results (its grid)
        assert "SQ" in s_pipeline.get_available_current_weekend_sessions()


class TestWeekendFormatValidation:
    """Test that pipelines correctly model 2024 weekend format."""

    def test_2024_format_order(self):
        """
        2024 weekend format: FP1 -> SQ -> Sprint Race -> FP2 -> FP3 -> Q -> R

        Each session should only access data from sessions that happen before it.
        """
        # FP1 is first, available to all
        for pipeline_cls in [
            QualifyingFeaturePipeline,
            SprintQualiFeaturePipeline,
            SprintRaceFeaturePipeline,
            RaceFeaturePipeline,
        ]:
            pipeline = pipeline_cls()
            assert "FP1" in pipeline.get_available_practice_sessions()

        # SQ is second, only FP1 available
        sq = SprintQualiFeaturePipeline()
        assert sq.get_available_practice_sessions() == ["FP1"]

        # Sprint race is third, FP1 + SQ grid available
        s = SprintRaceFeaturePipeline()
        assert s.get_available_practice_sessions() == ["FP1"]
        assert "SQ" in s.get_available_current_weekend_sessions()

        # Q is sixth (after FP1, SQ, S, FP2, FP3), all practice available
        q = QualifyingFeaturePipeline()
        assert "FP1" in q.get_available_practice_sessions()
        assert "FP2" in q.get_available_practice_sessions()
        assert "FP3" in q.get_available_practice_sessions()

        # R is last, everything available
        r = RaceFeaturePipeline()
        assert "FP1" in r.get_available_practice_sessions()
        assert "FP2" in r.get_available_practice_sessions()
        assert "FP3" in r.get_available_practice_sessions()
        assert "Q" in r.get_available_current_weekend_sessions()
