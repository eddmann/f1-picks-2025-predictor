"""
Tests for temporal leakage enforcement in feature engineering.

CRITICAL: All rolling window operations MUST use .shift(1) before .rolling()
to prevent data leakage. This test enforces that rule across all feature files.

Example of CORRECT pattern:
    df.groupby("driver_code")["position"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

Example of INCORRECT pattern (LEAKS DATA):
    df.groupby("driver_code")["position"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

The .shift(1) ensures we only use data from previous sessions, not the current one.
"""

from pathlib import Path

import pytest

# Feature files to check for temporal leakage
FEATURE_FILES_DIR = Path(__file__).parent.parent.parent / "src" / "features"


def get_feature_files() -> list[Path]:
    """Get all Python files in the features directory."""
    return list(FEATURE_FILES_DIR.glob("*.py"))


def find_rolling_without_shift(file_path: Path) -> list[tuple[int, str]]:
    """
    Find potential temporal leakage: .rolling() calls without preceding .shift(1).

    Returns:
        List of (line_number, line_content) tuples for suspicious patterns.
    """
    suspicious_lines = []

    with open(file_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        # Skip comments and strings
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Pattern 1: .rolling( without .shift( on same line or previous line
        if ".rolling(" in line:
            # Check current line for shift
            if ".shift(" in line:
                continue

            # Check if this is part of a multi-line statement
            # Look back up to 3 lines for .shift(
            found_shift = False
            for j in range(max(0, i - 4), i - 1):
                if ".shift(" in lines[j]:
                    found_shift = True
                    break

            if not found_shift:
                # Check if it's an allowed pattern (ewm, expanding, etc.)
                # or if it's explicitly commented as safe
                if "# temporal-safe" in line or "# no-shift-needed" in line:
                    continue

                # Check if this is for rank-based operations (position-based, not time-series)
                if "rank()" in line or ".rank(" in line:
                    continue

                suspicious_lines.append((i, line.rstrip()))

    return suspicious_lines


def find_ewm_without_shift(file_path: Path) -> list[tuple[int, str]]:
    """
    Find potential temporal leakage: .ewm() calls without preceding .shift(1).

    Exponentially weighted moving averages also need shift(1) for temporal safety.
    """
    suspicious_lines = []

    with open(file_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        if ".ewm(" in line:
            # Check current line and previous lines for shift
            if ".shift(" in line:
                continue

            found_shift = False
            for j in range(max(0, i - 4), i - 1):
                if ".shift(" in lines[j]:
                    found_shift = True
                    break

            if not found_shift:
                if "# temporal-safe" in line or "# no-shift-needed" in line:
                    continue
                suspicious_lines.append((i, line.rstrip()))

    return suspicious_lines


class TestTemporalLeakageEnforcement:
    """Automated tests to prevent temporal data leakage in features."""

    def test_all_feature_files_exist(self):
        """Verify feature files directory exists and has files."""
        assert FEATURE_FILES_DIR.exists(), f"Features directory not found: {FEATURE_FILES_DIR}"
        files = get_feature_files()
        assert len(files) > 0, "No feature files found"

    @pytest.mark.parametrize(
        "file_path",
        get_feature_files(),
        ids=lambda p: p.name,
    )
    def test_rolling_has_shift(self, file_path: Path):
        """
        CRITICAL: All .rolling() calls must be preceded by .shift(1).

        This test scans each feature file for .rolling() calls and verifies
        they have a .shift(1) to prevent temporal leakage.

        If this test fails, you have two options:
        1. Add .shift(1) before .rolling() (recommended for time-series features)
        2. Add '# temporal-safe' comment if the pattern is intentionally different
        """
        suspicious = find_rolling_without_shift(file_path)

        if suspicious:
            msg_lines = [
                f"\nPotential temporal leakage in {file_path.name}:",
                "Found .rolling() without preceding .shift(1):",
                "",
            ]
            for line_num, content in suspicious:
                msg_lines.append(f"  Line {line_num}: {content}")

            msg_lines.extend(
                [
                    "",
                    "FIX: Add .shift(1) before .rolling() to prevent data leakage:",
                    "  WRONG: x.rolling(3).mean()",
                    "  RIGHT: x.shift(1).rolling(3).mean()",
                    "",
                    "Or add '# temporal-safe' comment if intentionally different.",
                ]
            )

            pytest.fail("\n".join(msg_lines))

    @pytest.mark.parametrize(
        "file_path",
        get_feature_files(),
        ids=lambda p: p.name,
    )
    def test_ewm_has_shift(self, file_path: Path):
        """
        EWM (exponentially weighted) operations also need shift(1).

        Same principle as rolling windows - we can't use current session data.
        """
        suspicious = find_ewm_without_shift(file_path)

        if suspicious:
            msg_lines = [
                f"\nPotential temporal leakage in {file_path.name}:",
                "Found .ewm() without preceding .shift(1):",
                "",
            ]
            for line_num, content in suspicious:
                msg_lines.append(f"  Line {line_num}: {content}")

            msg_lines.extend(
                [
                    "",
                    "FIX: Add .shift(1) before .ewm() to prevent data leakage:",
                    "  WRONG: x.ewm(span=3).mean()",
                    "  RIGHT: x.shift(1).ewm(span=3).mean()",
                    "",
                    "Or add '# temporal-safe' comment if intentionally different.",
                ]
            )

            pytest.fail("\n".join(msg_lines))


class TestTemporalLeakageDocumentation:
    """Test that temporal safety is documented in feature files."""

    def test_base_pipeline_has_temporal_docs(self):
        """Base pipeline should document temporal safety requirements."""
        base_pipeline = FEATURE_FILES_DIR / "base_pipeline.py"
        assert base_pipeline.exists()

        content = base_pipeline.read_text()
        assert "shift" in content.lower() or "temporal" in content.lower(), (
            "base_pipeline.py should document temporal safety requirements"
        )

    def test_claude_md_has_temporal_docs(self):
        """CLAUDE.md should document the shift(1) requirement."""
        claude_md = Path(__file__).parent.parent.parent / "CLAUDE.md"
        assert claude_md.exists()

        content = claude_md.read_text()
        assert ".shift(1)" in content, "CLAUDE.md should document the .shift(1) requirement"
        assert ".rolling()" in content, "CLAUDE.md should mention rolling() operations"
