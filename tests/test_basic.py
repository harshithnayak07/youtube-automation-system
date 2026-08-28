"""Basic smoke tests for project structure."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_project_root_exists():
    assert ROOT.is_dir()


def test_core_constants_importable():
    from core import constants

    assert constants.PROJECT_ROOT.is_dir()


def test_config_settings_importable():
    from config import settings

    assert hasattr(settings, "LOG_LEVEL")
    assert hasattr(settings, "GROQ_API_KEY")
    assert hasattr(settings, "OPENROUTER_API_KEY")


def test_storage_init_db(tmp_path):
    import os
    from importlib import reload
    os.environ["DATABASE_PATH"] = str(tmp_path / "test.db")

    from config import settings
    reload(settings)

    from storage import database
    reload(database)
    database.init_db()
    conn = database.get_connection()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    table_names = [t["name"] for t in tables]
    assert "jobs" in table_names


def test_modules_exist():
    expected_dirs = [
        "research", "llm", "content", "media",
        "video", "youtube", "notifications", "core",
        "config", "storage", "tests", "jobs", "output", "logs",
    ]
    for d in expected_dirs:
        assert (ROOT / d).is_dir(), f"Missing directory: {d}"


def test_env_example_exists():
    assert (ROOT / ".env.example").is_file()


def test_gitignore_exists():
    assert (ROOT / ".gitignore").is_file()


def test_requirements_exists():
    assert (ROOT / "requirements.txt").is_file()


def test_main_py_exists():
    assert (ROOT / "main.py").is_file()


# ---------------------------------------------------------------------------
# main.py — entry point behavior
# ---------------------------------------------------------------------------

class TestMainEntryPoint:
    def test_returns_zero_on_pipeline_success(self, monkeypatch):
        from unittest.mock import MagicMock
        from core.pipeline import JobResult, PipelineStage

        mock_result = JobResult(
            job_id="job-test",
            success=True,
            stage=PipelineStage.COMPLETE,
        )
        monkeypatch.setattr("core.pipeline.run_pipeline", lambda: mock_result)

        from main import main
        assert main() == 0

    def test_returns_one_on_pipeline_failure(self, monkeypatch):
        from unittest.mock import MagicMock
        from core.pipeline import JobResult, PipelineStage

        mock_result = JobResult(
            job_id="job-test",
            success=False,
            stage=PipelineStage.IMAGES,
            error="[images] RuntimeError: API down",
        )
        monkeypatch.setattr("core.pipeline.run_pipeline", lambda: mock_result)

        from main import main
        assert main() == 1

    def test_returns_one_on_unexpected_exception(self, monkeypatch):
        monkeypatch.setattr(
            "core.pipeline.run_pipeline",
            lambda: (_ for _ in ()).throw(RuntimeError("crash")),
        )

        from main import main
        assert main() == 1

    def test_imports_run_pipeline(self):
        from main import main
        import inspect
        source = inspect.getsource(main)
        assert "run_pipeline" in source
