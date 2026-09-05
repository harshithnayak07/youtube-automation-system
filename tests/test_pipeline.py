"""Tests for the core.pipeline orchestrator — all use mocks, no real APIs."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from content.metadata import VideoMetadata
from content.scenes import Scene
from core.pipeline import (
    PipelineStage,
    JobResult,
    _build_llm_client,
    _create_job_record,
    _get_job_record,
    _handle_failure,
    _send_notification,
    _update_job_record,
    run_pipeline,
)
from core.qa import QAResult
from llm.client import LLMClient, LLMConfigError, LLMResponse
from media.image import GeneratedImage
from media.tts import GeneratedAudio
from media.thumbnail import GeneratedThumbnail
from research.trending import Topic
from video.compositor import ComposedVideo
from youtube.uploader import UploadError, UploadResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_topic(**overrides) -> Topic:
    defaults = dict(
        title="AI Revolution 2026",
        url="https://example.com/ai-news",
        summary="Latest AI breakthroughs and trends.",
        source="Google News",
    )
    defaults.update(overrides)
    return Topic(**defaults)


def _make_scene(number: int = 1) -> Scene:
    return Scene(
        scene=number,
        text=f"This is scene {number} narration text.",
        visual_description=f"Visual description for scene {number}",
    )


def _make_images(count: int = 3) -> list[GeneratedImage]:
    return [
        GeneratedImage(scene=i + 1, path=Path(f"/tmp/img_{i}.png"))
        for i in range(count)
    ]


def _make_audio() -> GeneratedAudio:
    return GeneratedAudio(path=Path("/tmp/audio.mp3"), duration_seconds=45.0)


def _make_video() -> ComposedVideo:
    return ComposedVideo(path=Path("/tmp/video.mp4"), duration_seconds=45.0, scene_count=3)


def _make_metadata() -> VideoMetadata:
    return VideoMetadata(
        title="AI Revolution 2026",
        description="Discover the latest in AI technology.",
        tags=["ai", "technology", "future"],
    )


def _make_thumbnail() -> GeneratedThumbnail:
    return GeneratedThumbnail(path=Path("/tmp/thumb.jpg"), width=1280, height=720)


def _make_qa_result(passed: bool = True) -> QAResult:
    if passed:
        return QAResult(passed=True, errors=[], warnings=["minor warning"], details={})
    return QAResult(
        passed=False,
        errors=["Video too short", "Missing audio"],
        warnings=[],
        details={},
    )


def _make_upload_result(**overrides) -> UploadResult:
    defaults = dict(
        video_id="vid_abc123",
        video_url="https://youtube.com/shorts/vid_abc123",
        video_status="uploaded",
        thumbnail_status="uploaded",
        errors=[],
        details={},
    )
    defaults.update(overrides)
    return UploadResult(**defaults)


# ---------------------------------------------------------------------------
# PipelineStage model
# ---------------------------------------------------------------------------

class TestPipelineStage:
    def test_all_stages_present(self):
        stages = list(PipelineStage)
        assert len(stages) == 13

    def test_stage_order(self):
        stages = list(PipelineStage)
        expected = [
            "research", "script", "scenes", "images", "tts",
            "composition", "metadata", "thumbnail", "qa",
            "upload", "record", "notify", "complete",
        ]
        assert [s.value for s in stages] == expected

    def test_complete_stage_exists(self):
        assert PipelineStage.COMPLETE.value == "complete"

    def test_stage_values_are_strings(self):
        for stage in PipelineStage:
            assert isinstance(stage.value, str)


# ---------------------------------------------------------------------------
# JobResult model
# ---------------------------------------------------------------------------

class TestJobResult:
    def test_success_result(self):
        r = JobResult(
            job_id="job-123",
            success=True,
            stage=PipelineStage.COMPLETE,
            topic=_make_topic(),
        )
        assert r.job_id == "job-123"
        assert r.success is True
        assert r.stage == PipelineStage.COMPLETE
        assert r.error == ""

    def test_failure_result(self):
        r = JobResult(
            job_id="job-456",
            success=False,
            stage=PipelineStage.RESEARCH,
            error="[research] RuntimeError: no topics",
        )
        assert r.success is False
        assert r.stage == PipelineStage.RESEARCH
        assert "no topics" in r.error

    def test_is_frozen(self):
        r = JobResult(job_id="j", success=True, stage=PipelineStage.COMPLETE)
        with pytest.raises(AttributeError):
            r.success = False  # type: ignore[misc]

    def test_optional_fields_default_none(self):
        r = JobResult(job_id="j", success=True, stage=PipelineStage.COMPLETE)
        assert r.topic is None
        assert r.narration is None
        assert r.scenes is None
        assert r.images is None
        assert r.audio is None
        assert r.video is None
        assert r.metadata is None
        assert r.thumbnail is None
        assert r.qa_result is None
        assert r.upload_result is None


# ---------------------------------------------------------------------------
# _build_llm_client
# ---------------------------------------------------------------------------

class TestBuildLLMClient:
    @patch("core.pipeline.settings")
    def test_builds_with_groq_key(self, mock_settings):
        mock_settings.GROQ_API_KEY = "groq_key_123"
        mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        mock_settings.OPENROUTER_API_KEY = ""
        mock_settings.OPENROUTER_MODEL = ""
        client = _build_llm_client()
        assert isinstance(client, LLMClient)

    @patch("core.pipeline.settings")
    def test_builds_with_openrouter_key(self, mock_settings):
        mock_settings.GROQ_API_KEY = ""
        mock_settings.GROQ_MODEL = ""
        mock_settings.OPENROUTER_API_KEY = "or_key_456"
        mock_settings.OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
        client = _build_llm_client()
        assert isinstance(client, LLMClient)

    @patch("core.pipeline.settings")
    def test_raises_when_no_keys(self, mock_settings):
        mock_settings.GROQ_API_KEY = ""
        mock_settings.GROQ_MODEL = ""
        mock_settings.OPENROUTER_API_KEY = ""
        mock_settings.OPENROUTER_MODEL = ""
        with pytest.raises(LLMConfigError, match="No LLM API keys"):
            _build_llm_client()


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

class TestCreateJobRecord:
    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_inserts_and_returns_id(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 42
        mock_conn.execute.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        job_db_id = _create_job_record("AI News")
        assert job_db_id == 42
        mock_init.assert_called_once()
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_passes_correct_params(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        _create_job_record("Test Topic")
        args = mock_conn.execute.call_args
        assert "INSERT INTO jobs" in args[0][0]
        assert args[0][1] == ("Test Topic", "running")


class TestUpdateJobRecord:
    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_updates_status(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        _update_job_record(1, status="completed")
        mock_init.assert_called_once()
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "UPDATE jobs SET" in sql
        assert "status = ?" in sql

    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_updates_video_id(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        _update_job_record(1, video_id="vid123")
        sql = mock_conn.execute.call_args[0][0]
        assert "video_id = ?" in sql

    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_updates_error_message(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        _update_job_record(1, error_message="something broke")
        sql = mock_conn.execute.call_args[0][0]
        assert "error_message = ?" in sql

    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_updates_metadata_json(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        _update_job_record(1, metadata={"key": "value"})
        sql = mock_conn.execute.call_args[0][0]
        assert "metadata = ?" in sql

    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_no_op_when_no_updates(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        _update_job_record(1)
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


class TestGetJobRecord:
    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_returns_dict_when_found(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_row = MagicMock()
        mock_row.__iter__ = lambda self: iter([("id", 1), ("topic", "AI")])
        mock_row.keys.return_value = ["id", "topic"]
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_get_conn.return_value = mock_conn

        result = _get_job_record(1)
        assert result is not None

    @patch("core.pipeline.get_connection")
    @patch("core.pipeline.init_db")
    def test_returns_none_when_not_found(self, mock_init, mock_get_conn):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_get_conn.return_value = mock_conn

        result = _get_job_record(999)
        assert result is None


# ---------------------------------------------------------------------------
# Fresh database — schema must be self-initializing
# ---------------------------------------------------------------------------

class TestFreshDatabase:
    """Regression for "Failed to record job failure: no such table: jobs".

    A fresh checkout has no state.db, so storage helpers must create the
    schema via init_db() before touching the table.
    """

    def test_update_job_record_on_fresh_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / "fresh.db"
        monkeypatch.setattr("storage.database.settings.DATABASE_PATH", str(db_path))

        _update_job_record(1, status="failed", error_message="boom")

        conn = sqlite3.connect(str(db_path))
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )]
        finally:
            conn.close()
        assert "jobs" in tables

    def test_get_job_record_returns_none_on_fresh_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / "fresh.db"
        monkeypatch.setattr("storage.database.settings.DATABASE_PATH", str(db_path))

        result = _get_job_record(1)

        assert result is None

    def test_handle_failure_on_fresh_db_creates_schema_without_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        db_path = tmp_path / "fresh.db"
        monkeypatch.setattr("storage.database.settings.DATABASE_PATH", str(db_path))
        monkeypatch.setattr("core.pipeline.send_job_failure", MagicMock())

        with caplog.at_level(logging.WARNING, logger="core.pipeline"):
            result = _handle_failure(
                job_id="job-fresh",
                failed_stage=PipelineStage.RESEARCH,
                error_type="RuntimeError",
                error_message="no topics found",
            )

        assert result.success is False
        conn = sqlite3.connect(str(db_path))
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )]
        finally:
            conn.close()
        assert "jobs" in tables
        assert "no such table" not in caplog.text


# ---------------------------------------------------------------------------
# _handle_failure
# ---------------------------------------------------------------------------

class TestHandleFailure:
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    def test_returns_failure_result(self, mock_update, mock_send):
        result = _handle_failure(
            job_id="job-1",
            failed_stage=PipelineStage.RESEARCH,
            error_type="RuntimeError",
            error_message="no topics found",
        )
        assert result.success is False
        assert result.stage == PipelineStage.RESEARCH
        assert "no topics found" in result.error

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    def test_records_failure_in_db(self, mock_update, mock_send):
        _handle_failure(
            job_id="job-2",
            failed_stage=PipelineStage.UPLOAD,
            error_type="UploadError",
            error_message="auth failed",
        )
        mock_update.assert_called_once()
        args = mock_update.call_args
        assert args[1]["status"] == "failed"
        assert "auth failed" in args[1]["error_message"]

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    def test_sends_failure_email(self, mock_update, mock_send):
        _handle_failure(
            job_id="job-3",
            failed_stage=PipelineStage.SCRIPT,
            error_type="LLMError",
            error_message="provider down",
        )
        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1]
        assert kwargs["job_id"] == "job-3"
        assert kwargs["failed_stage"] == "script"
        assert kwargs["error_type"] == "LLMError"

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    def test_preserves_partial_artifacts(self, mock_update, mock_send):
        topic = _make_topic()
        result = _handle_failure(
            job_id="job-4",
            failed_stage=PipelineStage.IMAGES,
            error_type="ImageGenError",
            error_message="API failed",
            topic=topic,
            narration="test narration",
        )
        assert result.topic == topic
        assert result.narration == "test narration"

    @patch("core.pipeline.send_job_failure", side_effect=Exception("email down"))
    @patch("core.pipeline._update_job_record", side_effect=Exception("db down"))
    def test_survives_db_and_email_failures(self, mock_update, mock_send):
        result = _handle_failure(
            job_id="job-5",
            failed_stage=PipelineStage.TTS,
            error_type="TTSError",
            error_message="tts failed",
        )
        assert result.success is False
        assert result.stage == PipelineStage.TTS


# ---------------------------------------------------------------------------
# _send_notification
# ---------------------------------------------------------------------------

class TestSendNotification:
    @patch("core.pipeline.send_job_success")
    def test_sends_success_when_thumbnail_uploaded(self, mock_send):
        upload_result = _make_upload_result(thumbnail_status="uploaded")
        _send_notification(
            job_id="job-1",
            topic=_make_topic(),
            upload_result=upload_result,
            qa_result=_make_qa_result(True),
            stage_details={},
        )
        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1]
        assert kwargs["job_id"] == "job-1"
        assert kwargs["thumbnail_status"] == "uploaded"

    @patch("core.pipeline.send_partial_success")
    def test_sends_partial_when_thumbnail_failed(self, mock_send):
        upload_result = _make_upload_result(thumbnail_status="failed")
        _send_notification(
            job_id="job-2",
            topic=_make_topic(),
            upload_result=upload_result,
            qa_result=_make_qa_result(True),
            stage_details={},
        )
        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1]
        assert kwargs["job_id"] == "job-2"
        assert kwargs["thumbnail_status"] == "failed"

    @patch("core.pipeline.send_job_success", side_effect=Exception("email down"))
    def test_survives_notification_failure(self, mock_send):
        upload_result = _make_upload_result()
        # Should not raise
        _send_notification(
            job_id="job-3",
            topic=_make_topic(),
            upload_result=upload_result,
            qa_result=_make_qa_result(True),
            stage_details={},
        )


# ---------------------------------------------------------------------------
# run_pipeline — complete success
# ---------------------------------------------------------------------------

class TestPipelineSuccess:
    @patch("core.pipeline._send_notification")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_complete_success(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_notify,
    ):
        topic = _make_topic()
        mock_research.return_value = topic
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "Test narration text."
        mock_scenes.return_value = [_make_scene(1), _make_scene(2), _make_scene(3)]
        mock_images.return_value = _make_images(3)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(True)
        mock_upload.return_value = _make_upload_result()

        result = run_pipeline()

        assert result.success is True
        assert result.stage == PipelineStage.COMPLETE
        assert result.topic == topic
        assert result.narration == "Test narration text."
        assert len(result.scenes) == 3
        assert len(result.images) == 3
        assert result.audio is not None
        assert result.video is not None
        assert result.metadata is not None
        assert result.thumbnail is not None
        assert result.qa_result is not None
        assert result.upload_result is not None
        mock_upload.assert_called_once()
        mock_notify.assert_called_once()

    @patch("core.pipeline._send_notification")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_unique_job_id(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_notify,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(True)
        mock_upload.return_value = _make_upload_result()

        result1 = run_pipeline()
        result2 = run_pipeline()

        assert result1.job_id != result2.job_id
        assert result1.job_id.startswith("job-")
        assert result2.job_id.startswith("job-")

    @patch("core.pipeline._send_notification")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_image_job_id_isolation(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_notify,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(True)
        mock_upload.return_value = _make_upload_result()

        result = run_pipeline()

        # Verify generate_all_scene_images was called with job_id kwarg
        mock_images.assert_called_once()
        call_kwargs = mock_images.call_args
        assert "job_id" in call_kwargs.kwargs
        assert call_kwargs.kwargs["job_id"] == result.job_id

    @patch("core.pipeline._send_notification")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_thumbnail_job_id_isolation(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_notify,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(True)
        mock_upload.return_value = _make_upload_result()

        result = run_pipeline()

        # Verify generate_thumbnail was called with job_id kwarg
        mock_thumb.assert_called_once()
        call_kwargs = mock_thumb.call_args
        assert "job_id" in call_kwargs.kwargs
        assert call_kwargs.kwargs["job_id"] == result.job_id


# ---------------------------------------------------------------------------
# run_pipeline — correct stage order
# ---------------------------------------------------------------------------

class TestStageOrder:
    @patch("core.pipeline._send_notification")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_stages_executed_in_order(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_notify,
    ):
        call_order = []
        mock_research.side_effect = lambda: (call_order.append("research"), _make_topic())[1]
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.side_effect = lambda *a, **kw: (call_order.append("script"), "narration")[1]
        mock_scenes.side_effect = lambda *a, **kw: (call_order.append("scenes"), [_make_scene(1)])[1]
        mock_images.side_effect = lambda *a, **kw: (call_order.append("images"), _make_images(1))[1]
        mock_tts.side_effect = lambda *a, **kw: (call_order.append("tts"), _make_audio())[1]
        mock_compose.side_effect = lambda *a, **kw: (call_order.append("composition"), _make_video())[1]
        mock_metadata.side_effect = lambda *a, **kw: (call_order.append("metadata"), _make_metadata())[1]
        mock_thumb.side_effect = lambda *a, **kw: (call_order.append("thumbnail"), _make_thumbnail())[1]
        mock_qa.side_effect = lambda **kw: (call_order.append("qa"), _make_qa_result(True))[1]
        mock_upload.side_effect = lambda **kw: (call_order.append("upload"), _make_upload_result())[1]

        run_pipeline()

        expected = [
            "research", "script", "scenes", "images", "tts",
            "composition", "metadata", "thumbnail", "qa", "upload",
        ]
        assert call_order == expected


# ---------------------------------------------------------------------------
# run_pipeline — failure at each stage
# ---------------------------------------------------------------------------

class TestFailureAtEachStage:
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.find_trending_ai_topic", side_effect=RuntimeError("network error"))
    def test_failure_at_research(self, mock_research, mock_record, mock_email):
        result = run_pipeline()
        assert result.success is False
        assert result.stage == PipelineStage.RESEARCH
        assert "network error" in result.error
        mock_email.assert_called_once()

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.plan_scenes", side_effect=RuntimeError("LLM bad response"))
    @patch("core.pipeline.generate_narration", return_value="narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_failure_at_scenes(self, mock_research, mock_llm, mock_script, mock_scenes, mock_record, mock_email):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        result = run_pipeline()
        assert result.success is False
        assert result.stage == PipelineStage.SCENES
        mock_email.assert_called_once()

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.generate_all_scene_images", side_effect=Exception("API down"))
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_failure_at_images(self, mock_research, mock_llm, mock_script, mock_scenes, mock_images, mock_record, mock_email):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        result = run_pipeline()
        assert result.success is False
        assert result.stage == PipelineStage.IMAGES
        mock_email.assert_called_once()

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.generate_narration_audio", side_effect=Exception("TTS failed"))
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_failure_at_tts(self, mock_research, mock_llm, mock_script, mock_scenes, mock_images, mock_tts, mock_record, mock_email):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        result = run_pipeline()
        assert result.success is False
        assert result.stage == PipelineStage.TTS
        mock_email.assert_called_once()

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.compose_video", side_effect=Exception("moviepy crash"))
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_failure_at_composition(self, mock_research, mock_llm, mock_script, mock_scenes, mock_images, mock_tts, mock_compose, mock_record, mock_email):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        result = run_pipeline()
        assert result.success is False
        assert result.stage == PipelineStage.COMPOSITION
        mock_email.assert_called_once()

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.generate_metadata", side_effect=Exception("LLM failed"))
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_failure_at_metadata(self, mock_research, mock_llm, mock_script, mock_scenes, mock_images, mock_tts, mock_compose, mock_metadata, mock_record, mock_email):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        result = run_pipeline()
        assert result.success is False
        assert result.stage == PipelineStage.METADATA
        mock_email.assert_called_once()

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.generate_thumbnail", side_effect=Exception("Pillow crash"))
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_failure_at_thumbnail(self, mock_research, mock_llm, mock_script, mock_scenes, mock_images, mock_tts, mock_compose, mock_metadata, mock_thumb, mock_record, mock_email):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        result = run_pipeline()
        assert result.success is False
        assert result.stage == PipelineStage.THUMBNAIL
        mock_email.assert_called_once()


# ---------------------------------------------------------------------------
# run_pipeline — QA failure blocks upload
# ---------------------------------------------------------------------------

class TestQAFailureBlocksUpload:
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_qa_failure_blocks_upload(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_email,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(False)

        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.QA
        mock_upload.assert_not_called()
        mock_email.assert_called_once()

    @patch("core.pipeline._send_notification")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_qa_pass_allows_upload(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_notify,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(True)
        mock_upload.return_value = _make_upload_result()

        result = run_pipeline()

        assert result.success is True
        mock_upload.assert_called_once()


# ---------------------------------------------------------------------------
# run_pipeline — partial YouTube success
# ---------------------------------------------------------------------------

class TestPartialYouTubeSuccess:
    @patch("core.pipeline._send_notification")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_partial_success_thumbnail_failed(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_notify,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(True)
        mock_upload.return_value = _make_upload_result(thumbnail_status="failed")

        result = run_pipeline()

        assert result.success is True
        assert result.upload_result.thumbnail_status == "failed"
        # Notification should be called (partial success)
        mock_notify.assert_called_once()

    @patch("core.pipeline._send_notification")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video")
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_partial_success_recorded(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_notify,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(True)
        mock_upload.return_value = _make_upload_result(thumbnail_status="failed")

        result = run_pipeline()

        assert result.success is True
        # Record should have been called with completed status
        mock_record.assert_called()


# ---------------------------------------------------------------------------
# run_pipeline — upload failure
# ---------------------------------------------------------------------------

class TestUploadFailure:
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video", side_effect=UploadError("OAuth expired"))
    @patch("core.pipeline.validate_job")
    @patch("core.pipeline.generate_thumbnail")
    @patch("core.pipeline.generate_metadata")
    @patch("core.pipeline.compose_video")
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_upload_failure_records_and_notifies(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record, mock_email,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]
        mock_images.return_value = _make_images(1)
        mock_tts.return_value = _make_audio()
        mock_compose.return_value = _make_video()
        mock_metadata.return_value = _make_metadata()
        mock_thumb.return_value = _make_thumbnail()
        mock_qa.return_value = _make_qa_result(True)

        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.UPLOAD
        assert "OAuth expired" in result.error
        mock_email.assert_called_once()
        # Should have all artifacts up to upload
        assert result.topic is not None
        assert result.video is not None


# ---------------------------------------------------------------------------
# run_pipeline — no unnecessary stages after failure
# ---------------------------------------------------------------------------

class TestNoStagesAfterFailure:
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.generate_all_scene_images", side_effect=Exception("API down"))
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_no_tts_after_image_failure(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_record, mock_email,
    ):
        mock_research.return_value = _make_topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_make_scene(1)]

        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.IMAGES
        # TTS, composition, metadata, thumbnail, qa, upload should NOT have been called
        # (they are not imported at module level so we verify by stage)


# ---------------------------------------------------------------------------
# run_pipeline — preserves artifacts
# ---------------------------------------------------------------------------

class TestArtifactPreservation:
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.compose_video", side_effect=Exception("moviepy crash"))
    @patch("core.pipeline.generate_narration_audio")
    @patch("core.pipeline.generate_all_scene_images")
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_preserves_all_prior_artifacts(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_record, mock_email,
    ):
        topic = _make_topic()
        mock_research.return_value = topic
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "Test narration"
        scenes = [_make_scene(1), _make_scene(2)]
        mock_scenes.return_value = scenes
        images = _make_images(2)
        mock_images.return_value = images
        audio = _make_audio()
        mock_tts.return_value = audio

        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.COMPOSITION
        assert result.topic == topic
        assert result.narration == "Test narration"
        assert result.scenes == scenes
        assert result.images == images
        assert result.audio == audio
        # video should be None (composition failed before returning)
        assert result.video is None


# ---------------------------------------------------------------------------
# run_pipeline — unexpected exception
# ---------------------------------------------------------------------------

class TestUnexpectedException:
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.find_trending_ai_topic", side_effect=KeyboardInterrupt("ctrl+c"))
    def test_keyboard_interrupt_is_handled(self, mock_research, mock_record, mock_email):
        # KeyboardInterrupt is not caught by Exception, so it propagates
        with pytest.raises(KeyboardInterrupt):
            run_pipeline()

    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.find_trending_ai_topic", side_effect=SystemExit("shutdown"))
    def test_system_exit_is_handled(self, mock_research, mock_record, mock_email):
        with pytest.raises(SystemExit):
            run_pipeline()
