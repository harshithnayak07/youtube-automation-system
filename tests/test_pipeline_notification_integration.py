"""Integration tests — pipeline notification paths trigger correct email functions.

Mocks every external stage (LLM, Pexels, TTS, video, YouTube, DB) but lets
the real _send_notification / _handle_failure / send_job_* code paths execute.
The email send functions themselves are mocked at the core.pipeline level so
no SMTP connection is attempted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from content.metadata import VideoMetadata
from content.scenes import Scene
from core.pipeline import PipelineStage, JobResult, run_pipeline
from core.qa import QAResult
from llm.client import LLMClient
from media.image import GeneratedImage
from media.tts import GeneratedAudio
from media.thumbnail import GeneratedThumbnail
from research.trending import Topic
from video.compositor import ComposedVideo
from youtube.uploader import UploadResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _topic(**kw) -> Topic:
    d = dict(title="AI News 2026", url="https://x.com/ai", summary="AI summary", source="Google")
    d.update(kw)
    return Topic(**d)


def _scene(n: int = 1) -> Scene:
    return Scene(scene=n, text=f"Scene {n} narration.", visual_description=f"Visual {n}")


def _images(n: int = 3) -> list[GeneratedImage]:
    return [GeneratedImage(scene=i + 1, path=Path(f"/tmp/img_{i}.png")) for i in range(n)]


def _audio() -> GeneratedAudio:
    return GeneratedAudio(path=Path("/tmp/audio.mp3"), duration_seconds=45.0)


def _video() -> ComposedVideo:
    return ComposedVideo(path=Path("/tmp/video.mp4"), duration_seconds=45.0, scene_count=3)


def _metadata() -> VideoMetadata:
    return VideoMetadata(title="AI News 2026", description="AI desc", tags=["ai"])


def _thumbnail() -> GeneratedThumbnail:
    return GeneratedThumbnail(path=Path("/tmp/thumb.jpg"), width=1280, height=720)


def _qa(passed: bool = True) -> QAResult:
    if passed:
        return QAResult(passed=True, errors=[], warnings=["minor"], details={})
    return QAResult(passed=False, errors=["too short"], warnings=[], details={})


def _upload(**kw) -> UploadResult:
    d = dict(
        video_id="vid_xyz",
        video_url="https://youtube.com/shorts/vid_xyz",
        video_status="uploaded",
        thumbnail_status="uploaded",
        errors=[],
        details={},
    )
    d.update(kw)
    return UploadResult(**d)


def _success_stubs(m: dict[str, MagicMock]) -> None:
    """Configure mocks for a full-success run."""
    m["find_trending_ai_topic"].return_value = _topic()
    m["_build_llm_client"].return_value = MagicMock(spec=LLMClient)
    m["generate_narration"].return_value = "Test narration."
    m["plan_scenes"].return_value = [_scene(1), _scene(2), _scene(3)]
    m["generate_all_scene_images"].return_value = _images(3)
    m["generate_narration_audio"].return_value = _audio()
    m["compose_video"].return_value = _video()
    m["generate_metadata"].return_value = _metadata()
    m["generate_thumbnail"].return_value = _thumbnail()
    m["validate_job"].return_value = _qa(True)
    m["upload_video"].return_value = _upload()
    m["send_job_success"].return_value = MagicMock(success=True, subject="[YT] OK")
    m["send_job_failure"].return_value = MagicMock(success=True, subject="[YT] FAIL")
    m["send_partial_success"].return_value = MagicMock(success=True, subject="[YT] PARTIAL")


# ---------------------------------------------------------------------------
# Success notification path
# ---------------------------------------------------------------------------

class TestPipelineSuccessNotification:
    """Pipeline completes all stages -> send_job_success() is called."""

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
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
    def test_success_calls_send_job_success(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        _success_stubs({
            "find_trending_ai_topic": mock_research, "_build_llm_client": mock_llm,
            "generate_narration": mock_script, "plan_scenes": mock_scenes,
            "generate_all_scene_images": mock_images, "generate_narration_audio": mock_tts,
            "compose_video": mock_compose, "generate_metadata": mock_metadata,
            "generate_thumbnail": mock_thumb, "validate_job": mock_qa,
            "upload_video": mock_upload, "send_job_success": mock_email_success,
            "send_job_failure": mock_email_failure, "send_partial_success": mock_email_partial,
        })

        result = run_pipeline()

        assert result.success is True
        assert result.stage == PipelineStage.COMPLETE
        mock_email_success.assert_called_once()
        mock_email_failure.assert_not_called()
        mock_email_partial.assert_not_called()

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
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
    def test_success_passes_correct_job_id(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        _success_stubs({
            "find_trending_ai_topic": mock_research, "_build_llm_client": mock_llm,
            "generate_narration": mock_script, "plan_scenes": mock_scenes,
            "generate_all_scene_images": mock_images, "generate_narration_audio": mock_tts,
            "compose_video": mock_compose, "generate_metadata": mock_metadata,
            "generate_thumbnail": mock_thumb, "validate_job": mock_qa,
            "upload_video": mock_upload, "send_job_success": mock_email_success,
            "send_job_failure": mock_email_failure, "send_partial_success": mock_email_partial,
        })

        result = run_pipeline()

        kwargs = mock_email_success.call_args[1]
        assert kwargs["job_id"] == result.job_id

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
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
    def test_success_passes_topic_and_video_details(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        _success_stubs({
            "find_trending_ai_topic": mock_research, "_build_llm_client": mock_llm,
            "generate_narration": mock_script, "plan_scenes": mock_scenes,
            "generate_all_scene_images": mock_images, "generate_narration_audio": mock_tts,
            "compose_video": mock_compose, "generate_metadata": mock_metadata,
            "generate_thumbnail": mock_thumb, "validate_job": mock_qa,
            "upload_video": mock_upload, "send_job_success": mock_email_success,
            "send_job_failure": mock_email_failure, "send_partial_success": mock_email_partial,
        })

        run_pipeline()

        kwargs = mock_email_success.call_args[1]
        assert kwargs["topic"] == "AI News 2026"
        assert kwargs["video_id"] == "vid_xyz"
        assert kwargs["video_url"] == "https://youtube.com/shorts/vid_xyz"
        assert kwargs["upload_status"] == "uploaded"
        assert kwargs["thumbnail_status"] == "uploaded"


# ---------------------------------------------------------------------------
# Partial success notification path
# ---------------------------------------------------------------------------

class TestPipelinePartialNotification:
    """Video uploaded but thumbnail failed -> send_partial_success() is called."""

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
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
    def test_partial_calls_send_partial_success(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        _success_stubs({
            "find_trending_ai_topic": mock_research, "_build_llm_client": mock_llm,
            "generate_narration": mock_script, "plan_scenes": mock_scenes,
            "generate_all_scene_images": mock_images, "generate_narration_audio": mock_tts,
            "compose_video": mock_compose, "generate_metadata": mock_metadata,
            "generate_thumbnail": mock_thumb, "validate_job": mock_qa,
            "upload_video": mock_upload, "send_job_success": mock_email_success,
            "send_job_failure": mock_email_failure, "send_partial_success": mock_email_partial,
        })
        mock_upload.return_value = _upload(thumbnail_status="failed")

        result = run_pipeline()

        assert result.success is True
        mock_email_partial.assert_called_once()
        mock_email_success.assert_not_called()
        mock_email_failure.assert_not_called()

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
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
    def test_partial_passes_warnings_and_video_details(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        _success_stubs({
            "find_trending_ai_topic": mock_research, "_build_llm_client": mock_llm,
            "generate_narration": mock_script, "plan_scenes": mock_scenes,
            "generate_all_scene_images": mock_images, "generate_narration_audio": mock_tts,
            "compose_video": mock_compose, "generate_metadata": mock_metadata,
            "generate_thumbnail": mock_thumb, "validate_job": mock_qa,
            "upload_video": mock_upload, "send_job_success": mock_email_success,
            "send_job_failure": mock_email_failure, "send_partial_success": mock_email_partial,
        })
        mock_upload.return_value = _upload(thumbnail_status="failed")

        run_pipeline()

        kwargs = mock_email_partial.call_args[1]
        assert kwargs["video_id"] == "vid_xyz"
        assert kwargs["upload_status"] == "uploaded"
        assert kwargs["thumbnail_status"] == "failed"
        assert "minor" in kwargs["warnings"]


# ---------------------------------------------------------------------------
# Failure notification path — each stage
# ---------------------------------------------------------------------------

class TestPipelineFailureNotification:
    """Pipeline fails at a stage -> send_job_failure() is called."""

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.find_trending_ai_topic", side_effect=RuntimeError("no internet"))
    def test_research_failure(
        self, mock_research, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.RESEARCH
        mock_email_failure.assert_called_once()
        mock_email_success.assert_not_called()
        mock_email_partial.assert_not_called()

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.find_trending_ai_topic", side_effect=RuntimeError("no internet"))
    def test_research_failure_passes_error_details(
        self, mock_research, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        result = run_pipeline()

        kwargs = mock_email_failure.call_args[1]
        assert kwargs["failed_stage"] == "research"
        assert kwargs["error_type"] == "RuntimeError"
        assert kwargs["error_message"] == "no internet"
        assert kwargs["job_id"] == result.job_id

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.plan_scenes", side_effect=RuntimeError("LLM parse error"))
    @patch("core.pipeline.generate_narration", return_value="narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_scenes_failure(
        self, mock_research, mock_llm, mock_script, mock_scenes, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        mock_research.return_value = _topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)

        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.SCENES
        kwargs = mock_email_failure.call_args[1]
        assert kwargs["failed_stage"] == "scenes"

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.generate_all_scene_images", side_effect=Exception("Pexels 429"))
    @patch("core.pipeline.plan_scenes")
    @patch("core.pipeline.generate_narration")
    @patch("core.pipeline._build_llm_client")
    @patch("core.pipeline.find_trending_ai_topic")
    def test_images_failure(
        self, mock_research, mock_llm, mock_script, mock_scenes, mock_images, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        mock_research.return_value = _topic()
        mock_llm.return_value = MagicMock(spec=LLMClient)
        mock_script.return_value = "narration"
        mock_scenes.return_value = [_scene(1)]

        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.IMAGES
        kwargs = mock_email_failure.call_args[1]
        assert kwargs["failed_stage"] == "images"

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
    @patch("core.pipeline._update_job_record")
    @patch("core.pipeline.upload_video", side_effect=Exception("quota exceeded"))
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
    def test_upload_failure(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        _success_stubs({
            "find_trending_ai_topic": mock_research, "_build_llm_client": mock_llm,
            "generate_narration": mock_script, "plan_scenes": mock_scenes,
            "generate_all_scene_images": mock_images, "generate_narration_audio": mock_tts,
            "compose_video": mock_compose, "generate_metadata": mock_metadata,
            "generate_thumbnail": mock_thumb, "validate_job": mock_qa,
            "upload_video": mock_upload, "send_job_success": mock_email_success,
            "send_job_failure": mock_email_failure, "send_partial_success": mock_email_partial,
        })
        mock_upload.side_effect = Exception("quota exceeded")

        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.UPLOAD
        kwargs = mock_email_failure.call_args[1]
        assert kwargs["failed_stage"] == "upload"

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
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
        mock_thumb, mock_qa, mock_upload, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        _success_stubs({
            "find_trending_ai_topic": mock_research, "_build_llm_client": mock_llm,
            "generate_narration": mock_script, "plan_scenes": mock_scenes,
            "generate_all_scene_images": mock_images, "generate_narration_audio": mock_tts,
            "compose_video": mock_compose, "generate_metadata": mock_metadata,
            "generate_thumbnail": mock_thumb, "validate_job": mock_qa,
            "upload_video": mock_upload, "send_job_success": mock_email_success,
            "send_job_failure": mock_email_failure, "send_partial_success": mock_email_partial,
        })
        mock_qa.return_value = _qa(False)

        result = run_pipeline()

        assert result.success is False
        assert result.stage == PipelineStage.QA
        mock_upload.assert_not_called()
        kwargs = mock_email_failure.call_args[1]
        assert kwargs["failed_stage"] == "qa"


# ---------------------------------------------------------------------------
# No YouTube / LLM / Pexels calls leak through
# ---------------------------------------------------------------------------

class TestNoExternalCalls:
    """Verify that mocked stages are never actually invoked."""

    @patch("core.pipeline.send_partial_success")
    @patch("core.pipeline.send_job_failure")
    @patch("core.pipeline.send_job_success")
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
    @patch("core.pipeline.find_trending_ai_topic", side_effect=RuntimeError("fail"))
    def test_no_upload_or_compose_on_research_failure(
        self, mock_research, mock_llm, mock_script, mock_scenes,
        mock_images, mock_tts, mock_compose, mock_metadata,
        mock_thumb, mock_qa, mock_upload, mock_record,
        mock_email_success, mock_email_failure, mock_email_partial,
    ):
        run_pipeline()

        mock_upload.assert_not_called()
        mock_compose.assert_not_called()
        mock_tts.assert_not_called()
        mock_images.assert_not_called()
