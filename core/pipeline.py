"""Core pipeline — orchestrates a complete daily YouTube Short job."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from config import settings
from content.metadata import VideoMetadata, generate_metadata
from content.script import generate_narration
from content.scenes import Scene, plan_scenes
from core.qa import QAResult, validate_job
from llm.client import GroqProvider, LLMClient, LLMConfigError, OpenRouterProvider
from media.image import GeneratedImage, generate_all_scene_images
from media.tts import GeneratedAudio, generate_narration_audio
from media.thumbnail import GeneratedThumbnail, generate_thumbnail
from notifications.email_sender import (
    send_job_failure,
    send_job_success,
    send_partial_success,
)
from research.trending import Topic, find_trending_ai_topic
from storage.database import get_connection, init_db
from video.compositor import ComposedVideo, compose_video
from youtube.uploader import UploadError, upload_video

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage tracking
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    """Explicit pipeline stages in execution order."""
    RESEARCH = "research"
    SCRIPT = "script"
    SCENES = "scenes"
    IMAGES = "images"
    TTS = "tts"
    COMPOSITION = "composition"
    METADATA = "metadata"
    THUMBNAIL = "thumbnail"
    QA = "qa"
    UPLOAD = "upload"
    RECORD = "record"
    NOTIFY = "notify"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Job result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobResult:
    """Structured outcome of a pipeline run."""
    job_id: str
    success: bool
    stage: PipelineStage
    topic: Topic | None = None
    narration: str | None = None
    scenes: list[Scene] | None = None
    images: list[GeneratedImage] | None = None
    audio: GeneratedAudio | None = None
    video: ComposedVideo | None = None
    metadata: VideoMetadata | None = None
    thumbnail: GeneratedThumbnail | None = None
    qa_result: QAResult | None = None
    upload_result: Any | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _create_job_record(topic: str) -> int:
    """Insert a new job record and return its ID."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO jobs (topic, status) VALUES (?, ?)",
            (topic, "running"),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def _update_job_record(
    job_db_id: int,
    *,
    status: str | None = None,
    video_id: str | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Update a job record.  Only non-None fields are written."""
    updates: list[str] = []
    values: list[Any] = []

    if status is not None:
        updates.append("status = ?")
        values.append(status)
    if video_id is not None:
        updates.append("video_id = ?")
        values.append(video_id)
    if error_message is not None:
        updates.append("error_message = ?")
        values.append(error_message)
    if metadata is not None:
        updates.append("metadata = ?")
        values.append(json.dumps(metadata, default=str))

    updates.append("updated_at = ?")
    values.append(datetime.now(timezone.utc).isoformat())

    values.append(job_db_id)

    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def _get_job_record(job_db_id: int) -> dict[str, Any] | None:
    """Fetch a job record by ID."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_db_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def _build_llm_client() -> LLMClient:
    """Build an LLMClient from environment configuration.

    Returns None if no API keys are available (degraded mode — LLM
    stages will raise, which the pipeline handles as failures).
    """
    primary = None
    fallback = None

    if settings.GROQ_API_KEY:
        try:
            primary = GroqProvider(settings.GROQ_API_KEY, default_model=settings.GROQ_MODEL)
        except LLMConfigError:
            pass

    if settings.OPENROUTER_API_KEY:
        try:
            fallback = OpenRouterProvider(settings.OPENROUTER_API_KEY, default_model=settings.OPENROUTER_MODEL)
        except LLMConfigError:
            pass

    if primary is None and fallback is None:
        raise LLMConfigError(
            "No LLM API keys configured. Set GROQ_API_KEY or OPENROUTER_API_KEY."
        )

    return LLMClient(
        primary=primary or fallback,  # type: ignore[arg-type]
        fallback=fallback if primary else None,
    )


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Terminal pipeline failure with stage tracking."""


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline() -> JobResult:
    """Execute a complete daily YouTube Short pipeline.

    Stages:
        1. Research trending AI topic
        2. Generate narration
        3. Plan scenes
        4. Generate scene images
        5. Generate TTS audio
        6. Compose video
        7. Generate metadata
        8. Generate thumbnail
        9. Run QA gate
        10. Upload to YouTube (only if QA passes)
        11. Record final job result
        12. Send email notification

    Returns
    -------
    JobResult
        Structured outcome with all artifacts and stage information.
    """
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    topic: Topic | None = None
    narration: str | None = None
    scenes: list[Scene] | None = None
    images: list[GeneratedImage] | None = None
    audio: GeneratedAudio | None = None
    video: ComposedVideo | None = None
    metadata: VideoMetadata | None = None
    thumbnail: GeneratedThumbnail | None = None
    qa_result: QAResult | None = None
    upload_result: Any | None = None
    failed_stage: PipelineStage | None = None
    error_message: str = ""
    error_type: str = ""
    stage_details: dict[str, Any] = {}

    logger.info("Pipeline started: %s", job_id)

    # ------------------------------------------------------------------
    # 1. Research
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: research")
        topic = find_trending_ai_topic()
        stage_details["research"] = {"topic": topic.title, "source": topic.source}
    except Exception as exc:
        failed_stage = PipelineStage.RESEARCH
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at research: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
        )

    # ------------------------------------------------------------------
    # 2. Script / narration
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: script")
        llm_client = _build_llm_client()
        narration = generate_narration(llm_client, topic)
        stage_details["script"] = {"length": len(narration)}
    except Exception as exc:
        failed_stage = PipelineStage.SCRIPT
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at script: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
        )

    # ------------------------------------------------------------------
    # 3. Scenes
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: scenes")
        scenes = plan_scenes(llm_client, narration)
        stage_details["scenes"] = {"count": len(scenes)}
    except Exception as exc:
        failed_stage = PipelineStage.SCENES
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at scenes: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
            scenes=scenes,
        )

    # ------------------------------------------------------------------
    # 4. Images
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: images")
        images = generate_all_scene_images(scenes, job_id=job_id)
        stage_details["images"] = {"count": len(images)}
    except Exception as exc:
        failed_stage = PipelineStage.IMAGES
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at images: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
        )

    # ------------------------------------------------------------------
    # 5. TTS
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: tts")
        audio = generate_narration_audio(narration)
        stage_details["tts"] = {"duration": audio.duration_seconds}
    except Exception as exc:
        failed_stage = PipelineStage.TTS
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at tts: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
            audio=audio,
        )

    # ------------------------------------------------------------------
    # 6. Composition
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: composition")
        video = compose_video(scenes, images, audio)
        stage_details["composition"] = {
            "duration": video.duration_seconds,
            "scene_count": video.scene_count,
        }
    except Exception as exc:
        failed_stage = PipelineStage.COMPOSITION
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at composition: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
            audio=audio,
            video=video,
        )

    # ------------------------------------------------------------------
    # 7. Metadata
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: metadata")
        metadata = generate_metadata(llm_client, topic, narration)
        stage_details["metadata"] = {"title": metadata.title}
    except Exception as exc:
        failed_stage = PipelineStage.METADATA
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at metadata: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
            audio=audio,
            video=video,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # 8. Thumbnail
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: thumbnail")
        thumbnail = generate_thumbnail(topic.title, images, job_id=job_id)
        stage_details["thumbnail"] = {
            "width": thumbnail.width,
            "height": thumbnail.height,
        }
    except Exception as exc:
        failed_stage = PipelineStage.THUMBNAIL
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at thumbnail: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
            audio=audio,
            video=video,
            metadata=metadata,
            thumbnail=thumbnail,
        )

    # ------------------------------------------------------------------
    # 9. QA gate
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: qa")
        qa_result = validate_job(
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
            audio=audio,
            video=video,
            metadata=metadata,
            thumbnail=thumbnail,
        )
        stage_details["qa"] = {
            "passed": qa_result.passed,
            "errors": qa_result.errors,
            "warnings": qa_result.warnings,
        }

        if not qa_result.passed:
            failed_stage = PipelineStage.QA
            error_type = "QAFailure"
            error_message = f"QA gate failed with {len(qa_result.errors)} error(s): {'; '.join(qa_result.errors[:3])}"
            logger.error("Pipeline failed at QA: %s", error_message)
            return _handle_failure(
                job_id=job_id,
                failed_stage=failed_stage,
                error_type=error_type,
                error_message=error_message,
                topic=topic,
                narration=narration,
                scenes=scenes,
                images=images,
                audio=audio,
                video=video,
                metadata=metadata,
                thumbnail=thumbnail,
                qa_result=qa_result,
            )
    except Exception as exc:
        failed_stage = PipelineStage.QA
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at QA: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
            audio=audio,
            video=video,
            metadata=metadata,
            thumbnail=thumbnail,
            qa_result=qa_result,
        )

    # ------------------------------------------------------------------
    # 10. Upload to YouTube
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: upload")
        upload_result = upload_video(
            video_path=video.path,
            thumbnail_path=thumbnail.path,
            video_metadata=metadata,
            qa_result=qa_result,
        )
        stage_details["upload"] = {
            "video_id": upload_result.video_id,
            "video_status": upload_result.video_status,
            "thumbnail_status": upload_result.thumbnail_status,
        }
    except Exception as exc:
        failed_stage = PipelineStage.UPLOAD
        error_type = type(exc).__name__
        error_message = str(exc)
        logger.error("Pipeline failed at upload: %s", exc)
        return _handle_failure(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
            audio=audio,
            video=video,
            metadata=metadata,
            thumbnail=thumbnail,
            qa_result=qa_result,
        )

    # ------------------------------------------------------------------
    # 11. Record result
    # ------------------------------------------------------------------
    try:
        logger.info("Stage: record")
        _update_job_record(
            1,  # placeholder — real DB ID wired in next phase
            status="completed",
            video_id=upload_result.video_id,
            metadata=stage_details,
        )
    except Exception as exc:
        logger.warning("Failed to record job result: %s", exc)

    # ------------------------------------------------------------------
    # 12. Notify
    # ------------------------------------------------------------------
    _send_notification(
        job_id=job_id,
        topic=topic,
        upload_result=upload_result,
        qa_result=qa_result,
        stage_details=stage_details,
    )

    logger.info("Pipeline completed: %s", job_id)
    return JobResult(
        job_id=job_id,
        success=True,
        stage=PipelineStage.COMPLETE,
        topic=topic,
        narration=narration,
        scenes=scenes,
        images=images,
        audio=audio,
        video=video,
        metadata=metadata,
        thumbnail=thumbnail,
        qa_result=qa_result,
        upload_result=upload_result,
    )


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def _handle_failure(
    *,
    job_id: str,
    failed_stage: PipelineStage,
    error_type: str,
    error_message: str,
    topic: Topic | None = None,
    narration: str | None = None,
    scenes: list[Scene] | None = None,
    images: list[GeneratedImage] | None = None,
    audio: GeneratedAudio | None = None,
    video: ComposedVideo | None = None,
    metadata: VideoMetadata | None = None,
    thumbnail: GeneratedThumbnail | None = None,
    qa_result: QAResult | None = None,
) -> JobResult:
    """Record the failure and send a failure notification."""
    # Record in database
    try:
        _update_job_record(
            1,  # placeholder — real DB ID wired in next phase
            status="failed",
            error_message=f"[{failed_stage.value}] {error_type}: {error_message}",
        )
    except Exception as exc:
        logger.warning("Failed to record job failure: %s", exc)

    # Send failure notification
    try:
        send_job_failure(
            job_id=job_id,
            failed_stage=failed_stage.value,
            error_type=error_type,
            error_message=error_message,
        )
    except Exception as exc:
        logger.warning("Failed to send failure notification: %s", exc)

    return JobResult(
        job_id=job_id,
        success=False,
        stage=failed_stage,
        topic=topic,
        narration=narration,
        scenes=scenes,
        images=images,
        audio=audio,
        video=video,
        metadata=metadata,
        thumbnail=thumbnail,
        qa_result=qa_result,
        error=f"[{failed_stage.value}] {error_type}: {error_message}",
    )


# ---------------------------------------------------------------------------
# Notification dispatch
# ---------------------------------------------------------------------------

def _send_notification(
    *,
    job_id: str,
    topic: Topic,
    upload_result: Any,
    qa_result: QAResult,
    stage_details: dict[str, Any],
) -> None:
    """Send the appropriate email notification based on upload outcome."""
    thumbnail_status = getattr(upload_result, "thumbnail_status", "unknown")

    if thumbnail_status == "failed":
        # Partial success — video uploaded but thumbnail failed
        try:
            send_partial_success(
                job_id=job_id,
                topic=topic.title,
                video_id=upload_result.video_id,
                video_url=upload_result.video_url,
                upload_status=upload_result.video_status,
                thumbnail_status=thumbnail_status,
                warnings=qa_result.warnings,
            )
        except Exception as exc:
            logger.warning("Failed to send partial success notification: %s", exc)
    else:
        # Full success
        try:
            send_job_success(
                job_id=job_id,
                topic=topic.title,
                video_id=upload_result.video_id,
                video_url=upload_result.video_url,
                upload_status=upload_result.video_status,
                thumbnail_status=thumbnail_status,
            )
        except Exception as exc:
            logger.warning("Failed to send success notification: %s", exc)
