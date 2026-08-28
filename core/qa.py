"""Core module — Production QA gate for video jobs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from content.metadata import VideoMetadata
from content.scenes import Scene
from media.image import GeneratedImage
from media.tts import GeneratedAudio
from media.thumbnail import GeneratedThumbnail
from research.trending import Topic
from video.compositor import ComposedVideo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MIN_NARRATION_WORDS = 30
_MAX_NARRATION_WORDS = 600
_MAX_DURATION_DRIFT_RATIO = 0.30   # warn above 30% drift
_SEVERE_DURATION_DRIFT_RATIO = 0.50  # fail above 50% drift
_EXPECTED_THUMBNAIL_WIDTH = 1280
_EXPECTED_THUMBNAIL_HEIGHT = 720

# ---------------------------------------------------------------------------
# Image / audio / video signature helpers (self-contained, no external calls)
# ---------------------------------------------------------------------------

_IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
    b"GIF8": "gif",
    b"RIFF": "webp",
}

_AUDIO_SIGNATURES = {
    b"ID3": "mp3",
    b"RIFF": "wav",
    b"OggS": "ogg",
    b"fLaC": "flac",
}

_VIDEO_SIGNATURES = {
    b"\x1a\x45\xdf\xa3": "webm",
    b"RIFF": "avi",
}


def _has_image_signature(header: bytes) -> bool:
    if len(header) < 4:
        return False
    for sig in _IMAGE_SIGNATURES:
        if header[: len(sig)] == sig:
            return True
    return False


def _has_audio_signature(header: bytes) -> bool:
    if len(header) < 4:
        return False
    # MP3 frame sync: 0xFF followed by 0xE0 mask
    if header[:3] == b"ID3":
        return True
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WAVE":
        return True
    if header[:4] in (b"OggS", b"fLaC"):
        return True
    return False


def _has_video_signature(header: bytes) -> bool:
    if len(header) < 4:
        return False
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return True
    for sig in _VIDEO_SIGNATURES:
        if header[: len(sig)] == sig:
            return True
    if header[0] == 0x47:
        return True
    return False


def _is_image_file_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if size < 100:
            return False
        with open(path, "rb") as f:
            header = f.read(12)
        return _has_image_signature(header)
    except OSError:
        return False


def _is_audio_file_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if size < 100:
            return False
        with open(path, "rb") as f:
            header = f.read(12)
        return _has_audio_signature(header)
    except OSError:
        return False


def _is_video_file_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if size < 1000:
            return False
        with open(path, "rb") as f:
            header = f.read(12)
        return _has_video_signature(header)
    except OSError:
        return False


def _check_mp3_duration(path: Path) -> float:
    """Estimate MP3 duration from Xing header."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 200:
            return 0.0
        import struct
        for offset in range(0, min(len(data), 400)):
            if data[offset: offset + 4] in (b"Xing", b"Info"):
                if offset + 12 <= len(data):
                    total_frames = struct.unpack(">I", data[offset + 8: offset + 12])[0]
                    if total_frames > 0:
                        return round(total_frames * 1152 / 44100, 2)
                break
        return 0.0
    except Exception:
        return 0.0


def _check_video_aspect_ratio(path: Path) -> tuple[int, int] | None:
    """Read MP4 dimensions from moov/trak if possible, else None."""
    try:
        with open(path, "rb") as f:
            data = f.read(64 * 1024)  # Read first 64KB
        # Look for 'tkhd' box which contains width/height
        idx = data.find(b"tkhd")
        if idx == -1:
            return None
        # tkhd version byte, then skip to width/height at fixed offsets
        # Version 0: width at offset +76 from box start, height at +80
        # Version 1: width at offset +88, height at +92
        version = data[idx + 4] if idx + 5 <= len(data) else 0
        if version == 0:
            w_off = idx + 76
            h_off = idx + 80
        else:
            w_off = idx + 88
            h_off = idx + 92
        if h_off + 4 > len(data):
            return None
        import struct
        w = struct.unpack(">I", data[w_off: w_off + 4])[0] >> 16  # Fixed-point 16.16
        h = struct.unpack(">I", data[h_off: h_off + 4])[0] >> 16
        if w > 0 and h > 0:
            return (w, h)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QAResult:
    """Structured result of a QA validation pass."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------

class _QARecorder:
    """Collects errors, warnings, and details during validation."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.details: dict[str, Any] = {}

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def detail(self, key: str, value: Any) -> None:
        self.details[key] = value

    def to_result(self) -> QAResult:
        return QAResult(
            passed=len(self.errors) == 0,
            errors=list(self.errors),
            warnings=list(self.warnings),
            details=dict(self.details),
        )


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def _validate_topic(topic: Topic | None, rec: _QARecorder) -> None:
    """Validate the researched topic."""
    if topic is None:
        rec.error("Topic is missing")
        return

    rec.detail("topic_title", topic.title)

    if not isinstance(topic.title, str) or not topic.title.strip():
        rec.error("Topic title is empty")

    if not isinstance(topic.url, str) or not topic.url.strip():
        rec.error("Topic source URL is missing")

    if not isinstance(topic.summary, str) or not topic.summary.strip():
        rec.warn("Topic summary is empty")

    if not isinstance(topic.source, str) or not topic.source.strip():
        rec.warn("Topic source name is empty")


def _validate_narration(narration: str | None, rec: _QARecorder) -> None:
    """Validate the narration text."""
    if narration is None:
        rec.error("Narration is missing")
        return

    if not isinstance(narration, str):
        rec.error("Narration is not a string")
        return

    stripped = narration.strip()
    if not stripped:
        rec.error("Narration is empty")
        return

    rec.detail("narration_length", len(stripped))

    words = stripped.split()
    word_count = len(words)
    rec.detail("narration_word_count", word_count)

    if word_count < _MIN_NARRATION_WORDS:
        rec.warn(
            f"Narration is short ({word_count} words, minimum recommended: {_MIN_NARRATION_WORDS})"
        )
    elif word_count > _MAX_NARRATION_WORDS:
        rec.warn(
            f"Narration is long ({word_count} words, maximum recommended: {_MAX_NARRATION_WORDS})"
        )


def _validate_scenes(scenes: list[Scene] | None, rec: _QARecorder) -> None:
    """Validate scene list structure and content."""
    if scenes is None:
        rec.error("Scenes are missing")
        return

    if not isinstance(scenes, list):
        rec.error("Scenes is not a list")
        return

    if len(scenes) == 0:
        rec.error("Scenes list is empty")
        return

    rec.detail("scene_count", len(scenes))

    scene_numbers = []
    for i, scene in enumerate(scenes):
        if not isinstance(scene, Scene):
            rec.error(f"Scene at index {i} is not a Scene instance")
            continue

        scene_numbers.append(scene.scene)

        if not isinstance(scene.text, str) or not scene.text.strip():
            rec.error(f"Scene {scene.scene} has empty text")

        if not isinstance(scene.visual_description, str) or not scene.visual_description.strip():
            rec.error(f"Scene {scene.scene} has empty visual_description")

    if scene_numbers:
        if scene_numbers[0] != 1:
            rec.error(f"Scenes do not start at 1 (first scene number: {scene_numbers[0]})")
        for j in range(1, len(scene_numbers)):
            if scene_numbers[j] != scene_numbers[j - 1] + 1:
                rec.error(
                    f"Scene numbering is not sequential: "
                    f"{scene_numbers[j - 1]} -> {scene_numbers[j]}"
                )


def _validate_images(
    scenes: list[Scene] | None,
    images: list[GeneratedImage] | None,
    rec: _QARecorder,
) -> None:
    """Validate image assets for each scene."""
    if images is None:
        rec.error("Images are missing")
        return

    if not isinstance(images, list):
        rec.error("Images is not a list")
        return

    if len(images) == 0:
        rec.error("Images list is empty")
        return

    rec.detail("image_count", len(images))

    if scenes is not None and isinstance(scenes, list) and len(scenes) > 0:
        if len(images) != len(scenes):
            rec.error(
                f"Image count ({len(images)}) does not match scene count ({len(scenes)})"
            )

    for img in images:
        if not isinstance(img, GeneratedImage):
            rec.error(f"Image entry is not a GeneratedImage instance")
            continue

        if not isinstance(img.path, Path):
            rec.error(f"Image for scene {img.scene} has invalid path type")
            continue

        if not img.path.is_file():
            rec.error(f"Image file missing for scene {img.scene}: {img.path}")
            continue

        try:
            size = img.path.stat().st_size
            if size == 0:
                rec.error(f"Image file is empty for scene {img.scene}: {img.path}")
                continue
            rec.detail(f"image_{img.scene}_size", size)
        except OSError as exc:
            rec.error(f"Cannot read image for scene {img.scene}: {exc}")
            continue

        if not _is_image_file_valid(img.path):
            rec.error(f"Image file is corrupt or invalid for scene {img.scene}: {img.path}")
        else:
            rec.detail(f"image_{img.scene}_valid", True)


def _validate_audio(audio: GeneratedAudio | None, rec: _QARecorder) -> None:
    """Validate TTS audio asset."""
    if audio is None:
        rec.error("TTS audio is missing")
        return

    if not isinstance(audio, GeneratedAudio):
        rec.error("Audio is not a GeneratedAudio instance")
        return

    if not isinstance(audio.path, Path):
        rec.error("Audio path is not a Path")
        return

    if not audio.path.is_file():
        rec.error(f"Audio file does not exist: {audio.path}")
        return

    try:
        size = audio.path.stat().st_size
        rec.detail("audio_size", size)
        if size == 0:
            rec.error(f"Audio file is empty: {audio.path}")
            return
    except OSError as exc:
        rec.error(f"Cannot read audio file: {exc}")
        return

    if not _is_audio_file_valid(audio.path):
        rec.error(f"Audio file is corrupt or invalid: {audio.path}")
        return

    duration = audio.duration_seconds
    rec.detail("audio_duration", duration)

    if duration <= 0:
        rec.error(f"Audio duration is not positive: {duration}")


def _validate_video(
    video: ComposedVideo | None,
    audio: GeneratedAudio | None,
    rec: _QARecorder,
) -> None:
    """Validate the composed video asset."""
    if video is None:
        rec.error("Final video is missing")
        return

    if not isinstance(video, ComposedVideo):
        rec.error("Video is not a ComposedVideo instance")
        return

    if not isinstance(video.path, Path):
        rec.error("Video path is not a Path")
        return

    if not video.path.is_file():
        rec.error(f"Video file does not exist: {video.path}")
        return

    try:
        size = video.path.stat().st_size
        rec.detail("video_size", size)
        if size == 0:
            rec.error(f"Video file is empty: {video.path}")
            return
    except OSError as exc:
        rec.error(f"Cannot read video file: {exc}")
        return

    if not _is_video_file_valid(video.path):
        rec.error(f"Video file is corrupt or invalid: {video.path}")
        return

    # Aspect ratio check
    dimensions = _check_video_aspect_ratio(video.path)
    if dimensions is not None:
        w, h = dimensions
        rec.detail("video_width", w)
        rec.detail("video_height", h)
        ratio = w / h if h > 0 else 0
        target_ratio = 1080 / 1920
        if abs(ratio - target_ratio) > 0.05:
            rec.error(
                f"Video aspect ratio is not 9:16 ({w}x{h}, ratio={ratio:.3f})"
            )
    else:
        rec.warn("Could not probe video dimensions for aspect ratio check")

    # Duration alignment with audio
    video_duration = video.duration_seconds
    rec.detail("video_duration", video_duration)

    if video_duration <= 0:
        rec.error(f"Video duration is not positive: {video_duration}")
    elif audio is not None and isinstance(audio, GeneratedAudio) and audio.duration_seconds > 0:
        audio_duration = audio.duration_seconds
        drift = abs(video_duration - audio_duration)
        drift_ratio = drift / audio_duration if audio_duration > 0 else 0
        rec.detail("duration_drift_ratio", round(drift_ratio, 4))

        if drift_ratio > _SEVERE_DURATION_DRIFT_RATIO:
            rec.error(
                f"Severe video/audio duration drift: "
                f"video={video_duration:.1f}s, audio={audio_duration:.1f}s "
                f"(drift={drift:.1f}s, {drift_ratio:.0%})"
            )
        elif drift_ratio > _MAX_DURATION_DRIFT_RATIO:
            rec.warn(
                f"Video/audio duration drift is large: "
                f"video={video_duration:.1f}s, audio={audio_duration:.1f}s "
                f"(drift={drift:.1f}s, {drift_ratio:.0%})"
            )


def _validate_metadata(metadata: VideoMetadata | None, rec: _QARecorder) -> None:
    """Validate YouTube metadata."""
    if metadata is None:
        rec.error("Metadata is missing")
        return

    if not isinstance(metadata, VideoMetadata):
        rec.error("Metadata is not a VideoMetadata instance")
        return

    if not isinstance(metadata.title, str) or not metadata.title.strip():
        rec.error("Metadata title is empty")
    else:
        rec.detail("metadata_title_length", len(metadata.title.strip()))

    if not isinstance(metadata.description, str) or not metadata.description.strip():
        rec.error("Metadata description is empty")
    else:
        rec.detail("metadata_description_length", len(metadata.description.strip()))

    if not isinstance(metadata.tags, list) or len(metadata.tags) == 0:
        rec.error("Metadata has no tags")
    else:
        clean_tags = [t for t in metadata.tags if isinstance(t, str) and t.strip()]
        if len(clean_tags) == 0:
            rec.error("Metadata tags list contains no valid tags")
        else:
            rec.detail("metadata_tag_count", len(clean_tags))
            if len(clean_tags) < 3:
                rec.warn(f"Metadata has very few tags ({len(clean_tags)})")


def _validate_thumbnail(thumbnail: GeneratedThumbnail | None, rec: _QARecorder) -> None:
    """Validate the thumbnail asset."""
    if thumbnail is None:
        rec.error("Thumbnail is missing")
        return

    if not isinstance(thumbnail, GeneratedThumbnail):
        rec.error("Thumbnail is not a GeneratedThumbnail instance")
        return

    if not isinstance(thumbnail.path, Path):
        rec.error("Thumbnail path is not a Path")
        return

    if not thumbnail.path.is_file():
        rec.error(f"Thumbnail file does not exist: {thumbnail.path}")
        return

    try:
        size = thumbnail.path.stat().st_size
        rec.detail("thumbnail_size", size)
        if size == 0:
            rec.error(f"Thumbnail file is empty: {thumbnail.path}")
            return
    except OSError as exc:
        rec.error(f"Cannot read thumbnail file: {exc}")
        return

    if not _is_image_file_valid(thumbnail.path):
        rec.error(f"Thumbnail file is corrupt or invalid: {thumbnail.path}")
        return

    if thumbnail.width != _EXPECTED_THUMBNAIL_WIDTH or thumbnail.height != _EXPECTED_THUMBNAIL_HEIGHT:
        rec.error(
            f"Thumbnail dimensions are not {_EXPECTED_THUMBNAIL_WIDTH}x{_EXPECTED_THUMBNAIL_HEIGHT}: "
            f"{thumbnail.width}x{thumbnail.height}"
        )
    else:
        rec.detail("thumbnail_dimensions", f"{thumbnail.width}x{thumbnail.height}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_job(
    *,
    topic: Topic | None = None,
    narration: str | None = None,
    scenes: list[Scene] | None = None,
    images: list[GeneratedImage] | None = None,
    audio: GeneratedAudio | None = None,
    video: ComposedVideo | None = None,
    metadata: VideoMetadata | None = None,
    thumbnail: GeneratedThumbnail | None = None,
) -> QAResult:
    """Validate a complete video job before upload.

    Parameters are all optional — any missing asset is reported as an error.
    The gate only inspects and reports; it never generates or repairs assets.

    Returns
    -------
    QAResult
        ``passed=True`` only if there are zero errors.
        ``errors`` are hard failures; ``warnings`` are advisory.
    """
    rec = _QARecorder()

    _validate_topic(topic, rec)
    _validate_narration(narration, rec)
    _validate_scenes(scenes, rec)
    _validate_images(scenes, images, rec)
    _validate_audio(audio, rec)
    _validate_video(video, audio, rec)
    _validate_metadata(metadata, rec)
    _validate_thumbnail(thumbnail, rec)

    result = rec.to_result()

    if result.passed:
        logger.info("QA gate PASSED (%d warnings)", len(result.warnings))
    else:
        logger.warning(
            "QA gate FAILED (%d errors, %d warnings)",
            len(result.errors),
            len(result.warnings),
        )

    return result
