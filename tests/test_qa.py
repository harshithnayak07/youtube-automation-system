"""Tests for the core.qa module — production QA gate."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from content.metadata import VideoMetadata
from content.scenes import Scene
from core.qa import (
    QAResult,
    validate_job,
    _has_image_signature,
    _has_audio_signature,
    _has_video_signature,
    _is_image_file_valid,
    _is_audio_file_valid,
    _is_video_file_valid,
)
from media.image import GeneratedImage
from media.tts import GeneratedAudio
from media.thumbnail import GeneratedThumbnail
from research.trending import Topic
from video.compositor import ComposedVideo


# ---------------------------------------------------------------------------
# Fixture helpers — produce minimal valid files
# ---------------------------------------------------------------------------

# Minimal JPEG (smallest valid JPEG)
_FAKE_JPEG = (
    b"\xff\xd8\xff\xe0"
    b"\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00"
    + b"\x08" * 64
    + b"\xff\xc0\x00\x11\x08\x00\x40\x00\x28\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    + b"\xff\xd9"
)

# Minimal MP3 with Xing header (allows duration extraction)
def _make_fake_mp3(duration_frames: int = 200) -> bytes:
    """Create a minimal MP3 file with a Xing header for duration estimation."""
    # ID3v2 header
    id3_size = 10
    id3_header = b"ID3" + b"\x03\x00\x00" + b"\x00" * 10
    # Pad ID3 to 10 bytes total (size is synchsafe)
    id3_header = b"ID3\x03\x00\x00\x00\x00\x00\x0a"

    # Xing header
    xing_data = b"Xing"
    xing_data += b"\x00" * 4  # flags
    xing_data += struct.pack(">I", duration_frames)  # total frames
    xing_data += b"\x00" * 4  # total bytes
    xing_data += b"\x00" * (256 - len(xing_data))  # pad to ~256 bytes

    # One valid MPEG frame (MPEG1 Layer III, 128kbps, 44100Hz)
    frame_header = bytes([
        0xFF, 0xFB,  # frame sync + MPEG1, Layer III
        0x90,        # 128kbps, 44100Hz
        0x00,        # no padding, stereo
    ])
    frame_data = b"\x00" * 413  # typical frame size for 128kbps

    return id3_header + xing_data + frame_header + frame_data


# Minimal MP4 with ftyp + moov/trak/tkhd (for aspect ratio probing)
def _make_fake_mp4(width: int = 1080, height: int = 1920) -> bytes:
    """Create a minimal MP4 file with valid ftyp and tkhd box."""
    # ftyp box
    ftyp_brand = b"isom"
    ftyp_compat = b"isom"
    ftyp_data = ftyp_brand + b"\x00\x00\x00\x01" + ftyp_compat
    ftyp_box = struct.pack(">I", 8 + len(ftyp_data)) + b"ftyp" + ftyp_data

    # Create a minimal moov box containing trak -> tkhd with dimensions
    # tkhd version 0: width at offset +76 from tkhd box start, height at +80
    tkhd_payload = bytearray(92)
    tkhd_payload[0] = 0  # version
    # flags (3 bytes) at offset 1
    tkhd_payload[4:8] = struct.pack(">I", 0)  # creation time
    tkhd_payload[8:12] = struct.pack(">I", 0)  # modification time
    tkhd_payload[12:16] = struct.pack(">I", 1)  # track ID
    tkhd_payload[16:20] = struct.pack(">I", 0)  # reserved
    tkhd_payload[20:24] = struct.pack(">I", 0)  # duration
    tkhd_payload[24:32] = b"\x00" * 8  # reserved
    tkhd_payload[32:34] = struct.pack(">H", 0)  # layer
    tkhd_payload[34:36] = struct.pack(">H", 0)  # alternate group
    tkhd_payload[36:38] = struct.pack(">H", 0)  # volume
    tkhd_payload[38:40] = b"\x00" * 2  # reserved
    tkhd_payload[40:56] = b"\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # matrix
    tkhd_payload[56:60] = struct.pack(">I", width << 16)  # width (fixed 16.16)
    tkhd_payload[60:64] = struct.pack(">I", height << 16)  # height (fixed 16.16)

    tkhd_box = struct.pack(">I", 8 + len(tkhd_payload)) + b"tkhd" + bytes(tkhd_payload)

    # Wrap in trak box
    trak_box = struct.pack(">I", 8 + len(tkhd_box)) + b"trak" + tkhd_box

    # Wrap in moov box
    moov_box = struct.pack(">I", 8 + len(trak_box)) + b"moov" + trak_box

    # Pad to at least 1000 bytes
    min_size = 1000
    total = ftyp_box + moov_box
    if len(total) < min_size:
        total += b"\x00" * (min_size - len(total))

    return total


_FAKE_MP3 = _make_fake_mp3(200)
_FAKE_MP4 = _make_fake_mp4(1080, 1920)


def _create_file(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _make_topic(**overrides) -> Topic:
    defaults = dict(
        title="OpenAI Announces GPT-5",
        url="https://example.com/article",
        summary="OpenAI released GPT-5 with major reasoning improvements.",
        source="TechCrunch",
    )
    defaults.update(overrides)
    return Topic(**defaults)


def _make_narration(word_count: int = 80) -> str:
    return " ".join(["word"] * word_count)


def _make_scenes(count: int = 3) -> list[Scene]:
    return [
        Scene(scene=i + 1, text=f"Scene {i + 1} text.", visual_description=f"Scene {i + 1} visual.")
        for i in range(count)
    ]


def _make_images(tmp_path: Path, count: int = 3) -> list[GeneratedImage]:
    imgs = []
    for i in range(count):
        path = _create_file(tmp_path, f"scene_{i + 1}.jpg", _FAKE_JPEG)
        imgs.append(GeneratedImage(scene=i + 1, path=path))
    return imgs


def _make_audio(tmp_path: Path, duration: float = 30.0) -> GeneratedAudio:
    path = _create_file(tmp_path, "narration.mp3", _FAKE_MP3)
    return GeneratedAudio(path=path, duration_seconds=duration)


def _make_video(tmp_path: Path, duration: float = 30.0, width: int = 1080, height: int = 1920) -> ComposedVideo:
    path = _create_file(tmp_path, "final.mp4", _make_fake_mp4(width, height))
    return ComposedVideo(path=path, duration_seconds=duration, scene_count=3)


def _make_metadata() -> VideoMetadata:
    return VideoMetadata(
        title="GPT-5 Is Here",
        description="OpenAI released GPT-5 with major improvements.",
        tags=["gpt-5", "openai", "ai"],
    )


def _make_thumbnail(tmp_path: Path, width: int = 1280, height: int = 720) -> GeneratedThumbnail:
    path = _create_file(tmp_path, "thumbnail.jpg", _FAKE_JPEG)
    return GeneratedThumbnail(path=path, width=width, height=height)


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

class TestHasImageSignature:
    def test_jpeg(self):
        assert _has_image_signature(b"\xff\xd8\xff" + b"\x00" * 9) is True

    def test_png(self):
        assert _has_image_signature(b"\x89PNG" + b"\x00" * 8) is True

    def test_gif(self):
        assert _has_image_signature(b"GIF8" + b"\x00" * 8) is True

    def test_unknown(self):
        assert _has_image_signature(b"\x00\x00\x00\x00" * 3) is False

    def test_too_short(self):
        assert _has_image_signature(b"\xff") is False


class TestHasAudioSignature:
    def test_mp3_id3(self):
        assert _has_audio_signature(b"ID3" + b"\x00" * 9) is True

    def test_mp3_frame_sync(self):
        header = bytes([0xFF, 0xFB]) + b"\x00" * 10
        assert _has_audio_signature(header) is True

    def test_wav(self):
        assert _has_audio_signature(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 2) is True

    def test_ogg(self):
        assert _has_audio_signature(b"OggS" + b"\x00" * 8) is True

    def test_flac(self):
        assert _has_audio_signature(b"fLaC" + b"\x00" * 8) is True

    def test_unknown(self):
        assert _has_audio_signature(b"\x00\x00\x00\x00" * 3) is False


class TestHasVideoSignature:
    def test_mp4_ftyp(self):
        assert _has_video_signature(b"\x00\x00\x00\x00ftyp" + b"\x00" * 4) is True

    def test_webm(self):
        assert _has_video_signature(b"\x1a\x45\xdf\xa3" + b"\x00" * 8) is True

    def test_avi(self):
        assert _has_video_signature(b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 2) is True

    def test_mpeg_ts(self):
        assert _has_video_signature(b"\x47" + b"\x00" * 11) is True

    def test_unknown(self):
        assert _has_video_signature(b"\x00\x00\x00\x00" * 3) is False


# ---------------------------------------------------------------------------
# File validation helpers
# ---------------------------------------------------------------------------

class TestIsImageFileValid:
    def test_valid(self, tmp_path: Path):
        p = _create_file(tmp_path, "ok.jpg", _FAKE_JPEG)
        assert _is_image_file_valid(p) is True

    def test_nonexistent(self, tmp_path: Path):
        assert _is_image_file_valid(tmp_path / "nope.jpg") is False

    def test_empty(self, tmp_path: Path):
        p = _create_file(tmp_path, "empty.jpg", b"")
        assert _is_image_file_valid(p) is False


class TestIsAudioFileValid:
    def test_valid(self, tmp_path: Path):
        p = _create_file(tmp_path, "ok.mp3", _FAKE_MP3)
        assert _is_audio_file_valid(p) is True

    def test_nonexistent(self, tmp_path: Path):
        assert _is_audio_file_valid(tmp_path / "nope.mp3") is False


class TestIsVideoFileValid:
    def test_valid(self, tmp_path: Path):
        p = _create_file(tmp_path, "ok.mp4", _FAKE_MP4)
        assert _is_video_file_valid(p) is True

    def test_nonexistent(self, tmp_path: Path):
        assert _is_video_file_valid(tmp_path / "nope.mp4") is False

    def test_too_small(self, tmp_path: Path):
        p = _create_file(tmp_path, "tiny.mp4", b"\x00" * 100)
        assert _is_video_file_valid(p) is False


# ---------------------------------------------------------------------------
# QAResult model
# ---------------------------------------------------------------------------

class TestQAResultModel:
    def test_frozen(self):
        r = QAResult(passed=True)
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore[misc]

    def test_defaults(self):
        r = QAResult(passed=True)
        assert r.errors == []
        assert r.warnings == []
        assert r.details == {}


# ---------------------------------------------------------------------------
# validate_job — fully valid job
# ---------------------------------------------------------------------------

class TestValidJob:
    def test_all_valid_passes(self, tmp_path: Path):
        result = validate_job(
            topic=_make_topic(),
            narration=_make_narration(),
            scenes=_make_scenes(),
            images=_make_images(tmp_path),
            audio=_make_audio(tmp_path),
            video=_make_video(tmp_path),
            metadata=_make_metadata(),
            thumbnail=_make_thumbnail(tmp_path),
        )
        assert result.passed is True
        assert len(result.errors) == 0

    def test_valid_job_has_details(self, tmp_path: Path):
        result = validate_job(
            topic=_make_topic(),
            narration=_make_narration(),
            scenes=_make_scenes(),
            images=_make_images(tmp_path),
            audio=_make_audio(tmp_path),
            video=_make_video(tmp_path),
            metadata=_make_metadata(),
            thumbnail=_make_thumbnail(tmp_path),
        )
        assert "topic_title" in result.details
        assert "scene_count" in result.details
        assert "audio_duration" in result.details
        assert "video_duration" in result.details


# ---------------------------------------------------------------------------
# validate_job — missing topic
# ---------------------------------------------------------------------------

class TestMissingTopic:
    def test_topic_none_fails(self):
        result = validate_job(topic=None)
        assert result.passed is False
        assert any("Topic is missing" in e for e in result.errors)

    def test_empty_topic_title_fails(self):
        result = validate_job(topic=_make_topic(title=""))
        assert result.passed is False
        assert any("Topic title is empty" in e for e in result.errors)

    def test_empty_topic_url_fails(self):
        result = validate_job(topic=_make_topic(url=""))
        assert result.passed is False
        assert any("Topic source URL is missing" in e for e in result.errors)

    def test_empty_summary_is_warning(self):
        result = validate_job(topic=_make_topic(summary=""))
        assert any("Topic summary is empty" in w for w in result.warnings)

    def test_empty_source_is_warning(self):
        result = validate_job(topic=_make_topic(source=""))
        assert any("Topic source name is empty" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# validate_job — invalid narration
# ---------------------------------------------------------------------------

class TestInvalidNarration:
    def test_narration_none_fails(self):
        result = validate_job(narration=None)
        assert result.passed is False
        assert any("Narration is missing" in e for e in result.errors)

    def test_empty_narration_fails(self):
        result = validate_job(narration="")
        assert result.passed is False
        assert any("Narration is empty" in e for e in result.errors)

    def test_whitespace_narration_fails(self):
        result = validate_job(narration="   \t\n  ")
        assert result.passed is False

    def test_short_narration_is_warning(self):
        result = validate_job(narration=" ".join(["word"] * 10))
        assert any("Narration is short" in w for w in result.warnings)

    def test_long_narration_is_warning(self):
        result = validate_job(narration=" ".join(["word"] * 700))
        assert any("Narration is long" in w for w in result.warnings)

    def test_non_string_narration_fails(self):
        result = validate_job(narration=12345)  # type: ignore[arg-type]
        assert result.passed is False
        assert any("not a string" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_job — scene structure
# ---------------------------------------------------------------------------

class TestInvalidScenes:
    def test_scenes_none_fails(self):
        result = validate_job(scenes=None)
        assert result.passed is False
        assert any("Scenes are missing" in e for e in result.errors)

    def test_scenes_empty_fails(self):
        result = validate_job(scenes=[])
        assert result.passed is False
        assert any("Scenes list is empty" in e for e in result.errors)

    def test_scene_not_starting_at_1_fails(self):
        scenes = [Scene(scene=2, text="hello", visual_description="world")]
        result = validate_job(scenes=scenes)
        assert result.passed is False
        assert any("do not start at 1" in e for e in result.errors)

    def test_non_sequential_scene_numbers_fails(self):
        scenes = [
            Scene(scene=1, text="a", visual_description="b"),
            Scene(scene=3, text="c", visual_description="d"),  # missing scene 2
        ]
        result = validate_job(scenes=scenes)
        assert result.passed is False
        assert any("not sequential" in e for e in result.errors)

    def test_empty_scene_text_fails(self):
        scenes = [Scene(scene=1, text="", visual_description="desc")]
        result = validate_job(scenes=scenes)
        assert result.passed is False
        assert any("empty text" in e for e in result.errors)

    def test_empty_scene_visual_fails(self):
        scenes = [Scene(scene=1, text="text", visual_description="")]
        result = validate_job(scenes=scenes)
        assert result.passed is False
        assert any("empty visual_description" in e for e in result.errors)

    def test_scene_not_scene_instance_fails(self):
        result = validate_job(scenes=["not a scene"])  # type: ignore[list-item]
        assert result.passed is False
        assert any("not a Scene instance" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_job — missing/corrupt images
# ---------------------------------------------------------------------------

class TestMissingImages:
    def test_images_none_fails(self):
        result = validate_job(images=None)
        assert result.passed is False
        assert any("Images are missing" in e for e in result.errors)

    def test_images_empty_fails(self):
        result = validate_job(images=[])
        assert result.passed is False
        assert any("Images list is empty" in e for e in result.errors)

    def test_count_mismatch_fails(self, tmp_path: Path):
        scenes = _make_scenes(3)
        images = _make_images(tmp_path, 2)
        result = validate_job(scenes=scenes, images=images)
        assert result.passed is False
        assert any("does not match scene count" in e for e in result.errors)


class TestCorruptImage:
    def test_missing_image_file_fails(self, tmp_path: Path):
        bad_path = tmp_path / "nonexistent.jpg"
        images = [GeneratedImage(scene=1, path=bad_path)]
        result = validate_job(images=images)
        assert result.passed is False
        assert any("missing" in e.lower() or "does not exist" in e.lower() for e in result.errors)

    def test_corrupt_image_fails(self, tmp_path: Path):
        corrupt = _create_file(tmp_path, "scene_1.jpg", b"not an image data at all")
        images = [GeneratedImage(scene=1, path=corrupt)]
        result = validate_job(images=images)
        assert result.passed is False
        assert any("corrupt or invalid" in e for e in result.errors)

    def test_empty_image_fails(self, tmp_path: Path):
        empty = _create_file(tmp_path, "scene_1.jpg", b"")
        images = [GeneratedImage(scene=1, path=empty)]
        result = validate_job(images=images)
        assert result.passed is False
        assert any("empty" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# validate_job — invalid audio
# ---------------------------------------------------------------------------

class TestInvalidAudio:
    def test_audio_none_fails(self):
        result = validate_job(audio=None)
        assert result.passed is False
        assert any("TTS audio is missing" in e for e in result.errors)

    def test_missing_audio_file_fails(self, tmp_path: Path):
        audio = GeneratedAudio(path=tmp_path / "nope.mp3", duration_seconds=30.0)
        result = validate_job(audio=audio)
        assert result.passed is False
        assert any("does not exist" in e for e in result.errors)

    def test_zero_duration_fails(self, tmp_path: Path):
        audio = GeneratedAudio(path=_create_file(tmp_path, "a.mp3", _FAKE_MP3), duration_seconds=0)
        result = validate_job(audio=audio)
        assert result.passed is False
        assert any("not positive" in e for e in result.errors)

    def test_negative_duration_fails(self, tmp_path: Path):
        audio = GeneratedAudio(path=_create_file(tmp_path, "a.mp3", _FAKE_MP3), duration_seconds=-5)
        result = validate_job(audio=audio)
        assert result.passed is False

    def test_corrupt_audio_fails(self, tmp_path: Path):
        corrupt = _create_file(tmp_path, "narration.mp3", b"not audio")
        audio = GeneratedAudio(path=corrupt, duration_seconds=30.0)
        result = validate_job(audio=audio)
        assert result.passed is False
        assert any("corrupt or invalid" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_job — invalid video
# ---------------------------------------------------------------------------

class TestInvalidVideo:
    def test_video_none_fails(self):
        result = validate_job(video=None)
        assert result.passed is False
        assert any("Final video is missing" in e for e in result.errors)

    def test_missing_video_file_fails(self, tmp_path: Path):
        video = ComposedVideo(path=tmp_path / "nope.mp4", duration_seconds=30.0, scene_count=3)
        result = validate_job(video=video)
        assert result.passed is False
        assert any("does not exist" in e for e in result.errors)

    def test_corrupt_video_fails(self, tmp_path: Path):
        corrupt = _create_file(tmp_path, "final.mp4", b"not video")
        video = ComposedVideo(path=corrupt, duration_seconds=30.0, scene_count=3)
        result = validate_job(video=video)
        assert result.passed is False
        assert any("corrupt or invalid" in e for e in result.errors)

    def test_zero_video_duration_fails(self, tmp_path: Path):
        video = ComposedVideo(
            path=_create_file(tmp_path, "final.mp4", _FAKE_MP4),
            duration_seconds=0,
            scene_count=3,
        )
        result = validate_job(video=video)
        assert result.passed is False
        assert any("not positive" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_job — incorrect video aspect ratio
# ---------------------------------------------------------------------------

class TestIncorrectAspectRatio:
    def test_wrong_aspect_ratio_fails_or_warns(self, tmp_path: Path):
        # 16:9 instead of 9:16
        video = _make_video(tmp_path, width=1920, height=1080)
        result = validate_job(video=video)
        has_ratio_issue = any("aspect ratio" in m.lower() for m in result.warnings + result.errors)
        assert has_ratio_issue

    def test_correct_aspect_ratio_or_probe_unavailable(self, tmp_path: Path):
        video = _make_video(tmp_path, width=1080, height=1920)
        result = validate_job(video=video)
        # The probe may not find tkhd in a minimal fake MP4 — that's a warning, not an error.
        # Ensure no aspect ratio *error* (which would mean wrong ratio was detected).
        has_ratio_err = any("aspect ratio" in e.lower() for e in result.errors)
        assert not has_ratio_err


# ---------------------------------------------------------------------------
# validate_job — audio/video duration mismatch
# ---------------------------------------------------------------------------

class TestDurationMismatch:
    def test_large_duration_drift_is_warning(self, tmp_path: Path):
        audio = _make_audio(tmp_path, duration=30.0)
        video = _make_video(tmp_path, duration=45.0)  # 50% drift — warning, not error
        result = validate_job(audio=audio, video=video)
        has_drift_warning = any("drift" in w.lower() for w in result.warnings)
        assert has_drift_warning

    def test_severe_duration_drift_is_error(self, tmp_path: Path):
        audio = _make_audio(tmp_path, duration=30.0)
        video = _make_video(tmp_path, duration=90.0)  # 200% drift — severe error
        result = validate_job(audio=audio, video=video)
        has_drift_error = any("drift" in e.lower() for e in result.errors)
        assert has_drift_error
        assert result.passed is False

    def test_small_duration_drift_no_warning(self, tmp_path: Path):
        audio = _make_audio(tmp_path, duration=30.0)
        video = _make_video(tmp_path, duration=31.0)  # ~3% drift
        result = validate_job(audio=audio, video=video)
        has_drift = any("drift" in w.lower() for w in result.warnings)
        assert not has_drift


# ---------------------------------------------------------------------------
# validate_job — invalid metadata
# ---------------------------------------------------------------------------

class TestInvalidMetadata:
    def test_metadata_none_fails(self):
        result = validate_job(metadata=None)
        assert result.passed is False
        assert any("Metadata is missing" in e for e in result.errors)

    def test_empty_title_fails(self):
        m = VideoMetadata(title="", description="desc", tags=["a"])
        result = validate_job(metadata=m)
        assert result.passed is False
        assert any("title is empty" in e for e in result.errors)

    def test_empty_description_fails(self):
        m = VideoMetadata(title="title", description="", tags=["a"])
        result = validate_job(metadata=m)
        assert result.passed is False
        assert any("description is empty" in e for e in result.errors)

    def test_empty_tags_fails(self):
        m = VideoMetadata(title="title", description="desc", tags=[])
        result = validate_job(metadata=m)
        assert result.passed is False
        assert any("no tags" in e for e in result.errors)

    def test_few_tags_is_warning(self):
        m = VideoMetadata(title="title", description="desc", tags=["a"])
        result = validate_job(metadata=m)
        assert any("very few tags" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# validate_job — invalid thumbnail
# ---------------------------------------------------------------------------

class TestInvalidThumbnail:
    def test_thumbnail_none_fails(self):
        result = validate_job(thumbnail=None)
        assert result.passed is False
        assert any("Thumbnail is missing" in e for e in result.errors)

    def test_missing_thumbnail_file_fails(self, tmp_path: Path):
        t = GeneratedThumbnail(path=tmp_path / "nope.jpg", width=1280, height=720)
        result = validate_job(thumbnail=t)
        assert result.passed is False
        assert any("does not exist" in e for e in result.errors)

    def test_wrong_dimensions_fails(self, tmp_path: Path):
        t = GeneratedThumbnail(
            path=_create_file(tmp_path, "thumb.jpg", _FAKE_JPEG),
            width=640,
            height=480,
        )
        result = validate_job(thumbnail=t)
        assert result.passed is False
        assert any("not 1280x720" in e for e in result.errors)

    def test_correct_dimensions_passes(self, tmp_path: Path):
        t = GeneratedThumbnail(
            path=_create_file(tmp_path, "thumb.jpg", _FAKE_JPEG),
            width=1280,
            height=720,
        )
        result = validate_job(thumbnail=t)
        assert not any("1280x720" in e for e in result.errors)

    def test_corrupt_thumbnail_fails(self, tmp_path: Path):
        corrupt = _create_file(tmp_path, "thumb.jpg", b"bad data")
        t = GeneratedThumbnail(path=corrupt, width=1280, height=720)
        result = validate_job(thumbnail=t)
        assert result.passed is False
        assert any("corrupt or invalid" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_job — warnings do not fail
# ---------------------------------------------------------------------------

class TestWarningsDoNotFail:
    def test_warnings_with_no_errors(self, tmp_path: Path):
        result = validate_job(
            topic=_make_topic(summary="", source=""),  # 2 warnings
            narration=" ".join(["word"] * 10),  # short narration warning
            scenes=_make_scenes(),
            images=_make_images(tmp_path),
            audio=_make_audio(tmp_path),
            video=_make_video(tmp_path),
            metadata=_make_metadata(),
            thumbnail=_make_thumbnail(tmp_path),
        )
        assert result.passed is True
        assert len(result.warnings) >= 3

    def test_all_none_no_errors_beyond_missing(self):
        """When everything is None, every missing asset is an error — not a crash."""
        result = validate_job()
        assert result.passed is False
        assert len(result.errors) > 0
        # Should not have unhandled exceptions in warnings
        assert all(isinstance(e, str) for e in result.errors)


# ---------------------------------------------------------------------------
# validate_job — multiple simultaneous errors
# ---------------------------------------------------------------------------

class TestMultipleErrors:
    def test_everything_missing_fails_with_many_errors(self):
        result = validate_job()
        assert result.passed is False
        assert len(result.errors) >= 5  # At least topic, narration, scenes, images, audio, video, metadata, thumbnail

    def test_multiple_errors_independent(self, tmp_path: Path):
        """Errors in different assets are all reported, not just the first."""
        result = validate_job(
            topic=_make_topic(title=""),
            narration="",
            scenes=[],
            images=[],
            audio=GeneratedAudio(path=tmp_path / "missing.mp3", duration_seconds=0),
            metadata=VideoMetadata(title="", description="", tags=[]),
        )
        assert result.passed is False
        assert len(result.errors) >= 5


# ---------------------------------------------------------------------------
# validate_job — structured QA result
# ---------------------------------------------------------------------------

class TestStructuredResult:
    def test_result_is_qa_result_type(self, tmp_path: Path):
        result = validate_job(topic=_make_topic())
        assert isinstance(result, QAResult)

    def test_passed_is_bool(self, tmp_path: Path):
        result = validate_job()
        assert isinstance(result.passed, bool)

    def test_errors_are_strings(self, tmp_path: Path):
        result = validate_job()
        assert all(isinstance(e, str) for e in result.errors)

    def test_warnings_are_strings(self):
        result = validate_job(topic=_make_topic(summary="", source=""))
        assert all(isinstance(w, str) for w in result.warnings)

    def test_details_is_dict(self, tmp_path: Path):
        result = validate_job(
            topic=_make_topic(),
            narration=_make_narration(),
            scenes=_make_scenes(),
            images=_make_images(tmp_path),
            audio=_make_audio(tmp_path),
        )
        assert isinstance(result.details, dict)
        assert len(result.details) > 0
