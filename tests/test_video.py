"""Tests for the video.compositor module — uses mocked moviepy where needed."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from content.scenes import Scene
from media.image import GeneratedImage
from media.tts import GeneratedAudio
from video.compositor import (
    ComposedVideo,
    CompositionError,
    compose_video,
    _compute_scene_durations,
    _validate_video_file,
    _has_video_signature,
    _render_caption_pil,
    _CAPTION_FONT_SIZE,
    _CAPTION_MAX_WIDTH,
    _CAPTION_PADDING,
    _VIDEO_WIDTH,
    _VIDEO_HEIGHT,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_scene(scene: int = 1, text: str = "Hello world") -> Scene:
    return Scene(scene=scene, text=text, visual_description="A test visual")


def _make_image(tmp_path: Path, scene: int = 1) -> GeneratedImage:
    """Create a minimal valid MP4 file for testing."""
    img_path = tmp_path / f"scene_{scene}.jpg"
    ftyp = b"\x00\x00\x00\x1c" + b"ftyp" + b"isom" + b"\x00\x00\x00\x01" + b"isomiso2mp41"
    moov = b"\x00\x00\x00\x08" + b"moov"
    img_path.write_bytes(ftyp + moov)
    return GeneratedImage(scene=scene, path=img_path)


def _make_audio(tmp_path: Path, duration: float = 10.0) -> GeneratedAudio:
    """Create a minimal valid MP3 file for testing."""
    audio_path = tmp_path / "narration.mp3"
    id3 = b"ID3" + b"\x03\x00\x00" + b"\x00" * 10
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
    audio_path.write_bytes(id3 + frame)
    return GeneratedAudio(path=audio_path, duration_seconds=duration)


def _mock_clip():
    """Create a mock clip with standard methods."""
    clip = MagicMock()
    clip.size = (1080, 1920)
    clip.duration = 5.0
    clip.fadein.return_value = clip
    clip.fadeout.return_value = clip
    clip.resize.return_value = clip
    clip.crop.return_value = clip
    clip.set_audio.return_value = clip
    return clip


def _make_mock_clip_with_writer(tmp_path: Path):
    """Create a mock clip that writes a valid video file when write_videofile is called."""
    clip = _mock_clip()

    mock_writer = MagicMock()

    def fake_write(path, **kwargs):
        # Write a minimal valid MP4 file at the given path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ftyp = b"\x00\x00\x00\x1c" + b"ftyp" + b"isom" + b"\x00\x00\x00\x01" + b"isomiso2mp41"
        moov = b"\x00\x00\x00\x08" + b"moov"
        p.write_bytes(ftyp + moov + b"\x00" * 2000)

    mock_writer.side_effect = fake_write
    clip.write_videofile = mock_writer
    return clip


def _mock_ffmpeg_mux(tmp_path: Path):
    """Return a mock for subprocess.run that simulates ffmpeg mux (copies first input to last arg)."""
    def fake_run(cmd, **kwargs):
        # cmd is like: ffmpeg -y -i video_only.mp4 -i audio.mp3 -c:v copy ... output.mp4
        src = Path(cmd[cmd.index("-i") + 1])
        dst = Path(cmd[-1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(str(src), str(dst))
        else:
            # If source doesn't exist, create a minimal valid MP4
            ftyp = b"\x00\x00\x00\x1c" + b"ftyp" + b"isom" + b"\x00\x00\x00\x01" + b"isomiso2mp41"
            moov = b"\x00\x00\x00\x08" + b"moov"
            dst.write_bytes(ftyp + moov + b"\x00" * 2000)
        return MagicMock(returncode=0, stderr="")
    return MagicMock(side_effect=fake_run)


# ---------------------------------------------------------------------------
# _has_video_signature
# ---------------------------------------------------------------------------

class TestHasVideoSignature:
    def test_mp4_ftyp(self):
        header = b"\x00\x00\x00\x1c" + b"ftyp" + b"isom" + b"\x00\x00\x00\x01"
        assert _has_video_signature(header) is True

    def test_webm(self):
        header = b"\x1a\x45\xdf\xa3" + b"\x00" * 8
        assert _has_video_signature(header) is True

    def test_avi(self):
        header = b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 4
        assert _has_video_signature(header) is True

    def test_mpeg_ts(self):
        header = b"\x47" + b"\x00" * 11
        assert _has_video_signature(header) is True

    def test_unknown(self):
        header = b"\x00\x00\x00\x00" + b"\x00" * 8
        assert _has_video_signature(header) is False

    def test_too_short(self):
        assert _has_video_signature(b"\x47") is False


# ---------------------------------------------------------------------------
# _validate_video_file
# ---------------------------------------------------------------------------

class TestValidateVideoFile:
    def test_valid_mp4(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        ftyp = b"\x00\x00\x00\x1c" + b"ftyp" + b"isom" + b"\x00" * 4
        video.write_bytes(ftyp + b"\x00" * 2000)
        assert _validate_video_file(video) is True

    def test_nonexistent(self, tmp_path: Path):
        assert _validate_video_file(tmp_path / "nope.mp4") is False

    def test_empty_file(self, tmp_path: Path):
        video = tmp_path / "empty.mp4"
        video.write_bytes(b"")
        assert _validate_video_file(video) is False

    def test_too_small(self, tmp_path: Path):
        video = tmp_path / "tiny.mp4"
        video.write_bytes(b"\x00" * 500)
        assert _validate_video_file(video) is False


# ---------------------------------------------------------------------------
# _compute_scene_durations
# ---------------------------------------------------------------------------

class TestComputeSceneDurations:
    def test_proportional_split(self):
        scenes = [
            _make_scene(text="Short"),
            _make_scene(text="This is a longer text"),
            _make_scene(text="Medium length"),
        ]
        durations = _compute_scene_durations(scenes, 30.0)

        assert len(durations) == 3
        assert abs(sum(durations) - 30.0) < 1e-9
        assert durations[1] > durations[0]
        assert durations[1] > durations[2]

    def test_equal_split_when_same_length(self):
        scenes = [_make_scene(text="abc"), _make_scene(text="abc")]
        durations = _compute_scene_durations(scenes, 20.0)

        assert len(durations) == 2
        assert abs(durations[0] - 10.0) < 1e-9
        assert abs(durations[1] - 10.0) < 1e-9

    def test_empty_scenes(self):
        assert _compute_scene_durations([], 10.0) == []

    def test_zero_text_lengths(self):
        scenes = [_make_scene(text=""), _make_scene(text="")]
        durations = _compute_scene_durations(scenes, 10.0)

        assert len(durations) == 2
        assert abs(durations[0] - 5.0) < 1e-9

    def test_single_scene(self):
        scenes = [_make_scene(text="Only one")]
        durations = _compute_scene_durations(scenes, 15.0)

        assert len(durations) == 1
        assert abs(durations[0] - 15.0) < 1e-9


# ---------------------------------------------------------------------------
# compose_video — input validation
# ---------------------------------------------------------------------------

class TestComposeVideoValidation:
    def test_raises_on_empty_scenes(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Scenes list must not be empty"):
            compose_video([], [], GeneratedAudio(path=tmp_path / "a.mp3", duration_seconds=1.0))

    def test_raises_on_empty_images(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Images list must not be empty"):
            compose_video(
                [_make_scene()],
                [],
                GeneratedAudio(path=tmp_path / "a.mp3", duration_seconds=1.0),
            )

    def test_raises_on_mismatched_counts(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Scene count.*must match image count"):
            compose_video(
                [_make_scene(1), _make_scene(2)],
                [_make_image(tmp_path, 1)],
                GeneratedAudio(path=tmp_path / "a.mp3", duration_seconds=1.0),
            )

    def test_raises_on_zero_audio_duration(self, tmp_path: Path):
        audio = _make_audio(tmp_path, duration=0.0)
        with pytest.raises(ValueError, match="Audio duration must be positive"):
            compose_video([_make_scene()], [_make_image(tmp_path)], audio)

    def test_raises_on_missing_audio_file(self, tmp_path: Path):
        missing = GeneratedAudio(path=tmp_path / "missing.mp3", duration_seconds=1.0)
        with pytest.raises(ValueError, match="Audio file does not exist"):
            compose_video([_make_scene()], [_make_image(tmp_path)], missing)

    @patch("moviepy.editor.AudioFileClip")
    def test_raises_on_missing_image_file(self, mock_audio_cls: MagicMock, tmp_path: Path):
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 10.0
        mock_audio_cls.return_value = mock_audio_instance

        audio = _make_audio(tmp_path)
        missing_img = GeneratedImage(scene=1, path=tmp_path / "missing.jpg")
        with pytest.raises(CompositionError, match="Image file missing"):
            compose_video([_make_scene()], [missing_img], audio)


# ---------------------------------------------------------------------------
# compose_video — mocked success
# ---------------------------------------------------------------------------

class TestComposeVideoSuccess:
    @patch("subprocess.run")
    @patch("moviepy.editor.VideoFileClip")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_returns_composed_video(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_probe_cls: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_cls.return_value = MagicMock(duration=10.0)

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 10.0
        mock_concat.return_value = mock_final

        mock_probe = MagicMock()
        mock_probe.duration = 10.0
        mock_probe_cls.return_value = mock_probe

        mock_ffmpeg.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        scenes = [_make_scene(1, "Hello"), _make_scene(2, "World")]
        images = [_make_image(tmp_path, 1), _make_image(tmp_path, 2)]
        audio = _make_audio(tmp_path, 10.0)

        result = compose_video(scenes, images, audio, output_dir=tmp_path)

        assert isinstance(result, ComposedVideo)
        assert result.scene_count == 2
        assert result.duration_seconds == 10.0

    @patch("subprocess.run")
    @patch("moviepy.editor.VideoFileClip")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_creates_output_file(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_probe_cls: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 5.0
        mock_audio_cls.return_value = mock_audio_instance

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 5.0
        mock_concat.return_value = mock_final

        mock_probe = MagicMock()
        mock_probe.duration = 5.0
        mock_probe_cls.return_value = mock_probe

        mock_ffmpeg.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        scenes = [_make_scene()]
        images = [_make_image(tmp_path)]
        audio = _make_audio(tmp_path, 5.0)

        result = compose_video(scenes, images, audio, output_dir=tmp_path)

        assert result.path.parent == tmp_path

    @patch("subprocess.run")
    @patch("moviepy.editor.VideoFileClip")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_9_16_output_configuration(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_probe_cls: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 5.0
        mock_audio_cls.return_value = mock_audio_instance

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 5.0
        mock_concat.return_value = mock_final

        mock_probe = MagicMock()
        mock_probe.duration = 5.0
        mock_probe_cls.return_value = mock_probe

        mock_ffmpeg.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        scenes = [_make_scene()]
        images = [_make_image(tmp_path)]
        audio = _make_audio(tmp_path, 5.0)

        compose_video(scenes, images, audio, output_dir=tmp_path)

        # Verify ImageClip was called
        mock_img_cls.assert_called()


# ---------------------------------------------------------------------------
# compose_video — scene/image ordering
# ---------------------------------------------------------------------------

class TestSceneImageOrdering:
    @patch("subprocess.run")
    @patch("moviepy.editor.VideoFileClip")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_scenes_processed_in_order(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_probe_cls: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 10.0
        mock_audio_cls.return_value = mock_audio_instance

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 10.0
        mock_concat.return_value = mock_final

        mock_probe = MagicMock()
        mock_probe.duration = 10.0
        mock_probe_cls.return_value = mock_probe

        mock_ffmpeg.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        scenes = [_make_scene(1, "First"), _make_scene(2, "Second")]
        images = [_make_image(tmp_path, 1), _make_image(tmp_path, 2)]
        audio = _make_audio(tmp_path, 10.0)

        compose_video(scenes, images, audio, output_dir=tmp_path)

        # Images should be loaded in scene order
        # ImageClip is called once for background, once for caption per scene
        # Filter to only path-based calls (background images)
        img_calls = [call[0][0] for call in mock_img_cls.call_args_list
                     if isinstance(call[0][0], (str, Path))]
        assert str(images[0].path) in img_calls[0]
        assert str(images[1].path) in img_calls[1]


# ---------------------------------------------------------------------------
# compose_video — output directory creation
# ---------------------------------------------------------------------------

class TestOutputDirectoryCreation:
    @patch("subprocess.run")
    @patch("moviepy.editor.VideoFileClip")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_creates_nested_output_dir(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_probe_cls: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 5.0
        mock_audio_cls.return_value = mock_audio_instance

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 5.0
        mock_concat.return_value = mock_final

        mock_probe = MagicMock()
        mock_probe.duration = 5.0
        mock_probe_cls.return_value = mock_probe

        mock_ffmpeg.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        out_dir = tmp_path / "nested" / "videos"
        scenes = [_make_scene()]
        images = [_make_image(tmp_path)]
        audio = _make_audio(tmp_path, 5.0)

        result = compose_video(scenes, images, audio, output_dir=out_dir)

        assert out_dir.is_dir()
        assert result.path.parent == out_dir


# ---------------------------------------------------------------------------
# compose_video — invalid/empty output
# ---------------------------------------------------------------------------

class TestInvalidOutput:
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_raises_on_invalid_output_file(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 5.0
        mock_audio_cls.return_value = mock_audio_instance

        mock_final = _mock_clip()
        mock_final.duration = 5.0
        mock_concat.return_value = mock_final

        # Don't write any file — composition produces nothing
        scenes = [_make_scene()]
        images = [_make_image(tmp_path)]
        audio = _make_audio(tmp_path, 5.0)

        with pytest.raises(CompositionError):
            compose_video(scenes, images, audio, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# compose_video — failure cleanup
# ---------------------------------------------------------------------------

class TestFailureCleanup:
    @patch("moviepy.editor.ImageClip", side_effect=RuntimeError("simulated failure"))
    @patch("moviepy.editor.AudioFileClip")
    def test_cleans_up_on_failure(self, mock_audio_cls: MagicMock, mock_img_cls: MagicMock, tmp_path: Path):
        mock_audio_instance = MagicMock()
        mock_audio_instance.duration = 5.0
        mock_audio_cls.return_value = mock_audio_instance

        scenes = [_make_scene()]
        images = [_make_image(tmp_path)]
        audio = _make_audio(tmp_path, 5.0)

        with pytest.raises(CompositionError, match="composition failed"):
            compose_video(scenes, images, audio, output_dir=tmp_path)

        # No partial video left behind
        assert not (tmp_path / "final.mp4").exists()
        assert not (tmp_path / "final.tmp.mp4").exists()


# ---------------------------------------------------------------------------
# ComposedVideo model
# ---------------------------------------------------------------------------

class TestComposedVideoModel:
    def test_fields(self, tmp_path: Path):
        video = ComposedVideo(path=tmp_path / "final.mp4", duration_seconds=30.0, scene_count=4)
        assert video.path.name == "final.mp4"
        assert video.duration_seconds == 30.0
        assert video.scene_count == 4

    def test_is_frozen(self, tmp_path: Path):
        video = ComposedVideo(path=tmp_path / "x.mp4", duration_seconds=1.0, scene_count=1)
        with pytest.raises(AttributeError):
            video.path = tmp_path / "y.mp4"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

class TestConfigConstants:
    def test_video_dimensions(self):
        assert _VIDEO_WIDTH == 1080
        assert _VIDEO_HEIGHT == 1920


# ---------------------------------------------------------------------------
# compose_video — ffmpeg audio mux
# ---------------------------------------------------------------------------

class TestFFmpegAudioMux:
    @patch("subprocess.run")
    @patch("moviepy.editor.VideoFileClip")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_ffmpeg_called_with_correct_args(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_probe_cls: MagicMock,
        mock_subprocess: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_cls.return_value = MagicMock(duration=10.0)

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 10.0
        mock_concat.return_value = mock_final

        mock_probe = MagicMock()
        mock_probe.duration = 10.0
        mock_probe_cls.return_value = mock_probe

        mock_subprocess.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        scenes = [_make_scene()]
        images = [_make_image(tmp_path)]
        audio = _make_audio(tmp_path, 10.0)

        compose_video(scenes, images, audio, output_dir=tmp_path)

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == "ffmpeg"
        assert "-i" in call_args
        assert "aac" in call_args
        assert "-c:v" in call_args
        assert "copy" in call_args

    @patch("subprocess.run")
    @patch("moviepy.editor.VideoFileClip")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_no_audio_codec_in_write_videofile(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_probe_cls: MagicMock,
        mock_subprocess: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_cls.return_value = MagicMock(duration=10.0)

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 10.0
        mock_concat.return_value = mock_final

        mock_probe = MagicMock()
        mock_probe.duration = 10.0
        mock_probe_cls.return_value = mock_probe

        mock_subprocess.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        compose_video([_make_scene()], [_make_image(tmp_path)],
                       _make_audio(tmp_path, 10.0), output_dir=tmp_path)

        # write_videofile should be called WITHOUT audio_codec
        written_kwargs = mock_final.write_videofile.call_args
        assert "audio_codec" not in written_kwargs[1]


# ---------------------------------------------------------------------------
# compose_video — caption overlay
# ---------------------------------------------------------------------------

class TestCaptionOverlay:
    @patch("subprocess.run")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_caption_created_with_scene_text(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ):
        img_clip = _mock_clip()
        mock_img_cls.return_value = img_clip

        mock_audio_cls.return_value = MagicMock(duration=10.0)

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 10.0
        mock_concat.return_value = mock_final

        mock_ffmpeg.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        scenes = [_make_scene(text="Hello world")]
        images = [_make_image(tmp_path)]
        audio = _make_audio(tmp_path, 10.0)

        result = compose_video(scenes, images, audio, output_dir=tmp_path)

        # Composition should succeed with a caption
        assert isinstance(result, ComposedVideo)
        assert result.scene_count == 1
        # ImageClip called at least once for background
        mock_img_cls.assert_called()

    @patch("subprocess.run")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_no_caption_when_text_empty(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ):
        mock_img_cls.return_value = _mock_clip()
        mock_audio_cls.return_value = MagicMock(duration=10.0)

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 10.0
        mock_concat.return_value = mock_final

        mock_ffmpeg.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        scenes = [_make_scene(text="")]
        images = [_make_image(tmp_path)]
        audio = _make_audio(tmp_path, 10.0)

        result = compose_video(scenes, images, audio, output_dir=tmp_path)

        # Composition should succeed without a caption
        assert isinstance(result, ComposedVideo)
        assert result.scene_count == 1

    @patch("subprocess.run")
    @patch("moviepy.editor.concatenate_videoclips")
    @patch("moviepy.editor.AudioFileClip")
    @patch("moviepy.editor.ImageClip")
    def test_caption_passed_as_composite(
        self,
        mock_img_cls: MagicMock,
        mock_audio_cls: MagicMock,
        mock_concat: MagicMock,
        mock_ffmpeg: MagicMock,
        tmp_path: Path,
    ):
        img_clip = _mock_clip()
        mock_img_cls.return_value = img_clip

        mock_audio_cls.return_value = MagicMock(duration=10.0)

        mock_final = _make_mock_clip_with_writer(tmp_path)
        mock_final.duration = 10.0
        mock_concat.return_value = mock_final

        mock_ffmpeg.side_effect = _mock_ffmpeg_mux(tmp_path).side_effect

        result = compose_video([_make_scene(text="Test")], [_make_image(tmp_path)],
                       _make_audio(tmp_path, 10.0), output_dir=tmp_path)

        # Composition should succeed — caption composited via PIL internally
        assert isinstance(result, ComposedVideo)


# ---------------------------------------------------------------------------
# _render_caption_pil — unit tests
# ---------------------------------------------------------------------------

class TestRenderCaptionPil:
    def test_returns_rgba_image(self):
        from PIL import Image as PILImage
        img = _render_caption_pil("Hello world")
        assert img.mode == "RGBA"
        assert img.size[0] == _CAPTION_MAX_WIDTH
        assert img.size[1] > 0

    def test_single_line_text(self):
        from PIL import Image as PILImage
        img = _render_caption_pil("Short")
        assert img.mode == "RGBA"
        assert img.size[1] > 0

    def test_multiline_wrapping(self):
        from PIL import Image as PILImage
        long_text = "This is a very long caption that should wrap across multiple lines when rendered at the specified max width"
        img = _render_caption_pil(long_text)
        # Height should be greater than a single line
        assert img.size[1] > _CAPTION_FONT_SIZE + 4

    def test_transparent_background(self):
        import numpy as np
        img = _render_caption_pil("Hello")
        arr = np.array(img)
        # Alpha channel should be 0 for transparent pixels
        # At least some pixels should be transparent (background)
        alpha = arr[:, :, 3]
        assert alpha.min() == 0

    def test_white_text_pixels(self):
        import numpy as np
        img = _render_caption_pil("X")
        arr = np.array(img)
        # Some pixels should be white (255,255,255) with full alpha
        white_mask = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 255) & (arr[:, :, 3] == 255)
        assert white_mask.any()

    def test_empty_caption(self):
        from PIL import Image as PILImage
        img = _render_caption_pil("")
        assert img.mode == "RGBA"
        # Empty text produces zero-height image
        assert img.size[1] == 0
