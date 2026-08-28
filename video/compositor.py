"""Video module — composition of scenes into a YouTube Shorts video."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from content.scenes import Scene
from core.constants import FINAL_VIDEOS_DIR
from media.image import GeneratedImage
from media.tts import GeneratedAudio

# Pillow 11+ removed Image.ANTIALIAS; moviepy 1.x still references it.
try:
    from PIL import Image as _PILImage
    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.LANCZOS  # type: ignore[attr-defined]
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VIDEO_WIDTH = 1080
_VIDEO_HEIGHT = 1920
_FPS = 30
_TRANSITION_DURATION = 0.3  # seconds for crossfade
_CAPTION_FONT_SIZE = 48
_CAPTION_PADDING = 30  # px from bottom edge
_CAPTION_MAX_WIDTH = 1020  # px — 30px margin each side on 1080


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CompositionError(Exception):
    """Raised when video composition fails."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComposedVideo:
    """Result of a video composition call."""
    path: Path
    duration_seconds: float
    scene_count: int


# ---------------------------------------------------------------------------
# Scene timing
# ---------------------------------------------------------------------------

def _compute_scene_durations(
    scenes: list[Scene],
    total_duration: float,
) -> list[float]:
    """Divide *total_duration* proportionally by scene text length."""
    if not scenes:
        return []

    text_lengths = [len(s.text) for s in scenes]
    total_length = sum(text_lengths)

    if total_length == 0:
        # Fallback: equal distribution
        equal_dur = total_duration / len(scenes)
        return [equal_dur] * len(scenes)

    durations = [(length / total_length) * total_duration for length in text_lengths]

    # Adjust rounding误差 so sum == total_duration exactly
    current_sum = sum(durations)
    diff = total_duration - current_sum
    if abs(diff) > 1e-9:
        durations[-1] += diff

    return durations


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _render_caption_pil(
    caption: str,
    *,
    max_width: int = _CAPTION_MAX_WIDTH,
    font_size: int = _CAPTION_FONT_SIZE,
) -> "PILImage.Image":
    """Render caption text into a transparent RGBA PIL Image (word-wrapped, centered).

    Returns a PIL Image with white text on a transparent background.
    The image width equals *max_width* and height fits the wrapped text.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    # --- Load Arial font (with fallbacks) ---
    font = None
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # --- Word-wrap to max_width ---
    words = caption.split()
    lines: list[str] = []
    current_line: list[str] = []
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = font.getbbox(test_line)
        line_w = bbox[2] - bbox[0]
        if line_w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    # --- Measure text height ---
    line_height = font_size + 4  # small inter-line spacing
    total_height = line_height * len(lines)

    # --- Draw on transparent canvas ---
    img = PILImage.new("RGBA", (max_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = 0
    for line in lines:
        bbox = font.getbbox(line)
        line_w = bbox[2] - bbox[0]
        x = (max_width - line_w) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height

    return img


def _prepare_image_clip(
    image_path: Path,
    duration: float,
    *,
    caption: str = "",
    width: int = _VIDEO_WIDTH,
    height: int = _VIDEO_HEIGHT,
    fps: int = _FPS,
):
    """Load an image, resize/crop to fill 9:16 frame, and optionally overlay a caption.

    Captions are pre-rendered once per scene using PIL and composited directly
    onto the background image — no per-frame MoviePy compositing.
    """
    from PIL import Image as PILImage
    from moviepy.editor import ImageClip

    clip = ImageClip(str(image_path), duration=duration)

    # Resize to fill the frame while maintaining aspect ratio
    img_w, img_h = clip.size
    target_ratio = width / height
    img_ratio = img_w / img_h

    if img_ratio > target_ratio:
        # Image is wider — scale by height, crop width
        new_h = height
        new_w = int(img_w * (height / img_h))
    else:
        # Image is taller — scale by width, crop height
        new_w = width
        new_h = int(img_h * (width / img_w))

    clip = clip.resize((new_w, new_h))

    # Center crop to exact target size
    x_center = new_w // 2
    y_center = new_h // 2
    x1 = x_center - width // 2
    y1 = y_center - height // 2
    clip = clip.crop(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)

    # --- Caption overlay (PIL pre-render → composite onto background) ---
    if caption:
        try:
            import numpy as np

            # Get the background image as a PIL RGBA image
            bg_arr = clip.get_frame(0)
            bg_pil = PILImage.fromarray(bg_arr).convert("RGBA")

            # Render caption text
            caption_img = _render_caption_pil(caption)

            # Paste caption onto background at correct position
            txt_y = height - _CAPTION_PADDING - caption_img.size[1]
            txt_x = (width - caption_img.size[0]) // 2
            bg_pil.paste(caption_img, (txt_x, txt_y), caption_img)

            # Replace clip with the composited image
            composited_arr = np.array(bg_pil.convert("RGB"))
            clip = ImageClip(composited_arr, duration=duration)
        except Exception:
            logger.debug("PIL caption rendering failed, skipping caption overlay")

    return clip


def _apply_fade_transition(clip, duration: float, fade_duration: float):
    """Apply fade-in and fade-out to a clip."""
    fade = min(fade_duration, duration / 3)
    clip = clip.fadein(fade).fadeout(fade)
    return clip


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_video_file(path: Path) -> bool:
    """Return True if the file exists, is non-empty, and looks like a video."""
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if size < 1000:
            return False
        # Check for common video container signatures
        with open(path, "rb") as f:
            header = f.read(12)
        return _has_video_signature(header)
    except OSError:
        return False


def _has_video_signature(header: bytes) -> bool:
    """Check if bytes match known video file signatures."""
    if len(header) < 4:
        return False
    # MP4/MOV: ftyp at offset 4
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return True
    # WebM: starts with 0x1A 0x45 0xDF 0xA3 (EBML header)
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return True
    # AVI: starts with RIFF....AVI
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"AVI ":
        return True
    # MPEG-TS: starts with 0x47
    if header[0] == 0x47:
        return True
    return False


# ---------------------------------------------------------------------------
# Main composition
# ---------------------------------------------------------------------------

def compose_video(
    scenes: list[Scene],
    images: list[GeneratedImage],
    audio: GeneratedAudio,
    *,
    output_dir: Path | None = None,
    width: int = _VIDEO_WIDTH,
    height: int = _VIDEO_HEIGHT,
    fps: int = _FPS,
    transition: bool = True,
) -> ComposedVideo:
    """Compose scenes, images, and audio into a YouTube Shorts video.

    Parameters
    ----------
    scenes:
        Ordered list of ``Scene`` objects.
    images:
        Corresponding ``GeneratedImage`` objects (same count as scenes).
    audio:
        ``GeneratedAudio`` with path and duration.
    output_dir:
        Directory for the final video.  Defaults to ``FINAL_VIDEOS_DIR``.
    width:
        Output width in pixels.  Defaults to 1080.
    height:
        Output height in pixels.  Defaults to 1920.
    fps:
        Frames per second.  Defaults to 30.
    transition:
        Whether to add crossfade transitions.  Defaults to True.

    Returns
    -------
    ComposedVideo
        Path, duration, and scene count of the final video.

    Raises
    ------
    ValueError
        If inputs are inconsistent (empty, mismatched counts, etc.).
    CompositionError
        If composition fails or produces invalid output.
    """
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    # --- Validate inputs ---
    if not scenes:
        raise ValueError("Scenes list must not be empty")
    if not images:
        raise ValueError("Images list must not be empty")
    if len(scenes) != len(images):
        raise ValueError(
            f"Scene count ({len(scenes)}) must match image count ({len(images)})"
        )
    if audio.duration_seconds <= 0:
        raise ValueError(f"Audio duration must be positive: {audio.duration_seconds}")
    if not audio.path.is_file():
        raise ValueError(f"Audio file does not exist: {audio.path}")

    out_dir = output_dir or FINAL_VIDEOS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "final.mp4"

    try:
        # Load audio clip to get actual duration from the file,
        # avoiding any mismatch between reported and actual MP3 duration.
        audio_clip = AudioFileClip(str(audio.path))
        total_duration = audio_clip.duration
        logger.info(
            "Composing %d scenes into %.1fs video (%dx%d @ %d fps)",
            len(scenes), total_duration, width, height, fps,
        )

        # --- Compute scene durations ---
        durations = _compute_scene_durations(scenes, total_duration)

        # --- Build image clips ---
        clips = []
        for scene, img, dur in zip(scenes, images, durations):
            if not img.path.is_file():
                raise CompositionError(f"Image file missing for scene {scene.scene}: {img.path}")

            clip = _prepare_image_clip(
                img.path, dur, caption=scene.text,
                width=width, height=height, fps=fps,
            )

            if transition and dur > _TRANSITION_DURATION * 2:
                clip = _apply_fade_transition(clip, dur, _TRANSITION_DURATION)

            clips.append(clip)

        # --- Concatenate clips ---
        final_clip = concatenate_videoclips(clips, method="compose")

        # --- Write video only (no audio) ---
        tmp_path = output_path.with_suffix(".tmp.mp4")
        tmp_video_only = output_path.with_suffix(".tmp_video.mp4")
        final_clip.write_videofile(
            str(tmp_video_only),
            fps=fps,
            codec="libx264",
            threads=4,
            logger=None,
        )

        # Cleanup moviepy clips
        final_clip.close()
        for c in clips:
            c.close()
        audio_clip.close()

        # --- Mux audio via ffmpeg (bypasses moviepy audio handling) ---
        import subprocess
        mux_result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(tmp_video_only),
                "-i", str(audio.path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-ar", "44100",
                "-ac", "2",
                "-b:a", "192k",
                "-shortest",
                str(tmp_path),
            ],
            capture_output=True, text=True,
        )
        # Clean up video-only temp file
        if tmp_video_only.exists():
            tmp_video_only.unlink(missing_ok=True)
        if mux_result.returncode != 0:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise CompositionError(f"ffmpeg audio mux failed: {mux_result.stderr[-500:]}")

        # --- Validate output ---
        if not _validate_video_file(tmp_path):
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise CompositionError("Composition produced invalid video file")

        # Atomic rename
        tmp_path.replace(output_path)

        # Verify duration
        actual_duration = _probe_duration(output_path)
        if actual_duration > 0:
            drift = abs(actual_duration - total_duration)
            if drift > 5.0:
                logger.warning(
                    "Video duration drift: expected %.1fs, got %.1fs (Δ%.1fs)",
                    total_duration, actual_duration, drift,
                )
            final_duration = actual_duration
        else:
            final_duration = total_duration

        logger.info(
            "Saved video: %s (%.1fs, %d bytes)",
            output_path, final_duration, output_path.stat().st_size,
        )
        return ComposedVideo(
            path=output_path,
            duration_seconds=final_duration,
            scene_count=len(scenes),
        )

    except CompositionError:
        raise
    except Exception as exc:
        # Cleanup partial output
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        for suffix in (".tmp.mp4", ".tmp_video.mp4"):
            tmp = output_path.with_suffix(suffix)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        raise CompositionError(f"Video composition failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Duration probing
# ---------------------------------------------------------------------------

def _probe_duration(path: Path) -> float:
    """Probe video duration using moviepy."""
    try:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(str(path))
        duration = clip.duration
        clip.close()
        return duration
    except Exception:
        return 0.0
