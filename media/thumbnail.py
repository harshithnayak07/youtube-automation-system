"""Media module — YouTube thumbnail generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from content.scenes import Scene
from core.constants import GENERATED_IMAGES_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_THUMBNAIL_WIDTH = 1280
_THUMBNAIL_HEIGHT = 720
_THUMBNAIL_DIR = OUTPUT_DIR / "thumbnails"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ThumbnailError(Exception):
    """Raised when thumbnail generation fails."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedThumbnail:
    """Result of a thumbnail generation call."""
    path: Path
    width: int
    height: int


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_image_file(path: Path) -> bool:
    """Return True if the file exists, is non-empty, and has an image signature."""
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


def _has_image_signature(header: bytes) -> bool:
    """Check if bytes match known image file signatures."""
    if len(header) < 4:
        return False
    if header[:3] == b"\xff\xd8\xff":  # JPEG
        return True
    if header[:4] == b"\x89PNG":  # PNG
        return True
    if header[:4] == b"GIF8":  # GIF
        return True
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WEBP":  # WebP
        return True
    return False


# ---------------------------------------------------------------------------
# Source image selection
# ---------------------------------------------------------------------------

def _select_best_source_image(
    images: list[Path],
    topic_title: str,
) -> Path | None:
    """Select the best source image for a thumbnail.

    Prefers the first image (scene 1) as it typically shows the main subject.
    """
    if not images:
        return None

    # Prefer scene 1 (first image)
    for img in images:
        if _is_valid_image_file(img) and img.name.startswith("scene_1"):
            return img

    # Fall back to first valid image
    for img in images:
        if _is_valid_image_file(img):
            return img

    return None


# ---------------------------------------------------------------------------
# Thumbnail creation
# ---------------------------------------------------------------------------

def _create_thumbnail_from_source(
    source_path: Path,
    output_path: Path,
    *,
    width: int = _THUMBNAIL_WIDTH,
    height: int = _THUMBNAIL_HEIGHT,
) -> None:
    """Create a thumbnail by resizing/cropping a source image to 1280x720."""
    from PIL import Image

    img = Image.open(source_path)

    # Resize to fill the target aspect ratio
    src_ratio = img.width / img.height
    target_ratio = width / height

    if src_ratio > target_ratio:
        # Source is wider — scale by height, crop width
        new_h = height
        new_w = int(img.width * (height / img.height))
    else:
        # Source is taller — scale by width, crop height
        new_w = width
        new_h = int(img.height * (width / img.width))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop to exact target size
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write
    tmp_path = output_path.with_suffix(".tmp.jpg")
    img.save(tmp_path, "JPEG", quality=90)
    tmp_path.replace(output_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_thumbnail(
    topic_title: str,
    scene_images: list[GeneratedImage] | None = None,
    *,
    output_dir: Path | None = None,
    width: int = _THUMBNAIL_WIDTH,
    height: int = _THUMBNAIL_HEIGHT,
    job_id: str | None = None,
) -> GeneratedThumbnail:
    """Generate a YouTube thumbnail from scene images.

    Parameters
    ----------
    topic_title:
        The topic title (used for logging/context).
    scene_images:
        List of ``GeneratedImage`` objects from image generation.
        If empty or None, a placeholder thumbnail is created.
    output_dir:
        Directory for the thumbnail.  Defaults to ``THUMBNAIL_DIR``.
    width:
        Output width.  Defaults to 1280.
    height:
        Output height.  Defaults to 720.
    job_id:
        Optional run/job identifier.  When provided, the thumbnail is saved
        in a subdirectory named after the job (e.g. ``thumbnails/<job_id>/``),
        preventing stale thumbnails from a previous run from being reused.

    Returns
    -------
    GeneratedThumbnail
        Path and dimensions of the generated thumbnail.

    Raises
    ------
    ThumbnailError
        If generation fails or no valid source image is available.
    """
    base_dir = output_dir or _THUMBNAIL_DIR
    if job_id:
        out_dir = base_dir / job_id
    else:
        out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "thumbnail.jpg"

    # Check for existing valid thumbnail
    if _is_valid_image_file(output_path):
        logger.info("Reusing existing thumbnail: %s", output_path)
        return GeneratedThumbnail(
            path=output_path,
            width=width,
            height=height,
        )

    # Select source image
    source_images = [img.path for img in scene_images] if scene_images else []
    source = _select_best_source_image(source_images, topic_title)

    try:
        if source is None:
            # Create a placeholder thumbnail
            logger.warning("No valid source image, creating placeholder thumbnail")
            _create_placeholder_thumbnail(output_path, width, height)
        else:
            logger.info("Creating thumbnail from source: %s", source)
            _create_thumbnail_from_source(source, output_path, width=width, height=height)
    except Exception as exc:
        # Cleanup failed output
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        tmp = output_path.with_suffix(".tmp.jpg")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise ThumbnailError("Failed to generate valid thumbnail") from exc

    if not _is_valid_image_file(output_path):
        # Cleanup failed output
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        tmp = output_path.with_suffix(".tmp.jpg")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise ThumbnailError("Failed to generate valid thumbnail")

    logger.info("Saved thumbnail: %s", output_path)
    return GeneratedThumbnail(
        path=output_path,
        width=width,
        height=height,
    )


def _create_placeholder_thumbnail(
    output_path: Path,
    width: int,
    height: int,
) -> None:
    """Create a simple placeholder thumbnail."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color=(15, 15, 25))
    draw = ImageDraw.Draw(img)

    # Draw a simple gradient-like background
    for y in range(height):
        r = int(15 + (y / height) * 30)
        g = int(15 + (y / height) * 20)
        b = int(40 + (y / height) * 60)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Draw centered text
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text = "AI NEWS"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.jpg")
    img.save(tmp_path, "JPEG", quality=90)
    tmp_path.replace(output_path)
