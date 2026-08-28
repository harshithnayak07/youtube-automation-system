"""Media module — scene image sourcing via Pexels (free, no AI generation)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from config import settings
from content.scenes import Scene
from core.constants import GENERATED_IMAGES_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
_MAX_RETRIES = 3
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ImageGenError(Exception):
    """Raised when image sourcing fails after retries."""


class ImageGenConfigError(ImageGenError):
    """Raised for missing or invalid configuration (non-retryable)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedImage:
    """Result of a single image sourcing call."""
    scene: int
    path: Path


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
    b"GIF8": "gif",
    b"RIFF": "webp",
}


def _detect_image_format(data: bytes) -> str | None:
    """Detect image format from file signature bytes."""
    for sig, fmt in _IMAGE_SIGNATURES.items():
        if data[:len(sig)] == sig:
            return fmt
    return None


def _validate_image_bytes(data: bytes) -> bool:
    """Return True if *data* looks like a valid image."""
    if len(data) < 100:
        return False
    return _detect_image_format(data) is not None


def _is_image_file_valid(path: Path) -> bool:
    """Return True if an existing file is a non-empty, decodable image."""
    if not path.is_file():
        return False
    try:
        data = path.read_bytes()
        return _validate_image_bytes(data)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------

def _is_transient(exc: Exception) -> bool:
    """Return True if the exception is worth retrying."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    return False


# ---------------------------------------------------------------------------
# Pexels API
# ---------------------------------------------------------------------------

def _build_search_query(visual_description: str) -> str:
    """Build a concise Pexels search query from a scene's visual description.

    Pexels search works best with short, keyword-style queries.
    We extract the core subject matter from the visual description.
    """
    # Strip common cinematic/film phrasing to get searchable keywords
    query = visual_description.strip()
    # Remove leading articles and common prefixes
    for prefix in ("A ", "An ", "The ", "a ", "an ", "the "):
        if query.startswith(prefix):
            query = query[len(prefix):]
    # Strip quotes and special characters that break Pexels API queries
    query = query.replace('"', "").replace("'", "").replace("`", "")
    query = query.replace("(", "").replace(")", "")
    # Limit length — Pexels works well with focused queries
    words = query.split()
    if len(words) > 8:
        query = " ".join(words[:8])
    return query


def _search_pexels(
    query: str,
    *,
    api_key: str,
    orientation: str = "portrait",
    per_page: int = 10,
) -> list[dict]:
    """Search Pexels for photos matching *query*.

    Returns the raw ``photos`` list from the Pexels response.
    """
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "orientation": orientation,
        "per_page": per_page,
        "size": "medium",
    }
    resp = requests.get(_PEXELS_SEARCH_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("photos", [])


def _select_best_photo(photos: list[dict]) -> dict | None:
    """Pick the most relevant photo from Pexels search results.

    Prefers landscape-oriented source images that can be cropped to 9:16,
    falling back to portrait, then any orientation.
    """
    if not photos:
        return None

    portrait = [p for p in photos if p.get("height", 0) > p.get("width", 0)]
    landscape = [p for p in photos if p.get("width", 0) > p.get("height", 0)]

    # Prefer portrait (natural 9:16 fit), then landscape (crop to fit)
    candidates = portrait or landscape or photos
    return candidates[0]


def _download_photo(photo: dict) -> bytes:
    """Download the source image for a Pexels photo.

    Tries URLs in quality order (original → large2x → large), falling back
    to the next if the download fails (e.g. 422, 404).
    """
    src = photo.get("src", {})
    candidates = [
        src.get("original"),
        src.get("large2x"),
        src.get("large"),
    ]
    urls = [u for u in candidates if u]
    if not urls:
        raise ImageGenError("Pexels photo has no downloadable URL")

    last_exc: Exception | None = None
    for url in urls:
        try:
            img_resp = requests.get(url, timeout=60)
            img_resp.raise_for_status()
            return img_resp.content
        except Exception as exc:
            last_exc = exc
            continue

    raise ImageGenError(f"All download URLs failed for Pexels photo: {last_exc}")


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_scene_image(
    scene: Scene,
    *,
    output_dir: Path | None = None,
    api_key: str | None = None,
    max_retries: int = _MAX_RETRIES,
    job_id: str | None = None,
) -> GeneratedImage:
    """Generate or reuse an image for a single scene.

    Parameters
    ----------
    scene:
        A ``Scene`` from the scene planner.
    output_dir:
        Directory for saved images.  Defaults to ``GENERATED_IMAGES_DIR``.
    api_key:
        Pexels API key.  Falls back to ``PEXELS_API_KEY`` env var.
    max_retries:
        Maximum attempts for transient failures.
    job_id:
        Optional run/job identifier.  When provided, images are saved in a
        subdirectory named after the job (e.g. ``output/images/<job_id>/``),
        preventing stale images from a previous run from being reused.

    Returns
    -------
    GeneratedImage
        Scene number and path to the saved image.

    Raises
    ------
    ImageGenConfigError
        If no API key is available.
    ImageGenError
        If sourcing fails after retries or returns invalid data.
    """
    base_dir = output_dir or GENERATED_IMAGES_DIR
    if job_id:
        out_dir = base_dir / job_id
    else:
        out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    key = api_key or settings.PEXELS_API_KEY
    if not key:
        raise ImageGenConfigError("PEXELS_API_KEY is not set")

    image_path = out_dir / f"scene_{scene.scene}.jpg"

    # Reuse existing valid image
    if _is_image_file_valid(image_path):
        logger.info("Reusing existing image for scene %d: %s", scene.scene, image_path)
        return GeneratedImage(scene=scene.scene, path=image_path)

    query = _build_search_query(scene.visual_description)
    logger.info("Searching Pexels for scene %d: %s", scene.scene, query)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            photos = _search_pexels(query, api_key=key)
            photo = _select_best_photo(photos)
            if photo is None:
                raise ImageGenError(f"Pexels returned no results for query: {query}")

            image_bytes = _download_photo(photo)

            if not _validate_image_bytes(image_bytes):
                raise ImageGenError("Pexels returned non-image data")

            # Atomic write: write to temp then rename
            tmp_path = image_path.with_suffix(".tmp")
            tmp_path.write_bytes(image_bytes)
            tmp_path.replace(image_path)

            logger.info(
                "Saved image for scene %d: %s (%d bytes)",
                scene.scene, image_path, len(image_bytes),
            )
            return GeneratedImage(scene=scene.scene, path=image_path)

        except Exception as exc:
            last_exc = exc
            # Clean up any partial file
            if image_path.with_suffix(".tmp").exists():
                image_path.with_suffix(".tmp").unlink(missing_ok=True)
            if image_path.exists() and not _is_image_file_valid(image_path):
                image_path.unlink(missing_ok=True)

            if not _is_transient(exc):
                logger.error(
                    "Image gen failed (non-retryable) for scene %d: %s",
                    scene.scene, exc,
                )
                break

            if attempt < max_retries:
                delay = 1.0 * (2 ** (attempt - 1))
                logger.warning(
                    "Image gen attempt %d failed (transient) for scene %d, retrying in %.1fs: %s",
                    attempt, scene.scene, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "Image gen exhausted %d retries for scene %d: %s",
                    max_retries, scene.scene, exc,
                )

    raise ImageGenError(
        f"Failed to generate image for scene {scene.scene}: {last_exc}"
    ) from last_exc


def generate_all_scene_images(
    scenes: list[Scene],
    *,
    output_dir: Path | None = None,
    api_key: str | None = None,
    max_retries: int = _MAX_RETRIES,
    job_id: str | None = None,
) -> list[GeneratedImage]:
    """Generate images for all scenes.

    Parameters
    ----------
    scenes:
        List of ``Scene`` objects from the scene planner.
    output_dir:
        Directory for saved images.
    api_key:
        Pexels API key.
    max_retries:
        Maximum attempts per scene for transient failures.
    job_id:
        Optional run/job identifier for per-run image isolation.

    Returns
    -------
    list[GeneratedImage]
        One entry per scene, in order.

    Raises
    ------
    ImageGenConfigError
        If no API key is available.
    ImageGenError
        If any scene fails after retries.
    """
    if not scenes:
        raise ValueError("Scenes list must not be empty")

    results: list[GeneratedImage] = []
    for scene in scenes:
        img = generate_scene_image(
            scene,
            output_dir=output_dir,
            api_key=api_key,
            max_retries=max_retries,
            job_id=job_id,
        )
        results.append(img)

    return results
