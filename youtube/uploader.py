"""YouTube module — OAuth authentication and resumable video upload."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from config import settings
from content.metadata import VideoMetadata
from core.qa import QAResult
from media.thumbnail import GeneratedThumbnail

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YT_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
_YT_THUMBNAIL_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
_YT_TOKEN_URL = "https://oauth2.googleapis.com/token"

_DEFAULT_CATEGORY_ID = "28"
_DEFAULT_PRIVACY_STATUS = "private"
_DEFAULT_CHUNK_SIZE = 1024 * 1024 * 5  # 5 MB
_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 2.0

_RETRYABLE_UPLOAD_CODES = frozenset({408, 429, 500, 502, 503, 504})
_NON_RETRYABLE_CODES = frozenset({400, 401, 403, 404})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UploadConfig:
    """YouTube upload configuration loaded from environment."""
    client_id: str
    client_secret: str
    refresh_token: str
    category_id: str = _DEFAULT_CATEGORY_ID
    privacy_status: str = _DEFAULT_PRIVACY_STATUS
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    max_retries: int = _MAX_RETRIES


def load_upload_config() -> UploadConfig:
    """Load YouTube upload configuration from environment variables.

    Returns
    -------
    UploadConfig
        Validated configuration object.

    Raises
    ------
    UploadConfigError
        If required credentials are missing.
    """
    client_id = settings.YOUTUBE_CLIENT_ID
    client_secret = settings.YOUTUBE_CLIENT_SECRET
    refresh_token = settings.YOUTUBE_REFRESH_TOKEN

    missing = []
    if not client_id:
        missing.append("YOUTUBE_CLIENT_ID")
    if not client_secret:
        missing.append("YOUTUBE_CLIENT_SECRET")
    if not refresh_token:
        missing.append("YOUTUBE_REFRESH_TOKEN")

    if missing:
        raise UploadConfigError(
            f"Missing required YouTube credentials: {', '.join(missing)}. "
            "Set them in your .env file or environment."
        )

    return UploadConfig(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class UploadError(Exception):
    """Raised when video upload fails."""


class UploadConfigError(UploadError):
    """Raised for missing or invalid upload configuration."""


class UploadAuthError(UploadError):
    """Raised when OAuth authentication fails."""


class UploadPermissionError(UploadError):
    """Raised when the YouTube API returns a 403 permission error."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UploadResult:
    """Structured result of a YouTube upload operation."""
    video_id: str
    video_url: str
    video_status: str
    thumbnail_status: str
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OAuth token refresh
# ---------------------------------------------------------------------------

def _refresh_access_token(config: UploadConfig, *, session: requests.Session | None = None) -> str:
    """Refresh the OAuth2 access token using stored credentials.

    Parameters
    ----------
    config:
        Upload configuration with client credentials.
    session:
        Optional requests session for connection pooling.

    Returns
    -------
    str
        A fresh access token.

    Raises
    ------
    UploadAuthError
        If the token refresh fails.
    """
    sess = session or requests.Session()

    payload = {
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "refresh_token": config.refresh_token,
        "grant_type": "refresh_token",
    }

    try:
        resp = sess.post(_YT_TOKEN_URL, data=payload, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status in (400, 401):
            raise UploadAuthError(
                "OAuth token refresh failed — invalid or expired credentials. "
                "Re-authorize your YouTube OAuth credentials."
            ) from exc
        raise UploadAuthError(f"OAuth token refresh failed (HTTP {status}): {exc}") from exc
    except requests.RequestException as exc:
        raise UploadAuthError(f"OAuth token refresh network error: {exc}") from exc

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        raise UploadAuthError("OAuth token refresh returned no access_token")

    logger.info("YouTube OAuth access token refreshed successfully")
    return access_token


# ---------------------------------------------------------------------------
# Video upload (resumable)
# ---------------------------------------------------------------------------

def _init_resumable_upload(
    *,
    video_metadata: VideoMetadata,
    file_size: int,
    access_token: str,
    config: UploadConfig,
    session: requests.Session | None = None,
) -> str:
    """Initiate a resumable upload session and return the upload URI.

    Parameters
    ----------
    video_metadata:
        Title, description, and tags for the video.
    file_size:
        Size of the video file in bytes.
    access_token:
        Valid OAuth2 access token.
    config:
        Upload configuration.
    session:
        Optional requests session.

    Returns
    -------
    str
        The resumable upload URI for subsequent PUT requests.

    Raises
    ------
    UploadError
        If session initiation fails.
    """
    sess = session or requests.Session()

    body = {
        "snippet": {
            "title": video_metadata.title,
            "description": video_metadata.description,
            "tags": video_metadata.tags,
            "categoryId": config.category_id,
        },
        "status": {
            "privacyStatus": config.privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Length": str(file_size),
        "X-Upload-Content-Type": "video/mp4",
    }

    params = {
        "uploadType": "resumable",
        "part": "snippet,status",
    }

    try:
        resp = sess.post(
            _YT_UPLOAD_URL,
            json=body,
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        body_text = ""
        if exc.response is not None:
            try:
                body_text = exc.response.text[:500]
            except Exception:
                pass
        if status == 401:
            raise UploadAuthError("Upload initiation failed — invalid access token") from exc
        if status == 403:
            raise UploadPermissionError(
                f"YouTube API denied upload permission (HTTP 403): {body_text}"
            ) from exc
        raise UploadError(
            f"Failed to initiate resumable upload (HTTP {status}): {body_text}"
        ) from exc
    except requests.RequestException as exc:
        raise UploadError(f"Upload initiation network error: {exc}") from exc

    upload_uri = resp.headers.get("Location")
    if not upload_uri:
        raise UploadError("Upload initiation returned no Location header")

    logger.info("Resumable upload session initiated")
    return upload_uri


def _upload_video_data(
    *,
    video_path: Path,
    upload_uri: str,
    access_token: str,
    config: UploadConfig,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Upload the video file data via resumable upload.

    Parameters
    ----------
    video_path:
        Path to the MP4 file.
    upload_uri:
        Resumable upload URI from session initiation.
    access_token:
        Valid OAuth2 access token.
    config:
        Upload configuration.
    session:
        Optional requests session.

    Returns
    -------
    dict
        The YouTube API response body containing the video ID.

    Raises
    ------
    UploadError
        If upload fails after retries.
    """
    sess = session or requests.Session()
    file_size = video_path.stat().st_size
    chunk_size = config.chunk_size

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "video/mp4",
    }

    last_exc: Exception | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            with open(video_path, "rb") as f:
                offset = 0
                while offset < file_size:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    end = offset + len(chunk) - 1
                    headers["Content-Range"] = f"bytes {offset}-{end}/{file_size}"

                    resp = sess.put(upload_uri, data=chunk, headers=headers, timeout=300)

                    if resp.status_code in (200, 201):
                        logger.info("Video upload complete (HTTP %d)", resp.status_code)
                        return resp.json()

                    if resp.status_code == 308:
                        offset += len(chunk)
                        continue

                    if resp.status_code in _RETRYABLE_UPLOAD_CODES:
                        raise UploadError(f"Transient upload error (HTTP {resp.status_code})")

                    raise UploadError(
                        f"Upload failed (HTTP {resp.status_code}): {resp.text[:500]}"
                    )

            raise UploadError("Upload stream ended without completion response")

        except UploadError as exc:
            last_exc = exc
            status_code = _extract_status(exc)
            if status_code in _NON_RETRYABLE_CODES:
                break

            if attempt < config.max_retries:
                delay = _RETRY_DELAY_BASE ** attempt
                logger.warning(
                    "Upload attempt %d/%d failed, retrying in %.1fs: %s",
                    attempt, config.max_retries, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.warning("Upload exhausted %d retries", config.max_retries)

    raise UploadError(f"Video upload failed after {config.max_retries} attempts: {last_exc}") from last_exc


def _extract_status(exc: UploadError) -> int:
    """Extract HTTP status code from an error message, if present."""
    msg = str(exc)
    if "HTTP" in msg:
        try:
            start = msg.index("HTTP ") + 5
            end = msg.index(")", start)
            return int(msg[start:end])
        except (ValueError, IndexError):
            pass
    return 0


# ---------------------------------------------------------------------------
# Thumbnail upload
# ---------------------------------------------------------------------------

def _upload_thumbnail(
    *,
    video_id: str,
    thumbnail_path: Path,
    access_token: str,
    session: requests.Session | None = None,
) -> None:
    """Upload a thumbnail for an already-uploaded video.

    Parameters
    ----------
    video_id:
        YouTube video ID.
    thumbnail_path:
        Path to the thumbnail JPEG.
    access_token:
        Valid OAuth2 access token.
    session:
        Optional requests session.

    Raises
    ------
    UploadError
        If thumbnail upload fails.
    """
    sess = session or requests.Session()

    data = thumbnail_path.read_bytes()
    content_type = "image/jpeg"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": content_type,
        "Content-Length": str(len(data)),
    }

    params = {"videoId": video_id}

    try:
        resp = sess.post(
            _YT_THUMBNAIL_URL,
            data=data,
            headers=headers,
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        logger.info("Thumbnail uploaded for video %s", video_id)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        raise UploadError(f"Thumbnail upload failed (HTTP {status})") from exc
    except requests.RequestException as exc:
        raise UploadError(f"Thumbnail upload network error: {exc}") from exc


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_upload_inputs(
    video_path: Path,
    thumbnail_path: Path,
    video_metadata: VideoMetadata,
) -> None:
    """Validate inputs before attempting upload.

    Raises
    ------
    UploadError
        If any input is invalid.
    """
    if not isinstance(video_path, Path):
        raise UploadError("video_path must be a Path")

    if not video_path.is_file():
        raise UploadError(f"Video file does not exist: {video_path}")

    if video_path.stat().st_size == 0:
        raise UploadError(f"Video file is empty: {video_path}")

    if not isinstance(thumbnail_path, Path):
        raise UploadError("thumbnail_path must be a Path")

    if not thumbnail_path.is_file():
        raise UploadError(f"Thumbnail file does not exist: {thumbnail_path}")

    if thumbnail_path.stat().st_size == 0:
        raise UploadError(f"Thumbnail file is empty: {thumbnail_path}")

    if not isinstance(video_metadata, VideoMetadata):
        raise UploadError("video_metadata must be a VideoMetadata instance")

    if not video_metadata.title.strip():
        raise UploadError("Video metadata title is empty")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_video(
    *,
    video_path: Path,
    thumbnail_path: Path,
    video_metadata: VideoMetadata,
    qa_result: QAResult,
    config: UploadConfig | None = None,
) -> UploadResult:
    """Upload a validated video job to YouTube.

    This is the main entry point for the YouTube upload pipeline.

    Parameters
    ----------
    video_path:
        Path to the final MP4 video file.
    thumbnail_path:
        Path to the generated thumbnail JPEG.
    video_metadata:
        Validated title, description, and tags.
    qa_result:
        QA gate result — upload is refused if not passed.
    config:
        Upload configuration.  Loaded from environment if None.

    Returns
    -------
    UploadResult
        Structured result with video ID, URL, and status.

    Raises
    ------
    UploadConfigError
        If credential configuration is missing.
    UploadAuthError
        If OAuth authentication fails.
    UploadPermissionError
        If YouTube denies upload permission.
    UploadError
        If upload fails for other reasons.
    """
    # --- QA gate ---
    if not qa_result.passed:
        raise UploadError(
            f"QA gate failed with {len(qa_result.errors)} error(s): "
            f"{'; '.join(qa_result.errors[:3])}"
        )

    # --- Validate inputs ---
    _validate_upload_inputs(video_path, thumbnail_path, video_metadata)

    # --- Load config ---
    cfg = config or load_upload_config()

    # --- Refresh access token ---
    access_token = _refresh_access_token(cfg)

    # --- Upload video ---
    video_size = video_path.stat().st_size
    upload_uri = _init_resumable_upload(
        video_metadata=video_metadata,
        file_size=video_size,
        access_token=access_token,
        config=cfg,
    )
    video_response = _upload_video_data(
        video_path=video_path,
        upload_uri=upload_uri,
        access_token=access_token,
        config=cfg,
    )

    video_id = video_response.get("id", "")
    if not video_id:
        raise UploadError("YouTube API returned empty video ID")

    video_url = f"https://www.youtube.com/shorts/{video_id}"
    logger.info("Video uploaded: %s", video_url)

    # --- Upload thumbnail (partial-success on failure) ---
    thumbnail_status = "pending"
    try:
        _upload_thumbnail(
            video_id=video_id,
            thumbnail_path=thumbnail_path,
            access_token=access_token,
        )
        thumbnail_status = "uploaded"
    except UploadError as exc:
        thumbnail_status = "failed"
        logger.warning("Thumbnail upload failed (video still uploaded): %s", exc)

    return UploadResult(
        video_id=video_id,
        video_url=video_url,
        video_status="uploaded",
        thumbnail_status=thumbnail_status,
        details={
            "video_file_size": video_size,
            "title": video_metadata.title,
        },
    )
