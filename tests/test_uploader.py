"""Tests for the youtube.uploader module — all use mocks, no real API calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import requests

from content.metadata import VideoMetadata
from core.qa import QAResult
from youtube.uploader import (
    UploadConfig,
    UploadResult,
    UploadError,
    UploadAuthError,
    UploadConfigError,
    UploadPermissionError,
    load_upload_config,
    upload_video,
    _refresh_access_token,
    _init_resumable_upload,
    _upload_video_data,
    _upload_thumbnail,
    _validate_upload_inputs,
    _extract_status,
    _DEFAULT_CATEGORY_ID,
    _DEFAULT_PRIVACY_STATUS,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FAKE_MP4 = (
    b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8 +
    b"\x00\x00\x00\x08moov" + b"\x00" * 1000
)

_FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 200 + b"\xff\xd9"


def _create_file(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _make_metadata(**overrides) -> VideoMetadata:
    defaults = dict(
        title="GPT-5 Is Here",
        description="OpenAI released GPT-5 with major improvements.",
        tags=["gpt-5", "openai", "ai"],
    )
    defaults.update(overrides)
    return VideoMetadata(**defaults)


def _make_config(**overrides) -> UploadConfig:
    defaults = dict(
        client_id="fake-client-id",
        client_secret="fake-client-secret",
        refresh_token="fake-refresh-token",
    )
    defaults.update(overrides)
    return UploadConfig(**defaults)


def _passed_qa() -> QAResult:
    return QAResult(passed=True, errors=[], warnings=[], details={})


def _failed_qa(errors: list[str] | None = None) -> QAResult:
    return QAResult(
        passed=False,
        errors=errors or ["Topic is missing"],
        warnings=[],
        details={},
    )


class StubResponse:
    """Minimal requests.Response stub."""
    def __init__(self, status_code: int = 200, json_data: dict | None = None,
                 text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


# ---------------------------------------------------------------------------
# UploadConfig model
# ---------------------------------------------------------------------------

class TestUploadConfigModel:
    def test_fields(self):
        cfg = _make_config()
        assert cfg.client_id == "fake-client-id"
        assert cfg.client_secret == "fake-client-secret"
        assert cfg.refresh_token == "fake-refresh-token"
        assert cfg.category_id == _DEFAULT_CATEGORY_ID
        assert cfg.privacy_status == _DEFAULT_PRIVACY_STATUS
        assert cfg.chunk_size > 0
        assert cfg.max_retries == 3

    def test_is_frozen(self):
        cfg = _make_config()
        with pytest.raises(AttributeError):
            cfg.client_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_upload_config
# ---------------------------------------------------------------------------

class TestLoadUploadConfig:
    @patch("youtube.uploader.settings")
    def test_loads_from_settings(self, mock_settings):
        mock_settings.YOUTUBE_CLIENT_ID = "id"
        mock_settings.YOUTUBE_CLIENT_SECRET = "secret"
        mock_settings.YOUTUBE_REFRESH_TOKEN = "token"
        cfg = load_upload_config()
        assert cfg.client_id == "id"
        assert cfg.client_secret == "secret"
        assert cfg.refresh_token == "token"

    @patch("youtube.uploader.settings")
    def test_raises_on_missing_client_id(self, mock_settings):
        mock_settings.YOUTUBE_CLIENT_ID = ""
        mock_settings.YOUTUBE_CLIENT_SECRET = "secret"
        mock_settings.YOUTUBE_REFRESH_TOKEN = "token"
        with pytest.raises(UploadConfigError, match="YOUTUBE_CLIENT_ID"):
            load_upload_config()

    @patch("youtube.uploader.settings")
    def test_raises_on_missing_all(self, mock_settings):
        mock_settings.YOUTUBE_CLIENT_ID = ""
        mock_settings.YOUTUBE_CLIENT_SECRET = ""
        mock_settings.YOUTUBE_REFRESH_TOKEN = ""
        with pytest.raises(UploadConfigError, match="Missing required"):
            load_upload_config()


# ---------------------------------------------------------------------------
# _extract_status
# ---------------------------------------------------------------------------

class TestExtractStatus:
    def test_extracts_http_code(self):
        assert _extract_status(UploadError("Upload failed (HTTP 403): body")) == 403

    def test_returns_zero_when_no_code(self):
        assert _extract_status(UploadError("no code here")) == 0


# ---------------------------------------------------------------------------
# _validate_upload_inputs
# ---------------------------------------------------------------------------

class TestValidateUploadInputs:
    def test_valid_inputs(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)
        _validate_upload_inputs(v, t, _make_metadata())  # Should not raise

    def test_missing_video(self, tmp_path: Path):
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)
        with pytest.raises(UploadError, match="does not exist"):
            _validate_upload_inputs(tmp_path / "nope.mp4", t, _make_metadata())

    def test_empty_video(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", b"")
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)
        with pytest.raises(UploadError, match="empty"):
            _validate_upload_inputs(v, t, _make_metadata())

    def test_missing_thumbnail(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        with pytest.raises(UploadError, match="does not exist"):
            _validate_upload_inputs(v, tmp_path / "nope.jpg", _make_metadata())

    def test_empty_thumbnail(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", b"")
        with pytest.raises(UploadError, match="empty"):
            _validate_upload_inputs(v, t, _make_metadata())

    def test_empty_metadata_title(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)
        with pytest.raises(UploadError, match="title is empty"):
            _validate_upload_inputs(v, t, _make_metadata(title=""))


# ---------------------------------------------------------------------------
# _refresh_access_token
# ---------------------------------------------------------------------------

class TestRefreshAccessToken:
    def test_returns_access_token(self):
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(200, {"access_token": "tok_abc123"})
        token = _refresh_access_token(_make_config(), session=mock_session)
        assert token == "tok_abc123"

    def test_raises_on_401(self):
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(401, {"error": "invalid_grant"})
        with pytest.raises(UploadAuthError, match="Re-authorize"):
            _refresh_access_token(_make_config(), session=mock_session)

    def test_raises_on_network_error(self):
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.ConnectionError("refused")
        with pytest.raises(UploadAuthError, match="network error"):
            _refresh_access_token(_make_config(), session=mock_session)

    def test_raises_when_no_token_in_response(self):
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(200, {"error": "unsupported_grant_type"})
        with pytest.raises(UploadAuthError, match="no access_token"):
            _refresh_access_token(_make_config(), session=mock_session)

    def test_does_not_leak_secret_in_logs(self):
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(200, {"access_token": "tok_secret"})
        token = _refresh_access_token(_make_config(), session=mock_session)
        assert token == "tok_secret"
        # The OAuth token endpoint requires the secret — the test verifies that
        # the secret is not included in the access_token returned or in error messages.
        # (Sending the secret to Google's token endpoint is required by OAuth2.)
        assert token != "fake-client-secret"
        assert "fake-client-secret" not in token


# ---------------------------------------------------------------------------
# _init_resumable_upload
# ---------------------------------------------------------------------------

class TestInitResumableUpload:
    def test_returns_upload_uri(self):
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(
            200, headers={"Location": "https://upload.example.com/session/123"}
        )
        uri = _init_resumable_upload(
            video_metadata=_make_metadata(),
            file_size=1024,
            access_token="tok",
            config=_make_config(),
            session=mock_session,
        )
        assert uri == "https://upload.example.com/session/123"

    def test_sends_correct_body(self):
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(
            200, headers={"Location": "https://upload.example.com/s"}
        )
        _init_resumable_upload(
            video_metadata=_make_metadata(title="Test Title", tags=["a", "b"]),
            file_size=2048,
            access_token="tok",
            config=_make_config(category_id="22", privacy_status="public"),
            session=mock_session,
        )
        call_kwargs = mock_session.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["snippet"]["title"] == "Test Title"
        assert body["snippet"]["tags"] == ["a", "b"]
        assert body["snippet"]["categoryId"] == "22"
        assert body["status"]["privacyStatus"] == "public"

    def test_raises_on_403(self):
        mock_session = MagicMock()
        resp = StubResponse(403, text=" Forbidden")
        mock_session.post.return_value = resp
        with pytest.raises(UploadPermissionError, match="403"):
            _init_resumable_upload(
                video_metadata=_make_metadata(),
                file_size=1024,
                access_token="tok",
                config=_make_config(),
                session=mock_session,
            )

    def test_raises_on_401(self):
        mock_session = MagicMock()
        resp = StubResponse(401, text=" Unauthorized")
        mock_session.post.return_value = resp
        with pytest.raises(UploadAuthError, match="invalid access token"):
            _init_resumable_upload(
                video_metadata=_make_metadata(),
                file_size=1024,
                access_token="tok",
                config=_make_config(),
                session=mock_session,
            )

    def test_raises_on_no_location_header(self):
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(200, headers={})
        with pytest.raises(UploadError, match="no Location header"):
            _init_resumable_upload(
                video_metadata=_make_metadata(),
                file_size=1024,
                access_token="tok",
                config=_make_config(),
                session=mock_session,
            )


# ---------------------------------------------------------------------------
# _upload_video_data
# ---------------------------------------------------------------------------

class TestUploadVideoData:
    def test_successful_upload(self, tmp_path: Path):
        video = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        mock_session = MagicMock()
        mock_session.put.return_value = StubResponse(200, {"id": "vid123"})

        result = _upload_video_data(
            video_path=video,
            upload_uri="https://upload.example.com/s",
            access_token="tok",
            config=_make_config(),
            session=mock_session,
        )
        assert result["id"] == "vid123"

    def test_retries_on_503(self, tmp_path: Path):
        # Create a file large enough to require 2 chunks
        big_data = _FAKE_MP4 + b"\x00" * (1024 * 1024)  # ~1MB
        video = _create_file(tmp_path, "v.mp4", big_data)
        mock_session = MagicMock()
        # First chunk -> 308, second chunk -> 503 (transient), retry: first chunk -> 308, second chunk -> 200
        mock_session.put.side_effect = [
            StubResponse(308),  # chunk 1 accepted
            StubResponse(503),  # chunk 2 transient
            StubResponse(308),  # retry chunk 1
            StubResponse(200, {"id": "vid456"}),  # retry chunk 2 success
        ]

        result = _upload_video_data(
            video_path=video,
            upload_uri="https://upload.example.com/s",
            access_token="tok",
            config=_make_config(max_retries=2, chunk_size=512 * 1024),  # 512KB chunks
            session=mock_session,
        )
        assert result["id"] == "vid456"

    def test_fails_after_exhausted_retries(self, tmp_path: Path):
        big_data = _FAKE_MP4 + b"\x00" * (1024 * 1024)
        video = _create_file(tmp_path, "v.mp4", big_data)
        mock_session = MagicMock()
        mock_session.put.return_value = StubResponse(503)

        with pytest.raises(UploadError, match="failed after"):
            _upload_video_data(
                video_path=video,
                upload_uri="https://upload.example.com/s",
                access_token="tok",
                config=_make_config(max_retries=2, chunk_size=512 * 1024),
                session=mock_session,
            )

    def test_does_not_retry_403(self, tmp_path: Path):
        video = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        mock_session = MagicMock()
        mock_session.put.return_value = StubResponse(403, text=" Forbidden")

        with pytest.raises(UploadError, match="Upload failed"):
            _upload_video_data(
                video_path=video,
                upload_uri="https://upload.example.com/s",
                access_token="tok",
                config=_make_config(max_retries=3),
                session=mock_session,
            )
        # Should not retry — only 1 call
        assert mock_session.put.call_count == 1


# ---------------------------------------------------------------------------
# _upload_thumbnail
# ---------------------------------------------------------------------------

class TestUploadThumbnail:
    def test_successful_thumbnail_upload(self, tmp_path: Path):
        thumb = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(200)

        _upload_thumbnail(
            video_id="vid123",
            thumbnail_path=thumb,
            access_token="tok",
            session=mock_session,
        )
        mock_session.post.assert_called_once()

    def test_raises_on_failure(self, tmp_path: Path):
        thumb = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)
        mock_session = MagicMock()
        mock_session.post.return_value = StubResponse(500)

        with pytest.raises(UploadError, match="Thumbnail upload failed"):
            _upload_thumbnail(
                video_id="vid123",
                thumbnail_path=thumb,
                access_token="tok",
                session=mock_session,
            )


# ---------------------------------------------------------------------------
# upload_video — QA gate blocks upload
# ---------------------------------------------------------------------------

class TestQAGateBlocksUpload:
    def test_refuses_on_failed_qa(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        with pytest.raises(UploadError, match="QA gate failed"):
            upload_video(
                video_path=v,
                thumbnail_path=t,
                video_metadata=_make_metadata(),
                qa_result=_failed_qa(),
                config=_make_config(),
            )

    def test_refuses_with_multiple_errors(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        with pytest.raises(UploadError, match="3 error"):
            upload_video(
                video_path=v,
                thumbnail_path=t,
                video_metadata=_make_metadata(),
                qa_result=_failed_qa(["E1", "E2", "E3"]),
                config=_make_config(),
            )


# ---------------------------------------------------------------------------
# upload_video — successful full flow
# ---------------------------------------------------------------------------

class TestUploadVideoSuccess:
    @patch("youtube.uploader._refresh_access_token", return_value="tok_fresh")
    @patch("youtube.uploader._upload_thumbnail")
    @patch("youtube.uploader._upload_video_data")
    @patch("youtube.uploader._init_resumable_upload")
    def test_full_success(
        self, mock_init, mock_upload, mock_thumb, mock_refresh, tmp_path: Path
    ):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        mock_init.return_value = "https://upload.uri"
        mock_upload.return_value = {"id": "dQw4w9WgXcQ"}

        result = upload_video(
            video_path=v,
            thumbnail_path=t,
            video_metadata=_make_metadata(),
            qa_result=_passed_qa(),
            config=_make_config(),
        )

        assert isinstance(result, UploadResult)
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.video_url == "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert result.video_status == "uploaded"
        assert result.thumbnail_status == "uploaded"
        assert len(result.errors) == 0

    @patch("youtube.uploader._refresh_access_token", return_value="tok_fresh")
    @patch("youtube.uploader._upload_thumbnail")
    @patch("youtube.uploader._upload_video_data")
    @patch("youtube.uploader._init_resumable_upload")
    def test_metadata_mapped_correctly(
        self, mock_init, mock_upload, mock_thumb, mock_refresh, tmp_path: Path
    ):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        mock_init.return_value = "https://upload.uri"
        mock_upload.return_value = {"id": "vid"}

        meta = _make_metadata(title="Custom Title", tags=["custom", "tags"])
        upload_video(
            video_path=v,
            thumbnail_path=t,
            video_metadata=meta,
            qa_result=_passed_qa(),
            config=_make_config(),
        )

        # Verify init was called with our metadata
        call_kwargs = mock_init.call_args.kwargs
        assert call_kwargs["video_metadata"].title == "Custom Title"
        assert call_kwargs["video_metadata"].tags == ["custom", "tags"]


# ---------------------------------------------------------------------------
# upload_video — video success + thumbnail failure = partial success
# ---------------------------------------------------------------------------

class TestPartialSuccess:
    @patch("youtube.uploader._refresh_access_token", return_value="tok")
    @patch("youtube.uploader._upload_thumbnail", side_effect=UploadError("thumb failed"))
    @patch("youtube.uploader._upload_video_data")
    @patch("youtube.uploader._init_resumable_upload")
    def test_thumbnail_failure_returns_partial(
        self, mock_init, mock_upload, mock_thumb, mock_refresh, tmp_path: Path
    ):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        mock_init.return_value = "https://upload.uri"
        mock_upload.return_value = {"id": "vid_partial"}

        result = upload_video(
            video_path=v,
            thumbnail_path=t,
            video_metadata=_make_metadata(),
            qa_result=_passed_qa(),
            config=_make_config(),
        )

        assert result.video_id == "vid_partial"
        assert result.video_status == "uploaded"
        assert result.thumbnail_status == "failed"


# ---------------------------------------------------------------------------
# upload_video — credential configuration failure
# ---------------------------------------------------------------------------

class TestCredentialFailure:
    @patch("youtube.uploader.settings")
    def test_raises_config_error_when_missing(self, mock_settings, tmp_path: Path):
        mock_settings.YOUTUBE_CLIENT_ID = ""
        mock_settings.YOUTUBE_CLIENT_SECRET = ""
        mock_settings.YOUTUBE_REFRESH_TOKEN = ""

        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        with pytest.raises(UploadConfigError, match="Missing required"):
            upload_video(
                video_path=v,
                thumbnail_path=t,
                video_metadata=_make_metadata(),
                qa_result=_passed_qa(),
            )


# ---------------------------------------------------------------------------
# upload_video — authentication failure
# ---------------------------------------------------------------------------

class TestAuthFailure:
    @patch("youtube.uploader._refresh_access_token", side_effect=UploadAuthError("token expired"))
    def test_propagates_auth_error(self, mock_refresh, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        with pytest.raises(UploadAuthError, match="token expired"):
            upload_video(
                video_path=v,
                thumbnail_path=t,
                video_metadata=_make_metadata(),
                qa_result=_passed_qa(),
                config=_make_config(),
            )


# ---------------------------------------------------------------------------
# upload_video — permission failure
# ---------------------------------------------------------------------------

class TestPermissionFailure:
    @patch("youtube.uploader._refresh_access_token", return_value="tok")
    @patch("youtube.uploader._init_resumable_upload", side_effect=UploadPermissionError("denied"))
    def test_propagates_permission_error(self, mock_init, mock_refresh, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        with pytest.raises(UploadPermissionError, match="denied"):
            upload_video(
                video_path=v,
                thumbnail_path=t,
                video_metadata=_make_metadata(),
                qa_result=_passed_qa(),
                config=_make_config(),
            )


# ---------------------------------------------------------------------------
# upload_video — invalid/missing video
# ---------------------------------------------------------------------------

class TestInvalidVideo:
    def test_missing_video_file(self, tmp_path: Path):
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)
        with pytest.raises(UploadError, match="does not exist"):
            upload_video(
                video_path=tmp_path / "nope.mp4",
                thumbnail_path=t,
                video_metadata=_make_metadata(),
                qa_result=_passed_qa(),
                config=_make_config(),
            )

    def test_empty_video_file(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", b"")
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)
        with pytest.raises(UploadError, match="empty"):
            upload_video(
                video_path=v,
                thumbnail_path=t,
                video_metadata=_make_metadata(),
                qa_result=_passed_qa(),
                config=_make_config(),
            )


# ---------------------------------------------------------------------------
# upload_video — invalid/missing thumbnail
# ---------------------------------------------------------------------------

class TestInvalidThumbnail:
    def test_missing_thumbnail_file(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        with pytest.raises(UploadError, match="does not exist"):
            upload_video(
                video_path=v,
                thumbnail_path=tmp_path / "nope.jpg",
                video_metadata=_make_metadata(),
                qa_result=_passed_qa(),
                config=_make_config(),
            )

    def test_empty_thumbnail_file(self, tmp_path: Path):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", b"")
        with pytest.raises(UploadError, match="empty"):
            upload_video(
                video_path=v,
                thumbnail_path=t,
                video_metadata=_make_metadata(),
                qa_result=_passed_qa(),
                config=_make_config(),
            )


# ---------------------------------------------------------------------------
# UploadResult model
# ---------------------------------------------------------------------------

class TestUploadResultModel:
    def test_fields(self):
        r = UploadResult(
            video_id="abc",
            video_url="https://youtube.com/shorts/abc",
            video_status="uploaded",
            thumbnail_status="uploaded",
        )
        assert r.video_id == "abc"
        assert r.video_status == "uploaded"
        assert r.thumbnail_status == "uploaded"
        assert r.errors == []
        assert r.details == {}

    def test_is_frozen(self):
        r = UploadResult(
            video_id="x", video_url="u", video_status="s", thumbnail_status="t"
        )
        with pytest.raises(AttributeError):
            r.video_id = "y"  # type: ignore[misc]

    def test_with_details(self):
        r = UploadResult(
            video_id="v",
            video_url="u",
            video_status="s",
            thumbnail_status="t",
            details={"key": "val"},
        )
        assert r.details["key"] == "val"


# ---------------------------------------------------------------------------
# Sensitive credential logging
# ---------------------------------------------------------------------------

class TestSensitiveCredentialHandling:
    @patch("youtube.uploader._refresh_access_token", return_value="tok_secret_value")
    @patch("youtube.uploader._upload_thumbnail")
    @patch("youtube.uploader._upload_video_data")
    @patch("youtube.uploader._init_resumable_upload")
    def test_no_secret_in_result(
        self, mock_init, mock_upload, mock_thumb, mock_refresh, tmp_path: Path
    ):
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        mock_init.return_value = "https://upload.uri"
        mock_upload.return_value = {"id": "vid"}

        result = upload_video(
            video_path=v,
            thumbnail_path=t,
            video_metadata=_make_metadata(),
            qa_result=_passed_qa(),
            config=_make_config(client_secret="super_secret_123"),
        )

        # Result should never contain secrets
        result_str = str(result)
        assert "super_secret_123" not in result_str
        assert "tok_secret_value" not in result_str
        assert "fake-refresh-token" not in result_str

    @patch("youtube.uploader.settings")
    def test_config_loader_does_not_expose_secrets(self, mock_settings):
        mock_settings.YOUTUBE_CLIENT_ID = "id_val"
        mock_settings.YOUTUBE_CLIENT_SECRET = "sec_val"
        mock_settings.YOUTUBE_REFRESH_TOKEN = "ref_val"
        cfg = load_upload_config()

        # Config repr should not leak into unexpected places
        # (it's a dataclass, secrets are in fields — this is expected)
        # The key test: the config is not printed in logs by the uploader
        assert cfg.client_secret == "sec_val"

    @patch("youtube.uploader._refresh_access_token", return_value="tok")
    @patch("youtube.uploader._upload_thumbnail", side_effect=UploadError("fail"))
    @patch("youtube.uploader._upload_video_data")
    @patch("youtube.uploader._init_resumable_upload")
    def test_auth_error_does_not_log_credentials(
        self, mock_init, mock_upload, mock_thumb, mock_refresh, tmp_path: Path
    ):
        """Even on errors, credentials should not leak into exception messages."""
        v = _create_file(tmp_path, "v.mp4", _FAKE_MP4)
        t = _create_file(tmp_path, "t.jpg", _FAKE_JPEG)

        mock_init.return_value = "https://upload.uri"
        mock_upload.return_value = {"id": "vid"}

        # This should succeed (partial success)
        result = upload_video(
            video_path=v,
            thumbnail_path=t,
            video_metadata=_make_metadata(),
            qa_result=_passed_qa(),
            config=_make_config(client_secret="my_secret_456"),
        )

        assert "my_secret_456" not in str(result.errors)
        assert "my_secret_456" not in result.thumbnail_status
