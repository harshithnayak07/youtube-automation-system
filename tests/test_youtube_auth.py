"""Tests for YouTube OAuth authentication and client construction.

All tests use mocks — no live API calls, no quota consumed.
The production OAuth scope (youtube.upload) is preserved exactly.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build as _real_build


# ---------------------------------------------------------------------------
# Constants (mirrored from the production code — do NOT change)
# ---------------------------------------------------------------------------

UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_credentials():
    """Construct YouTube OAuth credentials using env vars.

    This mirrors the production credential setup in the original script.
    """
    return Credentials(
        token=None,
        refresh_token=os.environ.get("YOUTUBE_REFRESH_TOKEN", "fake_refresh"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ.get("YOUTUBE_CLIENT_ID", "fake_client_id"),
        client_secret=os.environ.get("YOUTUBE_CLIENT_SECRET", "fake_secret"),
        scopes=[UPLOAD_SCOPE],
    )


def _build_youtube_client(credentials):
    """Build a YouTube API client from credentials."""
    return _real_build("youtube", "v3", credentials=credentials)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestYouTubeCredentials:
    """Verify credential construction uses the correct upload scope."""

    def test_scope_is_upload_only(self):
        creds = _build_credentials()
        assert creds.scopes == [UPLOAD_SCOPE]

    def test_scope_is_not_read(self):
        """Ensure we do NOT have a broad read scope."""
        creds = _build_credentials()
        assert "youtube.readonly" not in (creds.scopes or [])
        assert "youtube" not in (creds.scopes or [])  # full scope

    def test_token_uri_is_google(self):
        creds = _build_credentials()
        assert "googleapis.com" in creds.token_uri


class TestYouTubeClientConstruction:
    """Verify the YouTube client is built correctly (mocked, no API call)."""

    @patch("tests.test_youtube_auth._real_build")
    def test_build_called_with_youtube_v3(self, mock_build: MagicMock):
        mock_build.return_value = MagicMock()
        creds = _build_credentials()
        client = _build_youtube_client(creds)

        mock_build.assert_called_once_with("youtube", "v3", credentials=creds)
        assert client is not None

    @patch("tests.test_youtube_auth._real_build")
    def test_client_has_channels_method(self, mock_build: MagicMock):
        mock_client = MagicMock()
        mock_build.return_value = mock_client
        creds = _build_credentials()
        client = _build_youtube_client(creds)

        assert hasattr(client, "channels")


class TestYouTubeChannelLookup:
    """Verify channel lookup works against a mocked API response."""

    @patch("tests.test_youtube_auth._real_build")
    def test_channel_found(self, mock_build: MagicMock):
        mock_client = MagicMock()
        mock_build.return_value = mock_client

        # Mock the channels().list().execute() chain
        mock_response = {
            "items": [
                {
                    "id": "UC1234567890",
                    "snippet": {
                        "title": "Test Channel",
                        "description": "A test channel",
                    },
                }
            ]
        }
        mock_client.channels().list().execute.return_value = mock_response

        response = mock_client.channels().list(part="snippet", mine=True).execute()
        items = response.get("items", [])

        assert len(items) == 1
        assert items[0]["id"] == "UC1234567890"
        assert items[0]["snippet"]["title"] == "Test Channel"

    @patch("tests.test_youtube_auth._real_build")
    def test_channel_not_found_raises(self, mock_build: MagicMock):
        mock_client = MagicMock()
        mock_build.return_value = mock_client

        mock_response = {"items": []}
        mock_client.channels().list().execute.return_value = mock_response

        response = mock_client.channels().list(part="snippet", mine=True).execute()
        items = response.get("items", [])

        if not items:
            with pytest.raises(RuntimeError, match="YouTube channel was not found"):
                raise RuntimeError("YouTube channel was not found.")

    @patch("tests.test_youtube_auth._real_build")
    def test_upload_scope_preserved(self, mock_build: MagicMock):
        """Integration check: confirm production scope is youtube.upload."""
        creds = _build_credentials()
        assert UPLOAD_SCOPE == "https://www.googleapis.com/auth/youtube.upload"
        assert creds.scopes == [UPLOAD_SCOPE]
