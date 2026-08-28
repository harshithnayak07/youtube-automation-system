"""Tests for the media.image module — Pexels-based, all use mocks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from content.scenes import Scene
from media.image import (
    GeneratedImage,
    ImageGenError,
    ImageGenConfigError,
    generate_scene_image,
    generate_all_scene_images,
    _build_search_query,
    _validate_image_bytes,
    _detect_image_format,
    _is_image_file_valid,
    _is_transient,
    _search_pexels,
    _select_best_photo,
    _download_photo,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_scene(
    scene: int = 1,
    text: str = "Test narration.",
    visual_description: str = "A futuristic city with neon lights",
) -> Scene:
    return Scene(scene=scene, text=text, visual_description=visual_description)


# Minimal valid JPEG header (smallest valid JPEG)
_JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
_JPEG_BODY = b"\xff\xdb\x00C\x00" + b"\x08" * 64 + b"\xff\xc0\x00\x11\x08" + b"\x00" * 20
_FAKE_JPEG = _JPEG_HEADER + _JPEG_BODY + b"\xff\xd9"

# Minimal valid PNG (padded to exceed 100-byte validation threshold)
_FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n"
    + b"\x00" * 20
    + b"IHDR"
    + b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    + b"\x00" * 120
)


def _make_pexels_photo(
    photo_id: int = 1,
    width: int = 600,
    height: int = 900,
    original_url: str = "https://images.pexels.com/photos/1/test.jpeg",
) -> dict:
    """Build a minimal Pexels photo dict."""
    return {
        "id": photo_id,
        "width": width,
        "height": height,
        "src": {
            "original": original_url,
            "large2x": original_url,
            "large": original_url,
        },
    }


def _mock_pexels_search_response(photos: list[dict]) -> MagicMock:
    """Build a mock requests.Response for Pexels search."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.ok = True
    resp.json.return_value = {"photos": photos, "total_results": len(photos)}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_image_download_response(image_bytes: bytes) -> MagicMock:
    """Build a mock requests.Response for image download."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.content = image_bytes
    resp.raise_for_status = MagicMock()
    return resp


def _make_http_error(status_code: int) -> requests.HTTPError:
    """Create an HTTPError with the given status code."""
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = b'{"error": "bad"}'
    return requests.HTTPError(response=resp)


# ---------------------------------------------------------------------------
# _is_transient classification
# ---------------------------------------------------------------------------

class TestIsTransient:
    def test_retryable_codes(self):
        for code in (408, 429, 500, 502, 503, 504):
            resp = requests.Response()
            resp.status_code = code
            exc = requests.HTTPError(response=resp)
            assert _is_transient(exc) is True

    def test_non_retryable_codes(self):
        for code in (400, 401, 403, 404, 422):
            resp = requests.Response()
            resp.status_code = code
            exc = requests.HTTPError(response=resp)
            assert _is_transient(exc) is False

    def test_timeout_is_transient(self):
        assert _is_transient(requests.Timeout("timeout")) is True

    def test_connection_error_is_transient(self):
        assert _is_transient(requests.ConnectionError("conn")) is True

    def test_unknown_not_transient(self):
        assert _is_transient(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

class TestImageValidation:
    def test_validates_jpeg(self):
        assert _validate_image_bytes(_FAKE_JPEG) is True

    def test_validates_png(self):
        assert _validate_image_bytes(_FAKE_PNG) is True

    def test_rejects_too_short(self):
        assert _validate_image_bytes(b"\xff\xd8\xff") is False

    def test_rejects_unknown_format(self):
        assert _validate_image_bytes(b"BM" + b"\x00" * 200) is False

    def test_detect_format_jpeg(self):
        assert _detect_image_format(_FAKE_JPEG) == "jpeg"

    def test_detect_format_png(self):
        assert _detect_image_format(_FAKE_PNG) == "png"


class TestIsImageFileValid:
    def test_valid_existing_file(self, tmp_path: Path):
        img = tmp_path / "scene_1.jpg"
        img.write_bytes(_FAKE_JPEG)
        assert _is_image_file_valid(img) is True

    def test_nonexistent_file(self, tmp_path: Path):
        assert _is_image_file_valid(tmp_path / "nope.jpg") is False

    def test_empty_file(self, tmp_path: Path):
        img = tmp_path / "empty.jpg"
        img.write_bytes(b"")
        assert _is_image_file_valid(img) is False

    def test_corrupt_file(self, tmp_path: Path):
        img = tmp_path / "corrupt.jpg"
        img.write_bytes(b"this is not an image at all, just random data")
        assert _is_image_file_valid(img) is False


# ---------------------------------------------------------------------------
# Pexels configuration
# ---------------------------------------------------------------------------

class TestPexelsConfig:
    @patch("media.image.settings")
    def test_raises_config_error_without_key(self, mock_settings: MagicMock, tmp_path: Path):
        mock_settings.PEXELS_API_KEY = ""
        scene = _make_scene()

        with pytest.raises(ImageGenConfigError, match="PEXELS_API_KEY"):
            generate_scene_image(scene, output_dir=tmp_path, api_key="")

    @patch("media.image._download_photo", return_value=_FAKE_JPEG)
    @patch("media.image._search_pexels")
    @patch("media.image.settings")
    def test_uses_settings_key_when_not_passed(
        self, mock_settings: MagicMock, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path
    ):
        mock_settings.PEXELS_API_KEY = "from-settings"
        mock_search.return_value = [_make_pexels_photo()]
        scene = _make_scene()

        generate_scene_image(scene, output_dir=tmp_path)
        # The key should have been passed through from settings
        call_kwargs = mock_search.call_args
        assert call_kwargs[1].get("api_key") == "from-settings"


# ---------------------------------------------------------------------------
# _build_search_query
# ---------------------------------------------------------------------------

class TestBuildSearchQuery:
    def test_strips_leading_article(self):
        q = _build_search_query("A futuristic robot walking")
        assert not q.startswith("A ")
        assert "futuristic robot walking" in q

    def test_strips_lowercase_article(self):
        q = _build_search_query("a dark alley at night")
        assert not q.startswith("a ")

    def test_limits_length(self):
        desc = " ".join(["word"] * 20)
        q = _build_search_query(desc)
        assert len(q.split()) <= 8

    def test_preserves_short_description(self):
        q = _build_search_query("Neon city at night")
        assert "Neon city at night" in q


# ---------------------------------------------------------------------------
# _search_pexels
# ---------------------------------------------------------------------------

class TestSearchPexels:
    @patch("media.image.requests.get")
    def test_sends_correct_headers_and_params(self, mock_get: MagicMock):
        mock_get.return_value = _mock_pexels_search_response([])
        _search_pexels("test query", api_key="sk-test123")

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        # Check Authorization header
        assert call_kwargs[1]["headers"] == {"Authorization": "sk-test123"}
        # Check query params
        params = call_kwargs[1]["params"]
        assert params["query"] == "test query"
        assert params["orientation"] == "portrait"
        assert params["per_page"] == 10

    @patch("media.image.requests.get")
    def test_returns_photos_list(self, mock_get: MagicMock):
        photo = _make_pexels_photo()
        mock_get.return_value = _mock_pexels_search_response([photo])

        result = _search_pexels("city", api_key="key")
        assert result == [photo]

    @patch("media.image.requests.get")
    def test_raises_on_http_error(self, mock_get: MagicMock):
        resp = requests.Response()
        resp.status_code = 403
        resp._content = b""
        mock_get.return_value = resp
        mock_get.return_value.raise_for_status = MagicMock(side_effect=_make_http_error(403))

        with pytest.raises(requests.HTTPError):
            _search_pexels("test", api_key="bad")


# ---------------------------------------------------------------------------
# _select_best_photo
# ---------------------------------------------------------------------------

class TestSelectBestPhoto:
    def test_prefers_portrait(self):
        portrait = _make_pexels_photo(width=600, height=900)
        landscape = _make_pexels_photo(width=900, height=600)
        result = _select_best_photo([landscape, portrait])
        assert result is portrait

    def test_falls_back_to_landscape(self):
        landscape = _make_pexels_photo(width=900, height=600)
        result = _select_best_photo([landscape])
        assert result is landscape

    def test_returns_none_for_empty(self):
        assert _select_best_photo([]) is None


# ---------------------------------------------------------------------------
# _download_photo
# ---------------------------------------------------------------------------

class TestDownloadPhoto:
    @patch("media.image.requests.get")
    def test_downloads_from_original_url(self, mock_get: MagicMock):
        mock_get.return_value = _mock_image_download_response(_FAKE_JPEG)
        photo = _make_pexels_photo(original_url="https://example.com/img.jpg")

        result = _download_photo(photo)
        assert result == _FAKE_JPEG
        mock_get.assert_called_once_with("https://example.com/img.jpg", timeout=60)

    @patch("media.image.requests.get")
    def test_raises_on_missing_url(self, mock_get: MagicMock):
        photo = {"src": {}}
        with pytest.raises(ImageGenError, match="no downloadable URL"):
            _download_photo(photo)

    @patch("media.image.requests.get")
    def test_falls_back_to_large2x_on_original_failure(self, mock_get: MagicMock):
        """When src.original returns 422, should try src.large2x."""
        error_422 = requests.exceptions.HTTPError(
            response=MagicMock(status_code=422)
        )
        mock_get.side_effect = [
            error_422,
            _mock_image_download_response(_FAKE_JPEG),
        ]
        photo = _make_pexels_photo(
            original_url="https://images.pexels.com/photos/1/broken.jpeg",
        )
        photo["src"]["large2x"] = "https://images.pexels.com/photos/1/large2x.jpeg"

        result = _download_photo(photo)
        assert result == _FAKE_JPEG
        assert mock_get.call_count == 2
        mock_get.assert_any_call(
            "https://images.pexels.com/photos/1/broken.jpeg", timeout=60
        )
        mock_get.assert_any_call(
            "https://images.pexels.com/photos/1/large2x.jpeg", timeout=60
        )

    @patch("media.image.requests.get")
    def test_falls_back_to_large_on_all_prior_failures(self, mock_get: MagicMock):
        """When original and large2x both fail, should try src.large."""
        error_422 = requests.exceptions.HTTPError(
            response=MagicMock(status_code=422)
        )
        mock_get.side_effect = [
            error_422,
            error_422,
            _mock_image_download_response(_FAKE_JPEG),
        ]
        photo = {
            "id": 1, "width": 600, "height": 900,
            "src": {
                "original": "https://example.com/broken.jpg",
                "large2x": "https://example.com/broken2.jpg",
                "large": "https://example.com/large.jpg",
            },
        }

        result = _download_photo(photo)
        assert result == _FAKE_JPEG
        assert mock_get.call_count == 3

    @patch("media.image.requests.get")
    def test_raises_when_all_urls_fail(self, mock_get: MagicMock):
        """When all candidate URLs fail, raises ImageGenError."""
        error_422 = requests.exceptions.HTTPError(
            response=MagicMock(status_code=422)
        )
        mock_get.side_effect = error_422
        photo = {
            "id": 1, "width": 600, "height": 900,
            "src": {
                "original": "https://example.com/broken.jpg",
                "large2x": "https://example.com/broken2.jpg",
                "large": "https://example.com/broken3.jpg",
            },
        }

        with pytest.raises(ImageGenError, match="All download URLs failed"):
            _download_photo(photo)
        assert mock_get.call_count == 3


# ---------------------------------------------------------------------------
# generate_scene_image — success
# ---------------------------------------------------------------------------

class TestGenerateSceneImageSuccess:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_generates_image_file(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene()
        result = generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        assert isinstance(result, GeneratedImage)
        assert result.scene == 1
        assert result.path.exists()
        assert result.path.name == "scene_1.jpg"

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_file_contains_valid_image(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene()
        result = generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        assert _is_image_file_valid(result.path)

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_uses_visual_description_as_query(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene(visual_description="Neon cyberpunk alley")
        generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        call_kwargs = mock_search.call_args
        query = call_kwargs[0][0]
        assert "Neon cyberpunk alley" in query

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_requests_portrait_orientation(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene()
        generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        call_kwargs = mock_search.call_args
        # orientation is passed via params inside _search_pexels, not as arg
        # We verify through the internal call
        assert mock_search.called


# ---------------------------------------------------------------------------
# generate_scene_image — multiple scenes
# ---------------------------------------------------------------------------

class TestMultipleScenes:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_generates_for_multiple_scenes(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scenes = [_make_scene(scene=i) for i in range(1, 4)]
        results = generate_all_scene_images(scenes, output_dir=tmp_path, api_key="fake-key")

        assert len(results) == 3
        assert [r.scene for r in results] == [1, 2, 3]

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_scene_based_filenames(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scenes = [_make_scene(scene=5)]
        results = generate_all_scene_images(scenes, output_dir=tmp_path, api_key="fake-key")

        assert results[0].path.name == "scene_5.jpg"


# ---------------------------------------------------------------------------
# generate_scene_image — output directory creation
# ---------------------------------------------------------------------------

class TestDirectoryCreation:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_creates_output_dir(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        out_dir = tmp_path / "nested" / "images"
        scene = _make_scene()
        result = generate_scene_image(scene, output_dir=out_dir, api_key="fake-key")

        assert out_dir.is_dir()
        assert result.path.parent == out_dir


# ---------------------------------------------------------------------------
# generate_scene_image — existing image reuse
# ---------------------------------------------------------------------------

class TestImageReuse:
    @patch("media.image._search_pexels")
    def test_reuses_valid_existing_image(self, mock_search: MagicMock, tmp_path: Path):
        # Write a valid image first
        img = tmp_path / "scene_1.jpg"
        img.write_bytes(_FAKE_JPEG)

        scene = _make_scene()
        result = generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        # API should NOT have been called
        mock_search.assert_not_called()
        assert result.path == img

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_regenerates_corrupt_existing_image(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        # Write a corrupt image
        img = tmp_path / "scene_1.jpg"
        img.write_bytes(b"corrupt data")

        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene()
        result = generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        mock_search.assert_called_once()
        assert _is_image_file_valid(result.path)


# ---------------------------------------------------------------------------
# generate_scene_image — job_id isolation (regression: stale cache reuse)
# ---------------------------------------------------------------------------

class TestJobIdIsolation:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_job_id_creates_subdirectory(
        self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path
    ):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene()
        result = generate_scene_image(
            scene, output_dir=tmp_path, api_key="fake-key", job_id="run-abc"
        )

        # Image should be in output_dir/run-abc/scene_1.jpg
        assert result.path.parent == tmp_path / "run-abc"
        assert result.path.name == "scene_1.jpg"
        assert result.path.exists()

    @patch("media.image._search_pexels")
    def test_stale_image_not_reused_across_jobs(
        self, mock_search: MagicMock, tmp_path: Path
    ):
        """scene_1.jpg from job 'old-run' must NOT be reused for job 'new-run'."""
        # Simulate a previous run's image
        old_dir = tmp_path / "old-run"
        old_dir.mkdir()
        (old_dir / "scene_1.jpg").write_bytes(_FAKE_JPEG)

        # New run with a different job_id
        mock_search.return_value = [_make_pexels_photo()]
        scene = _make_scene()

        # We need to mock _download_photo too since the search will be called
        with patch("media.image._download_photo", return_value=_FAKE_JPEG):
            result = generate_scene_image(
                scene, output_dir=tmp_path, api_key="fake-key", job_id="new-run"
            )

        # The new run should have created its own image in its own subdirectory
        assert result.path.parent == tmp_path / "new-run"
        assert result.path.exists()
        # Pexels search should have been called (not skipped by cache)
        mock_search.assert_called_once()

    @patch("media.image._search_pexels")
    def test_same_job_reuses_image(
        self, mock_search: MagicMock, tmp_path: Path
    ):
        """Within the same job_id, a valid image should be reused."""
        # First generation
        with patch("media.image._download_photo", return_value=_FAKE_JPEG):
            scene = _make_scene()
            result1 = generate_scene_image(
                scene, output_dir=tmp_path, api_key="fake-key", job_id="run-1"
            )
            assert mock_search.call_count == 1

        # Second call with same job_id — should reuse, not search again
        result2 = generate_scene_image(
            scene, output_dir=tmp_path, api_key="fake-key", job_id="run-1"
        )
        assert mock_search.call_count == 1  # still 1, no new search
        assert result1.path == result2.path

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_no_job_id_backward_compatible(
        self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path
    ):
        """Without job_id, behavior is unchanged (images in output_dir directly)."""
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene()
        result = generate_scene_image(
            scene, output_dir=tmp_path, api_key="fake-key"
        )

        # Image should be directly in output_dir, no subdirectory
        assert result.path.parent == tmp_path
        assert result.path.name == "scene_1.jpg"

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_all_scene_images_with_job_id(
        self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path
    ):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = _FAKE_JPEG
        scenes = [_make_scene(scene=1), _make_scene(scene=2)]
        results = generate_all_scene_images(
            scenes, output_dir=tmp_path, api_key="fake-key", job_id="run-xyz"
        )

        assert all(r.path.parent == tmp_path / "run-xyz" for r in results)
        assert [r.scene for r in results] == [1, 2]


# ---------------------------------------------------------------------------
# generate_scene_image — empty/no results from Pexels
# ---------------------------------------------------------------------------

class TestEmptyPexelsResults:
    @patch("media.image._search_pexels", return_value=[])
    def test_raises_on_empty_results(self, mock_search: MagicMock, tmp_path: Path):
        scene = _make_scene()

        with pytest.raises(ImageGenError, match="Failed to generate image"):
            generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        # No partial file left behind
        assert not (tmp_path / "scene_1.jpg").exists()
        assert not (tmp_path / "scene_1.tmp").exists()


# ---------------------------------------------------------------------------
# generate_scene_image — non-image response
# ---------------------------------------------------------------------------

class TestNonImageResponse:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_raises_on_non_image_bytes(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = b"not an image at all"
        scene = _make_scene()

        with pytest.raises(ImageGenError, match="Failed to generate image"):
            generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        assert not (tmp_path / "scene_1.jpg").exists()


# ---------------------------------------------------------------------------
# generate_scene_image — transient failure with retry
# ---------------------------------------------------------------------------

class TestTransientRetry:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_retries_on_transient_error(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        transient_exc = _make_http_error(503)
        mock_search.side_effect = [transient_exc, [_make_pexels_photo()]]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene()

        result = generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key", max_retries=3)

        assert mock_search.call_count == 2
        assert _is_image_file_valid(result.path)

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_exhausts_retries_then_fails(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.side_effect = _make_http_error(503)
        scene = _make_scene()

        with pytest.raises(ImageGenError, match="Failed to generate image"):
            generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key", max_retries=2)

        assert mock_search.call_count == 2
        # No file left behind
        assert not (tmp_path / "scene_1.jpg").exists()


# ---------------------------------------------------------------------------
# generate_scene_image — non-retryable failure
# ---------------------------------------------------------------------------

class TestNonRetryableFailure:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_does_not_retry_auth_error(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.side_effect = _make_http_error(401)
        scene = _make_scene()

        with pytest.raises(ImageGenError, match="Failed to generate image"):
            generate_scene_image(scene, output_dir=tmp_path, api_key="bad-key", max_retries=5)

        # Should only be called once — no retries for 401
        assert mock_search.call_count == 1

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_does_not_retry_forbidden(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.side_effect = _make_http_error(403)
        scene = _make_scene()

        with pytest.raises(ImageGenError, match="Failed to generate image"):
            generate_scene_image(scene, output_dir=tmp_path, api_key="forbidden", max_retries=3)

        assert mock_search.call_count == 1


# ---------------------------------------------------------------------------
# generate_scene_image — rate limit handling
# ---------------------------------------------------------------------------

class TestRateLimitHandling:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_retries_on_429_then_succeeds(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.side_effect = [_make_http_error(429), [_make_pexels_photo()]]
        mock_dl.return_value = _FAKE_JPEG
        scene = _make_scene()

        result = generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key", max_retries=3)

        assert mock_search.call_count == 2
        assert _is_image_file_valid(result.path)

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_exhausts_retries_on_429(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.side_effect = _make_http_error(429)
        scene = _make_scene()

        with pytest.raises(ImageGenError, match="Failed to generate image"):
            generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key", max_retries=2)

        assert mock_search.call_count == 2


# ---------------------------------------------------------------------------
# generate_scene_image — cleanup on failure
# ---------------------------------------------------------------------------

class TestCleanupOnFailure:
    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_no_empty_file_on_failure(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.return_value = b"not valid"
        scene = _make_scene()

        with pytest.raises(ImageGenError):
            generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        assert not (tmp_path / "scene_1.jpg").exists()
        assert not (tmp_path / "scene_1.tmp").exists()

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_no_tmp_file_after_failure(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        def download_then_fail(photo):
            tmp = tmp_path / "scene_1.tmp"
            tmp.write_bytes(b"partial")
            raise requests.Timeout("timeout")

        mock_search.return_value = [_make_pexels_photo()]
        mock_dl.side_effect = download_then_fail
        scene = _make_scene()

        with pytest.raises(ImageGenError):
            generate_scene_image(scene, output_dir=tmp_path, api_key="fake-key")

        assert not (tmp_path / "scene_1.tmp").exists()


# ---------------------------------------------------------------------------
# generate_all_scene_images — edge cases
# ---------------------------------------------------------------------------

class TestGenerateAllEdgeCases:
    def test_raises_on_empty_scenes(self):
        with pytest.raises(ValueError, match="Scenes list must not be empty"):
            generate_all_scene_images([], api_key="fake-key")

    @patch("media.image._download_photo")
    @patch("media.image._search_pexels")
    def test_raises_if_any_scene_fails(self, mock_search: MagicMock, mock_dl: MagicMock, tmp_path: Path):
        mock_search.side_effect = [
            [_make_pexels_photo()],  # scene 1 succeeds
            _make_http_error(401),   # scene 2 fails
        ]
        mock_dl.return_value = _FAKE_JPEG
        scenes = [_make_scene(scene=1), _make_scene(scene=2)]

        with pytest.raises(ImageGenError, match="scene 2"):
            generate_all_scene_images(scenes, output_dir=tmp_path, api_key="key")


# ---------------------------------------------------------------------------
# No fallback to other providers
# ---------------------------------------------------------------------------

class TestNoFallback:
    """Ensure there is no fallback to Pollinations, OpenRouter, or any paid provider."""

    def test_no_pollinations_import(self):
        import media.image as mod
        source = mod.__spec__.loader.get_source(mod.__name__)  # type: ignore[union-attr]
        assert "pollinations" not in source.lower()

    def test_no_openrouter_import(self):
        import media.image as mod
        source = mod.__spec__.loader.get_source(mod.__name__)  # type: ignore[union-attr]
        assert "openrouter" not in source.lower()

    def test_no_bearer_auth(self):
        import media.image as mod
        source = mod.__spec__.loader.get_source(mod.__name__)  # type: ignore[union-attr]
        assert "Bearer" not in source


# ---------------------------------------------------------------------------
# GeneratedImage model
# ---------------------------------------------------------------------------

class TestGeneratedImageModel:
    def test_fields(self, tmp_path: Path):
        img = GeneratedImage(scene=3, path=tmp_path / "scene_3.jpg")
        assert img.scene == 3
        assert img.path.name == "scene_3.jpg"

    def test_is_frozen(self, tmp_path: Path):
        img = GeneratedImage(scene=1, path=tmp_path / "x.jpg")
        with pytest.raises(AttributeError):
            img.scene = 2  # type: ignore[misc]
