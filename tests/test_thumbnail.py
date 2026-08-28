"""Tests for the media.thumbnail module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from media.image import GeneratedImage
from media.thumbnail import (
    GeneratedThumbnail,
    ThumbnailError,
    generate_thumbnail,
    _is_valid_image_file,
    _has_image_signature,
    _select_best_source_image,
    _THUMBNAIL_WIDTH,
    _THUMBNAIL_HEIGHT,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Minimal valid JPEG
_FAKE_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    + b"\xff\xdb\x00C\x00" + b"\x08" * 64
    + b"\xff\xc0\x00\x11\x08" + b"\x00" * 20
    + b"\xff\xd9"
)


def _make_image(tmp_path: Path, scene: int = 1) -> GeneratedImage:
    """Create a valid JPEG file for testing."""
    img_path = tmp_path / f"scene_{scene}.jpg"
    img_path.write_bytes(_FAKE_JPEG)
    return GeneratedImage(scene=scene, path=img_path)


# ---------------------------------------------------------------------------
# _has_image_signature
# ---------------------------------------------------------------------------

class TestHasImageSignature:
    def test_jpeg(self):
        assert _has_image_signature(b"\xff\xd8\xff" + b"\x00" * 9) is True

    def test_png(self):
        assert _has_image_signature(b"\x89PNG" + b"\x00" * 8) is True

    def test_gif(self):
        assert _has_image_signature(b"GIF8" + b"\x00" * 8) is True

    def test_webp(self):
        assert _has_image_signature(b"RIFF\x00\x00\x00\x00WEBP") is True

    def test_unknown(self):
        assert _has_image_signature(b"\x00\x00\x00\x00") is False

    def test_too_short(self):
        assert _has_image_signature(b"\xff") is False


# ---------------------------------------------------------------------------
# _is_valid_image_file
# ---------------------------------------------------------------------------

class TestIsValidImageFile:
    def test_valid_jpeg(self, tmp_path: Path):
        img = tmp_path / "test.jpg"
        img.write_bytes(_FAKE_JPEG)
        assert _is_valid_image_file(img) is True

    def test_nonexistent(self, tmp_path: Path):
        assert _is_valid_image_file(tmp_path / "nope.jpg") is False

    def test_empty_file(self, tmp_path: Path):
        img = tmp_path / "empty.jpg"
        img.write_bytes(b"")
        assert _is_valid_image_file(img) is False

    def test_too_small(self, tmp_path: Path):
        img = tmp_path / "tiny.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        assert _is_valid_image_file(img) is False


# ---------------------------------------------------------------------------
# _select_best_source_image
# ---------------------------------------------------------------------------

class TestSelectBestSourceImage:
    def test_prefers_scene_1(self, tmp_path: Path):
        img1 = _make_image(tmp_path, 1)
        img2 = _make_image(tmp_path, 2)
        result = _select_best_source_image([img2.path, img1.path], "Test Topic")
        assert result == img1.path

    def test_returns_first_valid_if_no_scene_1(self, tmp_path: Path):
        img2 = _make_image(tmp_path, 2)
        img3 = _make_image(tmp_path, 3)
        result = _select_best_source_image([img2.path, img3.path], "Test Topic")
        assert result == img2.path

    def test_returns_none_when_no_valid_images(self, tmp_path: Path):
        result = _select_best_source_image([], "Test Topic")
        assert result is None

    def test_skips_invalid_files(self, tmp_path: Path):
        invalid = tmp_path / "scene_1.jpg"
        invalid.write_bytes(b"not an image")
        valid = _make_image(tmp_path, 2)
        result = _select_best_source_image([invalid, valid.path], "Test Topic")
        assert result == valid.path


# ---------------------------------------------------------------------------
# generate_thumbnail — success
# ---------------------------------------------------------------------------

class TestGenerateThumbnailSuccess:
    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_returns_generated_thumbnail(self, mock_create: MagicMock, tmp_path: Path):
        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)

        images = [_make_image(tmp_path)]
        result = generate_thumbnail("Test Topic", images, output_dir=tmp_path)

        assert isinstance(result, GeneratedThumbnail)
        assert result.path.exists()
        assert result.width == _THUMBNAIL_WIDTH
        assert result.height == _THUMBNAIL_HEIGHT

    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_saves_to_deterministic_path(self, mock_create: MagicMock, tmp_path: Path):
        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)

        images = [_make_image(tmp_path)]
        result = generate_thumbnail("Test Topic", images, output_dir=tmp_path)

        assert result.path.name == "thumbnail.jpg"
        assert result.path.parent == tmp_path


# ---------------------------------------------------------------------------
# generate_thumbnail — 1280x720 dimensions
# ---------------------------------------------------------------------------

class TestThumbnailDimensions:
    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_default_dimensions(self, mock_create: MagicMock, tmp_path: Path):
        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)

        images = [_make_image(tmp_path)]
        result = generate_thumbnail("Test Topic", images, output_dir=tmp_path)

        assert result.width == 1280
        assert result.height == 720


# ---------------------------------------------------------------------------
# generate_thumbnail — output directory creation
# ---------------------------------------------------------------------------

class TestDirectoryCreation:
    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_creates_nested_output_dir(self, mock_create: MagicMock, tmp_path: Path):
        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)

        out_dir = tmp_path / "nested" / "thumbs"
        images = [_make_image(tmp_path)]
        result = generate_thumbnail("Test Topic", images, output_dir=out_dir)

        assert out_dir.is_dir()
        assert result.path.parent == out_dir


# ---------------------------------------------------------------------------
# generate_thumbnail — reuse existing
# ---------------------------------------------------------------------------

class TestThumbnailReuse:
    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_reuses_valid_existing_thumbnail(self, mock_create: MagicMock, tmp_path: Path):
        # Write a valid thumbnail first
        thumb = tmp_path / "thumbnail.jpg"
        thumb.write_bytes(_FAKE_JPEG)

        images = [_make_image(tmp_path)]
        result = generate_thumbnail("Test Topic", images, output_dir=tmp_path)

        # Should NOT have called create
        mock_create.assert_not_called()
        assert result.path == thumb

    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_regenerates_corrupt_existing_thumbnail(self, mock_create: MagicMock, tmp_path: Path):
        # Write corrupt thumbnail
        thumb = tmp_path / "thumbnail.jpg"
        thumb.write_bytes(b"corrupt data")

        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)
        images = [_make_image(tmp_path)]
        result = generate_thumbnail("Test Topic", images, output_dir=tmp_path)

        assert mock_create.call_count == 1
        assert _is_valid_image_file(result.path)


# ---------------------------------------------------------------------------
# generate_thumbnail — job_id isolation (regression: stale cache reuse)
# ---------------------------------------------------------------------------

class TestThumbnailJobIdIsolation:
    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_job_id_creates_subdirectory(self, mock_create: MagicMock, tmp_path: Path):
        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)

        images = [_make_image(tmp_path)]
        result = generate_thumbnail(
            "Test Topic", images, output_dir=tmp_path, job_id="run-abc"
        )

        assert result.path.parent == tmp_path / "run-abc"
        assert result.path.name == "thumbnail.jpg"
        assert result.path.exists()

    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_stale_thumbnail_not_reused_across_jobs(
        self, mock_create: MagicMock, tmp_path: Path
    ):
        """thumbnail.jpg from job 'old-run' must NOT be reused for job 'new-run'."""
        # Simulate a previous run's thumbnail
        old_dir = tmp_path / "old-run"
        old_dir.mkdir()
        (old_dir / "thumbnail.jpg").write_bytes(_FAKE_JPEG)

        # New run with a different job_id
        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)
        images = [_make_image(tmp_path)]

        result = generate_thumbnail(
            "Test Topic", images, output_dir=tmp_path, job_id="new-run"
        )

        # The new run should have created its own thumbnail in its own subdirectory
        assert result.path.parent == tmp_path / "new-run"
        assert result.path.exists()
        # _create_thumbnail_from_source should have been called (not skipped by cache)
        mock_create.assert_called_once()

    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_same_job_reuses_thumbnail(
        self, mock_create: MagicMock, tmp_path: Path
    ):
        """Within the same job_id, a valid thumbnail should be reused."""
        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)
        images = [_make_image(tmp_path)]

        # First generation
        result1 = generate_thumbnail(
            "Test Topic", images, output_dir=tmp_path, job_id="run-1"
        )
        assert mock_create.call_count == 1

        # Second call with same job_id — should reuse, not create again
        result2 = generate_thumbnail(
            "Test Topic", images, output_dir=tmp_path, job_id="run-1"
        )
        assert mock_create.call_count == 1  # still 1, no new creation
        assert result1.path == result2.path

    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_no_job_id_backward_compatible(
        self, mock_create: MagicMock, tmp_path: Path
    ):
        """Without job_id, behavior is unchanged (thumbnail in output_dir directly)."""
        mock_create.side_effect = lambda src, dst, **kw: dst.write_bytes(_FAKE_JPEG)
        images = [_make_image(tmp_path)]

        result = generate_thumbnail("Test Topic", images, output_dir=tmp_path)

        assert result.path.parent == tmp_path
        assert result.path.name == "thumbnail.jpg"


# ---------------------------------------------------------------------------
# generate_thumbnail — missing source image
# ---------------------------------------------------------------------------

class TestMissingSourceImage:
    def test_creates_placeholder_when_no_images(self, tmp_path: Path):
        result = generate_thumbnail("Test Topic", None, output_dir=tmp_path)

        assert result.path.exists()
        assert _is_valid_image_file(result.path)

    def test_creates_placeholder_when_empty_images(self, tmp_path: Path):
        result = generate_thumbnail("Test Topic", [], output_dir=tmp_path)

        assert result.path.exists()
        assert _is_valid_image_file(result.path)


# ---------------------------------------------------------------------------
# generate_thumbnail — failure cleanup
# ---------------------------------------------------------------------------

class TestFailureCleanup:
    @patch("media.thumbnail._create_thumbnail_from_source")
    def test_cleans_up_on_failure(self, mock_create: MagicMock, tmp_path: Path):
        mock_create.side_effect = RuntimeError("simulated failure")

        images = [_make_image(tmp_path)]
        with pytest.raises(ThumbnailError, match="Failed to generate"):
            generate_thumbnail("Test Topic", images, output_dir=tmp_path)

        assert not (tmp_path / "thumbnail.jpg").exists()
        assert not (tmp_path / "thumbnail.tmp.jpg").exists()


# ---------------------------------------------------------------------------
# GeneratedThumbnail model
# ---------------------------------------------------------------------------

class TestGeneratedThumbnailModel:
    def test_fields(self, tmp_path: Path):
        t = GeneratedThumbnail(path=tmp_path / "thumb.jpg", width=1280, height=720)
        assert t.path.name == "thumb.jpg"
        assert t.width == 1280
        assert t.height == 720

    def test_is_frozen(self, tmp_path: Path):
        t = GeneratedThumbnail(path=tmp_path / "x.jpg", width=100, height=100)
        with pytest.raises(AttributeError):
            t.width = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _create_thumbnail_from_source — unit test with PIL
# ---------------------------------------------------------------------------

class TestCreateThumbnailFromSource:
    def test_creates_valid_jpeg(self, tmp_path: Path):
        from media.thumbnail import _create_thumbnail_from_source

        # Create a source image (need PIL for this test)
        try:
            from PIL import Image
            src = tmp_path / "source.jpg"
            img = Image.new("RGB", (800, 600), color=(100, 150, 200))
            img.save(src, "JPEG")

            dst = tmp_path / "output.jpg"
            _create_thumbnail_from_source(src, dst)

            assert dst.exists()
            assert _is_valid_image_file(dst)

            # Verify dimensions
            result_img = Image.open(dst)
            assert result_img.size == (1280, 720)
        except ImportError:
            pytest.skip("PIL not installed")
