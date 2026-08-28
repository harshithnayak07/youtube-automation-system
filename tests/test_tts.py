"""Tests for the media.tts module — all use mocks, no real API calls."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from media.tts import (
    GeneratedAudio,
    TTSError,
    TTSConfigError,
    TTSProvider,
    EdgeTTSProvider,
    generate_narration_audio,
    _is_valid_audio_file,
    _has_audio_signature,
    _extract_mp3_duration,
    _is_transient,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_NARRATION = "OpenAI just announced GPT-5, a major leap in reasoning capabilities."


def _fake_mp3_bytes(duration_frames: int = 100) -> bytes:
    """Create minimal valid MP3 bytes with a Xing header."""
    # ID3v2 header
    id3 = b"ID3" + b"\x03\x00\x00" + b"\x00" * 10

    # Xing header (for duration parsing)
    xing_start = b"Xing"
    flags = b"\x00" * 4
    total_frames = struct.pack(">I", duration_frames)
    xing = xing_start + flags + total_frames + b"\x00" * 100

    # Frame sync + header (MPEG1 Layer III 128kbps)
    # 0xFF 0xFB 0x90 0x00 = MPEG1, Layer III, 128kbps, 44100Hz, stereo
    frame_header = b"\xff\xfb\x90\x00"
    frame_size = 417  # typical frame size for 128kbps
    frame_data = b"\x00" * (frame_size - 4)

    return id3 + xing + frame_header + frame_data


class StubTTSProvider:
    """Deterministic TTS provider for testing."""

    def __init__(
        self,
        audio_bytes: bytes | None = None,
        error: Exception | None = None,
    ):
        self._audio = audio_bytes or _fake_mp3_bytes()
        self._error = error
        self._call_count = 0

    def synthesize(self, text: str, output_path: Path, voice: str) -> None:
        self._call_count += 1
        if self._error:
            raise self._error
        output_path.write_bytes(self._audio)


class FailingTTSProvider:
    """Stub that always raises."""

    def synthesize(self, text: str, output_path: Path, voice: str) -> None:
        raise ConnectionError("simulated provider failure")


# ---------------------------------------------------------------------------
# _has_audio_signature
# ---------------------------------------------------------------------------

class TestHasAudioSignature:
    def test_mp3_with_id3(self):
        assert _has_audio_signature(b"ID3" + b"\x00" * 9) is True

    def test_mp3_with_frame_sync(self):
        assert _has_audio_signature(b"\xff\xfb\x90\x00" + b"\x00" * 8) is True

    def test_wav(self):
        assert _has_audio_signature(b"RIFF\x00\x00\x00\x00WAVE") is True

    def test_ogg(self):
        assert _has_audio_signature(b"OggS" + b"\x00" * 8) is True

    def test_flac(self):
        assert _has_audio_signature(b"fLaC" + b"\x00" * 8) is True

    def test_unknown(self):
        assert _has_audio_signature(b"\x00\x00\x00\x00") is False

    def test_too_short(self):
        assert _has_audio_signature(b"\xff") is False


# ---------------------------------------------------------------------------
# _is_valid_audio_file
# ---------------------------------------------------------------------------

class TestIsValidAudioFile:
    def test_valid_mp3(self, tmp_path: Path):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(_fake_mp3_bytes())
        assert _is_valid_audio_file(audio) is True

    def test_nonexistent(self, tmp_path: Path):
        assert _is_valid_audio_file(tmp_path / "nope.mp3") is False

    def test_empty_file(self, tmp_path: Path):
        audio = tmp_path / "empty.mp3"
        audio.write_bytes(b"")
        assert _is_valid_audio_file(audio) is False

    def test_too_small(self, tmp_path: Path):
        audio = tmp_path / "tiny.mp3"
        audio.write_bytes(b"\xff" * 50)
        assert _is_valid_audio_file(audio) is False

    def test_invalid_content(self, tmp_path: Path):
        audio = tmp_path / "bad.mp3"
        audio.write_bytes(b"this is not audio at all, just random text data here" * 10)
        assert _is_valid_audio_file(audio) is False


# ---------------------------------------------------------------------------
# _extract_mp3_duration
# ---------------------------------------------------------------------------

class TestExtractMp3Duration:
    def test_parses_xing_header(self, tmp_path: Path):
        # 100 frames * 1152 samples / 44100 Hz = ~2.61 seconds
        audio = tmp_path / "test.mp3"
        audio.write_bytes(_fake_mp3_bytes(duration_frames=100))
        duration = _extract_mp3_duration(audio)
        assert duration == 2.61

    def test_returns_zero_for_invalid(self, tmp_path: Path):
        audio = tmp_path / "bad.mp3"
        audio.write_bytes(b"not an mp3")
        duration = _extract_mp3_duration(audio)
        assert duration == 0.0

    def test_returns_zero_for_missing(self, tmp_path: Path):
        duration = _extract_mp3_duration(tmp_path / "missing.mp3")
        assert duration == 0.0


# ---------------------------------------------------------------------------
# _is_transient
# ---------------------------------------------------------------------------

class TestIsTransient:
    def test_connection_error(self):
        assert _is_transient(ConnectionError("conn")) is True

    def test_timeout(self):
        assert _is_transient(TimeoutError("timeout")) is True

    def test_os_error(self):
        assert _is_transient(OSError("disk full")) is True

    def test_value_error_not_transient(self):
        assert _is_transient(ValueError("bad input")) is False


# ---------------------------------------------------------------------------
# generate_narration_audio — success
# ---------------------------------------------------------------------------

class TestGenerateNarrationAudioSuccess:
    def test_returns_generated_audio(self, tmp_path: Path):
        provider = StubTTSProvider()
        result = generate_narration_audio(
            _NARRATION, output_dir=tmp_path, provider=provider,
        )

        assert isinstance(result, GeneratedAudio)
        assert result.path.exists()
        assert result.duration_seconds >= 0

    def test_creates_deterministic_filename(self, tmp_path: Path):
        provider = StubTTSProvider()
        result1 = generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)
        result2 = generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)

        # Same narration → same filename
        assert result1.path == result2.path

    def test_different_narration_different_file(self, tmp_path: Path):
        provider = StubTTSProvider()
        r1 = generate_narration_audio("First narration.", output_dir=tmp_path, provider=provider)
        r2 = generate_narration_audio("Second narration.", output_dir=tmp_path, provider=provider)

        assert r1.path != r2.path

    def test_file_is_valid_audio(self, tmp_path: Path):
        provider = StubTTSProvider()
        result = generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)

        assert _is_valid_audio_file(result.path)


# ---------------------------------------------------------------------------
# generate_narration_audio — output directory creation
# ---------------------------------------------------------------------------

class TestDirectoryCreation:
    def test_creates_nested_output_dir(self, tmp_path: Path):
        out_dir = tmp_path / "nested" / "audio"
        provider = StubTTSProvider()
        result = generate_narration_audio(_NARRATION, output_dir=out_dir, provider=provider)

        assert out_dir.is_dir()
        assert result.path.parent == out_dir


# ---------------------------------------------------------------------------
# generate_narration_audio — audio reuse
# ---------------------------------------------------------------------------

class TestAudioReuse:
    def test_reuses_valid_existing_audio(self, tmp_path: Path):
        import hashlib as _hashlib
        # Compute the actual hash for this narration
        content_hash = _hashlib.sha256(_NARRATION.encode("utf-8")).hexdigest()[:12]
        audio = tmp_path / f"narration_{content_hash}.mp3"
        audio.write_bytes(_fake_mp3_bytes())

        provider = StubTTSProvider()
        result = generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)

        # Provider should NOT have been called
        assert provider._call_count == 0
        assert result.path == audio

    def test_regenerates_invalid_existing_audio(self, tmp_path: Path):
        import hashlib as _hashlib
        content_hash = _hashlib.sha256(_NARRATION.encode("utf-8")).hexdigest()[:12]
        audio = tmp_path / f"narration_{content_hash}.mp3"
        audio.write_bytes(b"not valid audio")

        provider = StubTTSProvider()
        result = generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)

        assert provider._call_count == 1
        assert _is_valid_audio_file(result.path)


# ---------------------------------------------------------------------------
# generate_narration_audio — empty/invalid output
# ---------------------------------------------------------------------------

class TestEmptyOutput:
    def test_raises_on_empty_narration(self):
        provider = StubTTSProvider()
        with pytest.raises(ValueError, match="Narration must not be empty"):
            generate_narration_audio("", provider=provider)

    def test_raises_on_whitespace_narration(self):
        provider = StubTTSProvider()
        with pytest.raises(ValueError, match="Narration must not be empty"):
            generate_narration_audio("   \n  ", provider=provider)

    def test_raises_when_tts_produces_empty_file(self, tmp_path: Path):
        provider = StubTTSProvider(audio_bytes=b"\x00" * 50)  # Too small to be valid
        with pytest.raises(TTSError, match="invalid or empty"):
            generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)

    def test_raises_when_tts_produces_invalid_file(self, tmp_path: Path):
        provider = StubTTSProvider(audio_bytes=b"not audio data at all, just random garbage here" * 5)
        with pytest.raises(TTSError, match="invalid or empty"):
            generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)


# ---------------------------------------------------------------------------
# generate_narration_audio — duration extraction
# ---------------------------------------------------------------------------

class TestDurationExtraction:
    def test_returns_nonzero_duration(self, tmp_path: Path):
        provider = StubTTSProvider(audio_bytes=_fake_mp3_bytes(duration_frames=200))
        result = generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)

        assert result.duration_seconds > 0

    def test_duration_matches_xing_header(self, tmp_path: Path):
        # 300 frames * 1152 / 44100 ≈ 7.85 seconds
        provider = StubTTSProvider(audio_bytes=_fake_mp3_bytes(duration_frames=300))
        result = generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)

        assert abs(result.duration_seconds - 7.85) < 0.1


# ---------------------------------------------------------------------------
# generate_narration_audio — transient failure with retry
# ---------------------------------------------------------------------------

class TestTransientRetry:
    def test_retries_on_connection_error(self, tmp_path: Path):
        provider = StubTTSProvider(
            audio_bytes=_fake_mp3_bytes(),
            error=ConnectionError("connection refused"),
        )
        # First call fails, second succeeds
        call_count = 0
        original = provider.synthesize

        def counting_synthesize(text, output_path, voice):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection refused")
            output_path.write_bytes(_fake_mp3_bytes())

        provider.synthesize = counting_synthesize
        result = generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider, max_retries=3)

        assert call_count == 2
        assert result.path.exists()

    def test_exhausts_retries_then_fails(self, tmp_path: Path):
        provider = FailingTTSProvider()
        with pytest.raises(TTSError, match="Failed to generate"):
            generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider, max_retries=2)

        # No file left behind
        files = list(tmp_path.glob("narration_*.mp3"))
        assert len(files) == 0


# ---------------------------------------------------------------------------
# generate_narration_audio — non-retryable failure
# ---------------------------------------------------------------------------

class TestNonRetryableFailure:
    def test_does_not_retry_value_error(self, tmp_path: Path):
        provider = StubTTSProvider()
        provider.synthesize = MagicMock(side_effect=ValueError("bad voice"))

        with pytest.raises(TTSError, match="Failed to generate"):
            generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider, max_retries=5)

        # Should only be called once
        assert provider.synthesize.call_count == 1


# ---------------------------------------------------------------------------
# generate_narration_audio — cleanup after failure
# ---------------------------------------------------------------------------

class TestCleanupOnFailure:
    def test_cleans_up_invalid_file(self, tmp_path: Path):
        def bad_synthesize(text, output_path, voice):
            output_path.write_bytes(b"invalid")

        provider = StubTTSProvider()
        provider.synthesize = bad_synthesize

        with pytest.raises(TTSError):
            generate_narration_audio(_NARRATION, output_dir=tmp_path, provider=provider)

        files = list(tmp_path.glob("narration_*.mp3"))
        assert len(files) == 0


# ---------------------------------------------------------------------------
# GeneratedAudio model
# ---------------------------------------------------------------------------

class TestGeneratedAudioModel:
    def test_fields(self, tmp_path: Path):
        audio = GeneratedAudio(path=tmp_path / "test.mp3", duration_seconds=5.5)
        assert audio.path.name == "test.mp3"
        assert audio.duration_seconds == 5.5

    def test_is_frozen(self, tmp_path: Path):
        audio = GeneratedAudio(path=tmp_path / "x.mp3", duration_seconds=1.0)
        with pytest.raises(AttributeError):
            audio.path = tmp_path / "y.mp3"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TTSProvider protocol
# ---------------------------------------------------------------------------

class TestTTSProviderProtocol:
    def test_edge_tts_has_synthesize_interface(self):
        provider = EdgeTTSProvider()
        assert hasattr(provider, "synthesize")
        assert callable(provider.synthesize)


# ---------------------------------------------------------------------------
# EdgeTTSProvider — unit level
# ---------------------------------------------------------------------------

class TestEdgeTTSProvider:
    def test_has_synthesize_method(self):
        provider = EdgeTTSProvider()
        assert hasattr(provider, "synthesize")
        assert callable(provider.synthesize)
