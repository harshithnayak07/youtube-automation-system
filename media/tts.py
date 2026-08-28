"""Media module — Text-to-Speech via edge-tts (free Microsoft neural TTS)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from config import settings
from core.constants import GENERATED_AUDIO_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_VOICE = "en-US-GuyNeural"
_OUTPUT_FORMAT = "audio-24khz-96kbitrate-mono-mp3"
_MAX_RETRIES = 3
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TTSError(Exception):
    """Raised when TTS generation fails after retries."""


class TTSConfigError(TTSError):
    """Raised for missing or invalid configuration (non-retryable)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedAudio:
    """Result of a TTS generation call."""
    path: Path
    duration_seconds: float


# ---------------------------------------------------------------------------
# Provider protocol (swappable)
# ---------------------------------------------------------------------------

class TTSProvider:
    """Interface for pluggable TTS backends."""

    def synthesize(self, text: str, output_path: Path, voice: str) -> None:
        """Write audio for *text* to *output_path* using *voice*."""
        ...


# ---------------------------------------------------------------------------
# Default provider — edge-tts
# ---------------------------------------------------------------------------

class EdgeTTSProvider:
    """Microsoft Edge TTS — free, neural-quality voices."""

    def synthesize(self, text: str, output_path: Path, voice: str) -> None:
        import edge_tts

        async def _run() -> None:
            communicate = edge_tts.Communicate(text, voice, rate="+0%", pitch="+0Hz")
            await communicate.save(str(output_path))

        # Use existing event loop if available, otherwise create one
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(_run())).result()
        else:
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_audio_file(path: Path) -> bool:
    """Return True if the file exists, is non-empty, and has an audio signature."""
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if size < 100:
            return False
        # Check for common audio signatures
        with open(path, "rb") as f:
            header = f.read(12)
        return _has_audio_signature(header)
    except OSError:
        return False


def _has_audio_signature(header: bytes) -> bool:
    """Check if bytes match known audio file signatures."""
    if len(header) < 4:
        return False
    # MP3: starts with ID3 tag or frame sync (0xFF 0xFB/0xF3/0xF2)
    if header[:3] == b"ID3":
        return True
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    # WAV: starts with RIFF....WAVE
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WAVE":
        return True
    # OGG: starts with OggS
    if header[:4] == b"OggS":
        return True
    # FLAC: starts with fLaC
    if header[:4] == b"fLaC":
        return True
    return False


# ---------------------------------------------------------------------------
# Duration extraction
# ---------------------------------------------------------------------------

def _extract_mp3_duration(path: Path) -> float:
    """Estimate MP3 duration by parsing frame headers and Xing/LAME info."""
    try:
        with open(path, "rb") as f:
            data = f.read()

        # Try mutagen first (most reliable)
        duration = _duration_via_mutagen(path)
        if duration > 0:
            return duration

        # Try Xing/LAME header (more accurate)
        duration = _parse_xing_header(data)
        if duration > 0:
            return duration

        # Fall back to frame counting
        return _estimate_duration_by_frames(data)

    except Exception:
        return 0.0


def _duration_via_mutagen(path: Path) -> float:
    """Get duration via mutagen library if available."""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(str(path))
        if audio.info.length > 0:
            return round(audio.info.length, 2)
    except ImportError:
        pass
    except Exception:
        pass
    return 0.0


def _parse_xing_header(data: bytes) -> float:
    """Parse Xing/LAME header for accurate duration."""
    if len(data) < 200:
        return 0.0

    # Find Xing/Info header
    for offset in range(0, min(len(data), 400)):
        if data[offset : offset + 4] in (b"Xing", b"Info"):
            # Skip to total frames (big-endian uint32 at offset + 8 from Xing start)
            if offset + 12 <= len(data):
                total_frames = struct.unpack(">I", data[offset + 8 : offset + 12])[0]
                if total_frames > 0:
                    # MPEG1 Layer III: 1152 samples per frame at 44100 Hz
                    return round(total_frames * 1152 / 44100, 2)
            break
    return 0.0


def _estimate_duration_by_frames(data: bytes) -> float:
    """Estimate duration by counting MPEG frames."""
    if len(data) < 100:
        return 0.0

    # Bitrate lookup for MPEG1 Layer III (index = version+layer+bitrate_index)
    bitrate_table_mpeg1 = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ]
    # MPEG2/2.5 have lower bitrate ranges
    bitrate_table_mpeg2 = [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
    ]
    sample_rate_table = {
        0: [44100, 48000, 32000],  # MPEG1
        2: [22050, 24000, 16000],  # MPEG2
        3: [11025, 12000, 8000],   # MPEG2.5
    }

    frame_count = 0
    total_samples = 0
    i = 0

    # Skip ID3 tag
    if data[:3] == b"ID3":
        if len(data) >= 10:
            size = (
                (data[6] & 0x7F) << 21
                | (data[7] & 0x7F) << 14
                | (data[8] & 0x7F) << 7
                | (data[9] & 0x7F)
            )
            i = 10 + size

    while i < len(data) - 4:
        # Look for frame sync
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            # Parse header
            version = (data[i + 1] >> 3) & 0x03  # 3=MPEG1, 2=MPEG2, 0=MPEG2.5
            layer = (data[i + 1] >> 1) & 0x03  # 1=Layer III
            br_idx = (data[i + 2] >> 4) & 0x0F
            sr_idx = (data[i + 2] >> 2) & 0x03

            if version in (0, 2, 3) and layer == 1 and 0 < br_idx < 15:
                # Select appropriate bitrate table
                if version == 3:
                    br = bitrate_table_mpeg1[br_idx]
                    samples_per_frame = 1152
                else:
                    br = bitrate_table_mpeg2[br_idx]
                    samples_per_frame = 576  # MPEG2/MPEG2.5 Layer III

                sr_list = sample_rate_table.get(version, sample_rate_table[0])
                sr = sr_list[sr_idx] if sr_idx < 3 else sr_list[0]

                if br > 0 and sr > 0:
                    frame_count += 1
                    total_samples += samples_per_frame
                    frame_size = int(144 * br * 1000 / sr)
                    i += max(frame_size, 1)
                    continue
            i += 1
        else:
            i += 1

        # Safety: don't scan forever
        if frame_count > 50000:
            break

    if frame_count > 0 and total_samples > 0:
        # Use the most common sample rate for estimation
        # Default to 24000 Hz (common for TTS output)
        return round(total_samples / 24000, 2)
    return 0.0


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------

def _is_transient(exc: Exception) -> bool:
    """Return True if the exception is worth retrying."""
    return isinstance(exc, _RETRYABLE_EXCEPTIONS)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_narration_audio(
    narration: str,
    *,
    output_dir: Path | None = None,
    voice: str = _DEFAULT_VOICE,
    provider: TTSProvider | None = None,
    max_retries: int = _MAX_RETRIES,
) -> GeneratedAudio:
    """Generate narration audio from text.

    Parameters
    ----------
    narration:
        The narration text produced by ``content.script.generate_narration``.
    output_dir:
        Directory for saved audio.  Defaults to ``GENERATED_AUDIO_DIR``.
    voice:
        Voice identifier.  Defaults to ``en-US-GuyNeural``.
    provider:
        Pluggable TTS backend.  Defaults to ``EdgeTTSProvider``.
    max_retries:
        Maximum attempts for transient failures.

    Returns
    -------
    GeneratedAudio
        Path and duration of the generated audio file.

    Raises
    ------
    ValueError
        If the narration is empty or whitespace-only.
    TTSConfigError
        If the TTS provider is not available.
    TTSError
        If generation fails after retries or returns invalid data.
    """
    if not narration.strip():
        raise ValueError("Narration must not be empty")

    out_dir = output_dir or GENERATED_AUDIO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic filename based on content hash
    content_hash = hashlib.sha256(narration.encode("utf-8")).hexdigest()[:12]
    audio_path = out_dir / f"narration_{content_hash}.mp3"

    # Reuse existing valid audio
    if _is_valid_audio_file(audio_path):
        logger.info("Reusing existing audio: %s", audio_path)
        duration = _extract_mp3_duration(audio_path)
        return GeneratedAudio(path=audio_path, duration_seconds=duration)

    tts = provider or EdgeTTSProvider()

    logger.info("Generating narration audio (%d chars)", len(narration))

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            tts.synthesize(narration, audio_path, voice)

            if not _is_valid_audio_file(audio_path):
                # Clean up invalid file
                if audio_path.exists():
                    audio_path.unlink(missing_ok=True)
                raise TTSError("TTS produced invalid or empty audio file")

            duration = _extract_mp3_duration(audio_path)
            logger.info(
                "Saved narration audio: %s (%.1fs, %d bytes)",
                audio_path, duration, audio_path.stat().st_size,
            )
            return GeneratedAudio(path=audio_path, duration_seconds=duration)

        except Exception as exc:
            last_exc = exc
            # Clean up partial files
            if audio_path.exists() and not _is_valid_audio_file(audio_path):
                audio_path.unlink(missing_ok=True)

            if not _is_transient(exc):
                logger.error("TTS failed (non-retryable): %s", exc)
                break

            if attempt < max_retries:
                delay = 1.0 * (2 ** (attempt - 1))
                logger.warning(
                    "TTS attempt %d failed (transient), retrying in %.1fs: %s",
                    attempt, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.warning("TTS exhausted %d retries: %s", max_retries, exc)

    raise TTSError(f"Failed to generate narration audio: {last_exc}") from last_exc
