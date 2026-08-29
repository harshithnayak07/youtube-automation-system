"""Centralized configuration loaded from environment variables."""

from pathlib import Path
from dotenv import load_dotenv
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = _PROJECT_ROOT
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# LLM — Groq (primary)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# LLM — OpenRouter (fallback)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------
YOUTUBE_CLIENT_ID: str = os.getenv("YOUTUBE_CLIENT_ID", "")
YOUTUBE_CLIENT_SECRET: str = os.getenv("YOUTUBE_CLIENT_SECRET", "")
YOUTUBE_REFRESH_TOKEN: str = os.getenv("YOUTUBE_REFRESH_TOKEN", "")
YOUTUBE_VISIBILITY: str = os.getenv("YOUTUBE_VISIBILITY", "private")

# ---------------------------------------------------------------------------
# Media / Image Generation (Pexels)
# ---------------------------------------------------------------------------
PEXELS_API_KEY: str = os.getenv("PEXELS_API_KEY", "")

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "")
TTS_API_KEY: str = os.getenv("TTS_API_KEY", "")

# ---------------------------------------------------------------------------
# Email Notifications
# ---------------------------------------------------------------------------
SMTP_HOST: str = os.getenv("SMTP_HOST", "")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
NOTIFICATION_EMAIL_TO: str = os.getenv("NOTIFICATION_EMAIL_TO", "")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DATABASE_PATH: str = os.getenv("DATABASE_PATH") or str(PROJECT_ROOT / "storage" / "state.db")

# ---------------------------------------------------------------------------
# GitHub Actions
# ---------------------------------------------------------------------------
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
