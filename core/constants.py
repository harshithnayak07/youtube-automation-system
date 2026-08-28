"""Project-wide constants and path definitions."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Core directories
JOBS_DIR: Path = PROJECT_ROOT / "jobs"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# Module-level data directories
RESEARCH_CACHE_DIR: Path = OUTPUT_DIR / "research_cache"
GENERATED_SCRIPTS_DIR: Path = OUTPUT_DIR / "scripts"
GENERATED_IMAGES_DIR: Path = OUTPUT_DIR / "images"
GENERATED_AUDIO_DIR: Path = OUTPUT_DIR / "audio"
FINAL_VIDEOS_DIR: Path = OUTPUT_DIR / "videos"

# Database
DATABASE_PATH: Path = PROJECT_ROOT / "storage" / "state.db"
