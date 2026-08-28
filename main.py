"""YouTube Automation System - Main Entry Point."""

import sys
import logging
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from core.constants import LOG_DIR, JOBS_DIR, OUTPUT_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("YouTube Automation System starting up")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Jobs directory: %s", JOBS_DIR)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Log level: %s", settings.LOG_LEVEL)

    from core.pipeline import run_pipeline

    try:
        result = run_pipeline()
    except Exception:
        logger.exception("Pipeline crashed with an unexpected error")
        return 1

    if result.success:
        logger.info(
            "Pipeline succeeded — job=%s topic=%s video=%s",
            result.job_id,
            result.topic.title if result.topic else "?",
            result.upload_result.video_id if result.upload_result else "no-upload",
        )
        return 0
    else:
        logger.error(
            "Pipeline failed — job=%s stage=%s error=%s",
            result.job_id,
            result.stage.value,
            result.error,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
