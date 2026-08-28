"""End-to-end local test — runs pipeline up to QA stage only."""

import sys
import os
import time
import uuid
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOG_LEVEL", "INFO")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_test")


def main():
    job_id = f"e2e-{uuid.uuid4().hex[:8]}"
    logger.info("=" * 60)
    logger.info("END-TO-END LOCAL TEST — Job: %s", job_id)
    logger.info("=" * 60)

    timings = {}
    results = {}

    # ── Stage 1: Research ──────────────────────────────────────────
    logger.info("")
    logger.info("▶ Stage 1/9: Research trending AI topic")
    t0 = time.time()
    try:
        from research.trending import find_trending_ai_topic
        topic = find_trending_ai_topic()
        elapsed = time.time() - t0
        timings["research"] = elapsed
        results["topic"] = topic
        logger.info("  ✓ Topic: %s", topic.title)
        logger.info("  ✓ Source: %s", topic.source)
        logger.info("  ✓ URL: %s", topic.url)
        logger.info("  ✓ Summary: %.80s...", topic.summary)
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ RESEARCH FAILED: %s [%s]", e, type(e).__name__)
        return

    # ── Stage 2: LLM client + Narration ────────────────────────────
    logger.info("")
    logger.info("▶ Stage 2/9: Generate narration")
    t0 = time.time()
    try:
        from llm.client import LLMClient, GroqProvider, OpenRouterProvider
        from config import settings

        primary = None
        fallback = None
        if settings.GROQ_API_KEY:
            primary = GroqProvider(settings.GROQ_API_KEY, default_model="groq/compound")
        if settings.OPENROUTER_API_KEY:
            fallback = OpenRouterProvider(settings.OPENROUTER_API_KEY, default_model="meta-llama/llama-3.3-70b-instruct")

        llm_client = LLMClient(
            primary=primary or fallback,
            fallback=fallback if primary else None,
        )
        logger.info("  ✓ LLM client ready (primary=%s, fallback=%s)",
                     primary.name if primary else "none",
                     fallback.name if fallback else "none")

        from content.script import generate_narration
        narration = generate_narration(llm_client, topic)
        elapsed = time.time() - t0
        timings["script"] = elapsed
        results["narration"] = narration
        word_count = len(narration.split())
        logger.info("  ✓ Narration: %d words, %d chars", word_count, len(narration))
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ SCRIPT FAILED: %s [%s]", e, type(e).__name__)
        import traceback; traceback.print_exc()
        return

    # ── Stage 3: Scenes ────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Stage 3/9: Plan scenes")
    t0 = time.time()
    try:
        from content.scenes import plan_scenes
        scenes = plan_scenes(llm_client, narration)
        elapsed = time.time() - t0
        timings["scenes"] = elapsed
        results["scenes"] = scenes
        logger.info("  ✓ Scenes: %d", len(scenes))
        for s in scenes:
            logger.info("    Scene %d: %.60s...", s.scene, s.text)
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ SCENES FAILED: %s [%s]", e, type(e).__name__)
        import traceback; traceback.print_exc()
        return

    # ── Stage 4: Images ────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Stage 4/9: Generate scene images")
    t0 = time.time()
    try:
        from media.image import generate_all_scene_images
        images = generate_all_scene_images(scenes, job_id=job_id)
        elapsed = time.time() - t0
        timings["images"] = elapsed
        results["images"] = images
        logger.info("  ✓ Images: %d generated", len(images))
        for img in images:
            fpath = img.path
            fsize = fpath.stat().st_size if fpath.exists() else 0
            logger.info("    Scene %d: %s (%d bytes)", img.scene, fpath, fsize)
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ IMAGES FAILED: %s [%s]", e, type(e).__name__)
        import traceback; traceback.print_exc()
        return

    # ── Stage 5: TTS ───────────────────────────────────────────────
    logger.info("")
    logger.info("▶ Stage 5/9: Generate narration audio (Edge TTS)")
    t0 = time.time()
    try:
        from media.tts import generate_narration_audio
        audio = generate_narration_audio(narration)
        elapsed = time.time() - t0
        timings["tts"] = elapsed
        results["audio"] = audio
        fsize = audio.path.stat().st_size if audio.path.exists() else 0
        logger.info("  ✓ Audio: %s", audio.path)
        logger.info("  ✓ Duration: %.1fs, Size: %d bytes", audio.duration_seconds, fsize)
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ TTS FAILED: %s [%s]", e, type(e).__name__)
        import traceback; traceback.print_exc()
        return

    # ── Stage 6: Video composition ─────────────────────────────────
    logger.info("")
    logger.info("▶ Stage 6/9: Compose 9:16 video")
    t0 = time.time()
    try:
        from video.compositor import compose_video
        video = compose_video(scenes, images, audio)
        elapsed = time.time() - t0
        timings["composition"] = elapsed
        results["video"] = video
        fsize = video.path.stat().st_size if video.path.exists() else 0
        logger.info("  ✓ Video: %s", video.path)
        logger.info("  ✓ Duration: %.1fs, Scenes: %d", video.duration_seconds, video.scene_count)
        logger.info("  ✓ Resolution: 1080x1920, Size: %d bytes (%.1f MB)", fsize, fsize / (1024*1024))
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ COMPOSITION FAILED: %s [%s]", e, type(e).__name__)
        import traceback; traceback.print_exc()
        return

    # ── Stage 7: Metadata ──────────────────────────────────────────
    logger.info("")
    logger.info("▶ Stage 7/9: Generate metadata")
    t0 = time.time()
    try:
        from content.metadata import generate_metadata
        metadata = generate_metadata(llm_client, topic, narration)
        elapsed = time.time() - t0
        timings["metadata"] = elapsed
        results["metadata"] = metadata
        logger.info("  ✓ Title: %s", metadata.title)
        logger.info("  ✓ Description: %.80s...", metadata.description)
        logger.info("  ✓ Tags: %s", ", ".join(metadata.tags))
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ METADATA FAILED: %s [%s]", e, type(e).__name__)
        import traceback; traceback.print_exc()
        return

    # ── Stage 8: Thumbnail ─────────────────────────────────────────
    logger.info("")
    logger.info("▶ Stage 8/9: Generate thumbnail")
    t0 = time.time()
    try:
        from media.thumbnail import generate_thumbnail
        thumbnail = generate_thumbnail(topic.title, images, job_id=job_id)
        elapsed = time.time() - t0
        timings["thumbnail"] = elapsed
        results["thumbnail"] = thumbnail
        fsize = thumbnail.path.stat().st_size if thumbnail.path.exists() else 0
        logger.info("  ✓ Thumbnail: %s", thumbnail.path)
        logger.info("  ✓ Dimensions: %dx%d, Size: %d bytes", thumbnail.width, thumbnail.height, fsize)
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ THUMBNAIL FAILED: %s [%s]", e, type(e).__name__)
        import traceback; traceback.print_exc()
        return

    # ── Stage 9: QA gate ───────────────────────────────────────────
    logger.info("")
    logger.info("▶ Stage 9/9: QA gate")
    t0 = time.time()
    try:
        from core.qa import validate_job
        qa = validate_job(
            topic=topic,
            narration=narration,
            scenes=scenes,
            images=images,
            audio=audio,
            video=video,
            metadata=metadata,
            thumbnail=thumbnail,
        )
        elapsed = time.time() - t0
        timings["qa"] = elapsed
        results["qa"] = qa
        logger.info("  ✓ QA PASSED: %s", qa.passed)
        if qa.warnings:
            logger.info("  ⚠ Warnings (%d):", len(qa.warnings))
            for w in qa.warnings:
                logger.info("    - %s", w)
        if qa.errors:
            logger.info("  ✗ Errors (%d):", len(qa.errors))
            for e in qa.errors:
                logger.info("    - %s", e)
        logger.info("  ✓ Time: %.1fs", elapsed)
    except Exception as e:
        logger.error("  ✗ QA FAILED: %s [%s]", e, type(e).__name__)
        import traceback; traceback.print_exc()
        return

    # ── Summary ────────────────────────────────────────────────────
    total_time = sum(timings.values())
    logger.info("")
    logger.info("=" * 60)
    logger.info("END-TO-END TEST COMPLETE — Job: %s", job_id)
    logger.info("=" * 60)
    logger.info("")
    logger.info("TOPIC:       %s", topic.title)
    logger.info("SOURCE:      %s", topic.source)
    logger.info("NARRATION:   %d words", len(narration.split()))
    logger.info("SCENES:      %d", len(scenes))
    logger.info("IMAGES:      %d", len(images))
    for img in images:
        fpath = img.path
        fsize = fpath.stat().st_size if fpath.exists() else 0
        logger.info("  Scene %d: %s (%d bytes)", img.scene, fpath, fsize)
    logger.info("AUDIO:       %s (%.1fs)", audio.path, audio.duration_seconds)
    logger.info("VIDEO:       %s (%.1fs, %.1f MB)", video.path, video.duration_seconds, video.path.stat().st_size / (1024*1024))
    logger.info("METADATA:    %s", metadata.title)
    logger.info("THUMBNAIL:   %s (%dx%d)", thumbnail.path, thumbnail.width, thumbnail.height)
    logger.info("QA PASSED:   %s", qa.passed)
    if qa.warnings:
        logger.info("WARNINGS:    %s", "; ".join(qa.warnings))
    logger.info("")
    logger.info("TIMING BREAKDOWN:")
    for stage, t in timings.items():
        logger.info("  %-15s %.1fs", stage, t)
    logger.info("  %-15s %.1fs", "TOTAL", total_time)
    logger.info("")
    logger.info("All artifacts in: %s/output/", PROJECT_ROOT)


if __name__ == "__main__":
    main()
