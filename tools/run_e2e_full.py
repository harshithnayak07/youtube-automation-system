"""Full end-to-end pipeline test: research → private YouTube upload."""

import sys
import time
import logging
import hashlib
import re
import json
import subprocess
import uuid
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("e2e_full")

OUTPUT = Path(r"C:\Users\harsh\Youtube_Automation_system\output")
VIDEO_DIR = OUTPUT / "videos"
THUMB_PATH = OUTPUT / "thumbnails" / "thumbnail.jpg"

job_id = f"e2e-{uuid.uuid4().hex[:8]}"

t_start = time.time()


def elapsed():
    return f"{time.time() - t_start:.1f}s"


# ── Stage 1: Research ────────────────────────────────────────────────────────
log.info("▶ Stage 1/10: Research")
from research.trending import find_trending_ai_topic

topic = find_trending_ai_topic()
log.info("  Topic: %s", topic.title[:100])
log.info("  Source: %s", topic.source)
log.info("  ✓ Research done (%s)", elapsed())


# ── Stage 2: Script ──────────────────────────────────────────────────────────
log.info("▶ Stage 2/10: Script generation")
from llm.client import LLMClient, GroqProvider, OpenRouterProvider, LLMConfigError
from content.script import generate_narration
from config import settings

primary = None
fallback = None
if settings.GROQ_API_KEY:
    try:
        primary = GroqProvider(settings.GROQ_API_KEY, default_model=settings.GROQ_MODEL)
    except LLMConfigError:
        pass
if settings.OPENROUTER_API_KEY:
    try:
        fallback = OpenRouterProvider(settings.OPENROUTER_API_KEY, default_model=settings.OPENROUTER_MODEL)
    except LLMConfigError:
        pass
llm = LLMClient(primary=primary or fallback, fallback=fallback if primary else None)
narration = generate_narration(llm, topic)
word_count = len(narration.split())
log.info("  ✓ Narration: %d words, %d chars (%s)", word_count, len(narration), elapsed())


# ── Stage 3: Scenes ──────────────────────────────────────────────────────────
log.info("▶ Stage 3/10: Scene generation")
from content.scenes import plan_scenes

scenes = plan_scenes(llm, narration)
log.info("  ✓ Scenes: %d (%s)", len(scenes), elapsed())
for i, s in enumerate(scenes, 1):
    log.info("    Scene %d: %s", i, s.text[:60])


# ── Stage 4: Images (Pexels) ─────────────────────────────────────────────────
log.info("▶ Stage 4/10: Pexels image generation")
from media.image import generate_all_scene_images

images = generate_all_scene_images(scenes, job_id=job_id)
log.info("  ✓ Images: %d (%s)", len(images), elapsed())
for i, img in enumerate(images, 1):
    log.info("    Scene %d: %s (%d bytes)", i, img.path.name, img.path.stat().st_size)


# ── Stage 5: TTS ─────────────────────────────────────────────────────────────
log.info("▶ Stage 5/10: Edge TTS")
from media.tts import generate_narration_audio

audio = generate_narration_audio(narration)
log.info("  ✓ Audio: %.1fs, %d bytes (%s)", audio.duration_seconds, audio.path.stat().st_size, elapsed())


# ── Stage 6: Video composition (with captions) ───────────────────────────────
log.info("▶ Stage 6/10: Video composition + captions")
from video.compositor import compose_video

composed = compose_video(
    scenes=scenes,
    images=images,
    audio=audio,
    output_dir=VIDEO_DIR,
)
video_path = composed.path
log.info("  ✓ Video: %.1fs, %d bytes (%s)", composed.duration_seconds, video_path.stat().st_size, elapsed())


# ── Stage 7: Metadata ────────────────────────────────────────────────────────
log.info("▶ Stage 7/10: Metadata generation")
from content.metadata import generate_metadata

meta = generate_metadata(llm, topic, narration)
hashtags = re.findall(r"#[a-zA-Z0-9_]+", meta.description)
log.info("  ✓ Title: %s", meta.title)
log.info("  ✓ Tags: %s", meta.tags)
log.info("  ✓ Hashtags: %s", hashtags)
log.info("  ✓ Description end: ...%s", meta.description[-120:])
log.info("  ✓ Metadata done (%s)", elapsed())


# ── Stage 8: Thumbnail ───────────────────────────────────────────────────────
log.info("▶ Stage 8/10: Thumbnail generation")
from media.thumbnail import generate_thumbnail

thumb = generate_thumbnail(
    topic_title=topic.title,
    scene_images=images,
    job_id=job_id,
)
log.info("  ✓ Thumbnail: %s, %dx%d (%s)", thumb.path, thumb.width, thumb.height, elapsed())


# ── Stage 9: QA ──────────────────────────────────────────────────────────────
log.info("▶ Stage 9/10: QA gate")
from core.qa import validate_job
from video.compositor import ComposedVideo

video_obj = ComposedVideo(
    path=video_path,
    duration_seconds=composed.duration_seconds,
    scene_count=composed.scene_count,
)

qa = validate_job(
    topic=topic,
    narration=narration,
    scenes=scenes,
    images=images,
    audio=audio,
    video=video_obj,
    metadata=meta,
    thumbnail=thumb,
)
log.info("  ✓ QA passed: %s, warnings: %d, errors: %d (%s)", qa.passed, len(qa.warnings), len(qa.errors), elapsed())
for w in qa.warnings:
    log.info("    ⚠ %s", w)
for e in qa.errors:
    log.info("    ✗ %s", e)


# ── Stage 10: YouTube upload (PRIVATE) ───────────────────────────────────────
log.info("▶ Stage 10/10: YouTube upload (PRIVATE)")
from youtube.uploader import upload_video

upload_result = upload_video(
    video_path=video_path,
    thumbnail_path=THUMB_PATH,
    video_metadata=meta,
    qa_result=qa,
)
log.info("  ✓ Upload done (%s)", elapsed())


# ── Verification ─────────────────────────────────────────────────────────────
log.info("")
log.info("=" * 70)
log.info("VERIFICATION REPORT")
log.info("=" * 70)

# Video probe
r = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries",
     "stream=codec_type,codec_name,width,height,sample_rate,channels,duration",
     "-of", "json", str(video_path)],
    capture_output=True, text=True, timeout=30,
)
probe = json.loads(r.stdout)
v_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), {})
a_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), {})

# Audio volume
r2 = subprocess.run(
    ["ffmpeg", "-i", str(video_path), "-af", "volumedetect", "-f", "null", "-"],
    capture_output=True, text=True, timeout=30,
)
vol_lines = [l.strip() for l in r2.stderr.split("\n") if "mean_volume" in l or "max_volume" in l]

# Thumbnail hash
local_thumb_hash = hashlib.md5(THUMB_PATH.read_bytes()).hexdigest()

# All hashtags from description
all_hashtags = re.findall(r"#[a-zA-Z0-9_]+", meta.description)

report = f"""
{'='*70}
JOB REPORT — e2e-full-{int(t_start)}
{'='*70}

TOPIC:              {topic.title[:100]}
SOURCE:             {topic.source}

SCRIPT:             {word_count} words, {len(narration)} chars

SCENES:             {len(scenes)}
IMAGES:             {len(images)} Pexels images
TTS DURATION:       {audio.duration_seconds:.1f}s

VIDEO FILE:         {video_path}
RESOLUTION:         {v_stream.get('width')}x{v_stream.get('height')}
ASPECT RATIO:       9:16 ({"PASS" if v_stream.get('width') == 1080 and v_stream.get('height') == 1920 else "FAIL"})
DURATION:           {v_stream.get('duration')}s
VIDEO CODEC:        {v_stream.get('codec_name')}
AUDIO CODEC:        {a_stream.get('codec_name')} ({a_stream.get('sample_rate')}Hz, {a_stream.get('channels')}ch)
AUDIO PRESENT:      {"YES" if a_stream else "NO"}
AUDIO VOLUME:       {'; '.join(vol_lines)}

CAPTIONS:           Overlay via TextClip (scene.text on each image)

THUMBNAIL:          {THUMB_PATH}
THUMB DIMS:         {thumb.width}x{thumb.height}
THUMB MD5:          {local_thumb_hash}

METADATA TITLE:     {meta.title}
METADATA TAGS:      {meta.tags}
HASHTAGS:           {all_hashtags}
DESCRIPTION:        {meta.description}

QA PASSED:          {qa.passed}
QA WARNINGS:        {qa.warnings}
QA ERRORS:          {qa.errors}

YOUTUBE UPLOAD:     {upload_result.video_status}
VIDEO ID:           {upload_result.video_id}
YOUTUBE URL:        {upload_result.video_url}
PRIVACY:            PRIVATE
THUMB UPLOAD:       {upload_result.thumbnail_status}

TOTAL TIME:         {elapsed()}
{'='*70}
"""
print(report)
log.info(report)
log.info("DONE")
