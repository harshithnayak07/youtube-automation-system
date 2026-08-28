"""Content module — YouTube metadata generation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from llm.client import LLMClient, LLMError
from research.trending import Topic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoMetadata:
    """Structured YouTube video metadata."""
    title: str
    description: str
    tags: list[str]


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a YouTube metadata specialist for AI/technology Shorts.
Generate metadata for a YouTube Short based on the provided topic and narration.
Return ONLY a JSON object with exactly three keys:
  "title"       — concise, accurate, curiosity-driven (max 100 chars), no misleading clickbait
  "description" — factual, relevant to the video content (2–4 sentences), ending with 3–5 space-separated #hashtags on a new line
  "tags"        — array of 5–10 relevant lowercase tags (no # prefix)
Do NOT include any extra text, explanation, or markdown — only the JSON.
"""

_USER_PROMPT = """\
Topic title: {topic_title}
Topic source: {topic_source}
Topic summary: {topic_summary}

Narration excerpt (first 200 chars):
{narration_excerpt}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_metadata(
    llm_client: LLMClient,
    topic: Topic,
    narration: str,
    *,
    max_tokens: int = 512,
) -> VideoMetadata:
    """Generate YouTube metadata from a topic and narration.

    Parameters
    ----------
    llm_client:
        The project-wide LLM router.
    topic:
        A ``Topic`` produced by the research module.
    narration:
        The narration text produced by ``content.script.generate_narration``.
    max_tokens:
        Upper bound on the LLM response length.

    Returns
    -------
    VideoMetadata
        Validated title, description, and tags.

    Raises
    ------
    ValueError
        If topic or narration is empty.
    LLMError
        If the LLM call fails after retries and fallback.
    RuntimeError
        If the LLM returns malformed or empty metadata.
    """
    if not topic.title.strip():
        raise ValueError("Topic title must not be empty")
    if not narration.strip():
        raise ValueError("Narration must not be empty")

    narration_excerpt = narration[:200].strip()
    user_msg = _USER_PROMPT.format(
        topic_title=topic.title,
        topic_source=topic.source,
        topic_summary=topic.summary,
        narration_excerpt=narration_excerpt,
    )
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_msg}"

    logger.info("Generating metadata for topic: %s", topic.title)

    response = llm_client.complete(full_prompt, max_tokens=max_tokens)
    metadata = _parse_metadata(response.text)

    if metadata is None:
        raise RuntimeError("LLM returned malformed or empty metadata")

    logger.info("Generated metadata: title=%r, tags=%d", metadata.title, len(metadata.tags))
    return metadata


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

def _parse_metadata(raw: str) -> VideoMetadata | None:
    """Parse LLM JSON output into a VideoMetadata object."""
    cleaned = _strip_artefacts(raw)
    json_str = _extract_json_object(cleaned)

    if json_str is None:
        logger.warning("No JSON object found in LLM response (first 500 chars): %.500s", raw)
        return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s — attempting single-quote normalization", exc)
        logger.debug("Raw JSON string (first 500 chars): %.500s", json_str)
        try:
            data = json.loads(_normalize_json_quotes(json_str))
        except json.JSONDecodeError as exc2:
            logger.warning(
                "JSON parse also failed after normalization: %s (raw response first 500 chars): %.500s",
                exc2, raw,
            )
            return None

    if not isinstance(data, dict):
        logger.warning("Expected JSON object, got %s", type(data).__name__)
        return None

    return _validate_metadata(data)


def _strip_artefacts(raw: str) -> str:
    """Remove markdown fences and leading/trailing noise."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str | None:
    """Find the first JSON object in the text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_json_quotes(s: str) -> str:
    """Convert single-quote delimited JSON to standard double-quote JSON.

    Uses a state machine that tracks whether we are inside a string and which
    quote character opened it.  Apostrophes inside already-valid double-quoted
    strings (e.g. ``"AI's"``) are left untouched.

    Only called when ``json.loads()`` has already failed, so it is safe to
    apply aggressively — if the result is still invalid, the caller will
    report the original parse error.
    """
    OUTSIDE = 0
    IN_DOUBLE = 1
    IN_SINGLE = 2

    state = OUTSIDE
    escape_next = False
    out: list[str] = []

    for ch in s:
        # --- Handle backslash escapes inside strings ---
        if escape_next:
            out.append(ch)
            escape_next = False
            continue

        if state != OUTSIDE and ch == "\\":
            out.append(ch)
            escape_next = True
            continue

        # --- State transitions ---
        if state == OUTSIDE:
            if ch == '"':
                state = IN_DOUBLE
                out.append(ch)
            elif ch == "'":
                # Single-quote string delimiter → convert to double-quote
                state = IN_SINGLE
                out.append('"')
            else:
                out.append(ch)

        elif state == IN_DOUBLE:
            if ch == '"':
                state = OUTSIDE
            out.append(ch)

        elif state == IN_SINGLE:
            if ch == "'":
                state = OUTSIDE
                out.append('"')  # convert closing delimiter
            else:
                out.append(ch)

    return "".join(out)


def _validate_metadata(data: dict) -> VideoMetadata | None:
    """Validate a metadata dict and return VideoMetadata or None."""
    title = data.get("title", "")
    description = data.get("description", "")
    tags = data.get("tags", [])

    # Validate title
    if not isinstance(title, str) or not title.strip():
        logger.warning("Metadata has empty title")
        return None
    title = title.strip()[:100]  # Enforce max length

    # Validate description
    if not isinstance(description, str) or not description.strip():
        logger.warning("Metadata has empty description")
        return None
    description = _normalize_description_hashtags(description.strip())

    # Validate tags
    if not isinstance(tags, list) or len(tags) == 0:
        logger.warning("Metadata has no tags")
        return None
    clean_tags = [
        str(t).strip().lower().lstrip("#")
        for t in tags
        if isinstance(t, str) and t.strip()
    ]
    if not clean_tags:
        logger.warning("Metadata has no valid tags")
        return None

    return VideoMetadata(
        title=title,
        description=description,
        tags=clean_tags[:10],  # Cap at 10 tags
    )


# ---------------------------------------------------------------------------
# Hashtag helpers
# ---------------------------------------------------------------------------

_DEFAULT_HASHTAGS = ["#ai", "#technology", "#ainews"]

_HASHTAG_RE = re.compile(r"(?:^|\s)(#[a-zA-Z0-9_]+)")


def _normalize_description_hashtags(description: str) -> str:
    """Ensure the description ends with 3–5 lowercase space-separated hashtags.

    - Extracts any existing hashtags from the description text.
    - Appends defaults if fewer than 3 are present.
    - Deduplicates and caps at 5.
    - Hashtags are appended after a blank line separator.
    """
    existing = _HASHTAG_RE.findall(description)
    existing = [h.lower() for h in existing]

    # Strip hashtags from the body so we can re-append them cleanly
    body = _HASHTAG_RE.sub("", description).strip()

    # Merge with defaults, preserving order, deduplicating
    seen: set[str] = set()
    merged: list[str] = []
    for tag in existing + _DEFAULT_HASHTAGS:
        if tag not in seen:
            seen.add(tag)
            merged.append(tag)

    # Cap at 5
    hashtags = merged[:5]

    return f"{body}\n\n{' '.join(hashtags)}"
