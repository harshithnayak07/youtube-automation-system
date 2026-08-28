"""Content module — YouTube Shorts script generation."""

from __future__ import annotations

import logging
import re

from llm.client import LLMClient, LLMError
from research.trending import Topic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a narration scriptwriter for short-form YouTube videos (Shorts).
Write ONLY the raw spoken narration text.
Do NOT include scene labels, timestamps, headings, markdown, bullet points, \
or any formatting — only the words the narrator will speak.
Keep it factual and grounded in the provided topic.
Target 80–120 words so the narration fits a 30–60 second video.
Make it engaging, concise, and suitable for a general audience.
"""

_USER_PROMPT = """\
Write a YouTube Shorts narration for the following AI topic.

Title: {title}
Source: {source}
Summary: {summary}
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_narration(
    llm_client: LLMClient,
    topic: Topic,
    *,
    max_tokens: int = 300,
) -> str:
    """Generate a spoken narration for a YouTube Shorts video.

    Parameters
    ----------
    llm_client:
        The project-wide LLM router.  Do not call providers directly.
    topic:
        A ``Topic`` produced by the research module.
    max_tokens:
        Upper bound on the LLM response length.

    Returns
    -------
    str
        Clean narration text — no labels, no markdown, no metadata.

    Raises
    ------
    ValueError
        If the topic has an empty title or summary.
    LLMError
        If the LLM call fails after retries and fallback.
    RuntimeError
        If the LLM returns empty or unusable content.
    """
    if not topic.title.strip():
        raise ValueError("Topic title must not be empty")
    if not topic.summary.strip():
        raise ValueError("Topic summary must not be empty")

    user_msg = _USER_PROMPT.format(
        title=topic.title,
        source=topic.source,
        summary=topic.summary,
    )

    full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_msg}"

    logger.info("Generating narration for topic: %s", topic.title)

    response = llm_client.complete(full_prompt, max_tokens=max_tokens)
    narration = _clean_narration(response.text)

    if not narration:
        raise RuntimeError("LLM returned empty narration content")

    word_count = len(narration.split())
    logger.info("Generated narration: %d words", word_count)

    return narration


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_narration(raw: str) -> str:
    """Strip common LLM artefacts and normalise whitespace."""
    text = raw.strip()

    # Remove markdown-style headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove scene labels / timestamps like [Scene 1], (0:05), etc.
    text = re.sub(r"^\[.*?\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\(.*?\)\s*", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers (pair-aware)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?<!\w)\*(?!\*)", "", text)  # opening single *
    text = re.sub(r"(?<!\*)\*(?!\w)", "", text)  # closing single *
    text = re.sub(r"(?<!\w)_(?!_)", "", text)    # opening single _
    text = re.sub(r"(?!_)_(?!\w)", "", text)     # closing single _

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
