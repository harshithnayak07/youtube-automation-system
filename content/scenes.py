"""Content module — scene planning from narration."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from llm.client import LLMClient, LLMError, LLMResponse
from llm.retry import _SemanticRetry, run_retryable_llm_stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scene:
    """A single visual scene for a YouTube Short."""
    scene: int
    text: str
    visual_description: str


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a visual scene planner for short-form YouTube videos (Shorts).
Given a narration, split it into a small number of visual scenes.
Target 3–5 scenes for a typical Short, but use fewer for short narrations.
Every scene must preserve the narration's original order.
Return ONLY a JSON array — no markdown, no explanation, no extra text.
Each element must have exactly three keys:
  "scene"   — integer starting at 1
  "text"    — the narration portion for this scene
  "visual_description" — 2–8 word phrase describing the visual (for image search, not a full sentence)
"""

_USER_PROMPT = """\
Split this narration into visual scenes:

{narration}
"""


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

class _ScenesRetryable:
    """Stage adapter validating a single LLM response for scene planning.

    Uses the existing ``_parse_scenes`` validation as the source of truth.
    """

    def retry(self, response: LLMResponse, prompt: str, attempt_number: int) -> list[Scene]:
        scenes = _parse_scenes(response.text)
        if not scenes:
            raise _SemanticRetry("LLM returned no valid scenes")
        return scenes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_scenes(
    llm_client: LLMClient,
    narration: str,
    *,
    max_tokens: int = 1024,
) -> list[Scene]:
    """Split narration into visual scenes for a YouTube Short.

    Parameters
    ----------
    llm_client:
        The project-wide LLM router.
    narration:
        The narration text produced by ``content.script.generate_narration``.
    max_tokens:
        Upper bound on the LLM response length.

    Returns
    -------
    list[Scene]
        Ordered list of scenes covering the full narration.

    Raises
    ------
    ValueError
        If the narration is empty or whitespace-only.
    LLMError
        If the LLM call fails after retries and fallback.
    RuntimeError
        If the LLM returns malformed or empty scene data after all
        semantic retry attempts.
    """
    if not narration.strip():
        raise ValueError("Narration must not be empty")

    user_msg = _USER_PROMPT.format(narration=narration)
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_msg}"

    logger.info("Planning scenes for narration (%d chars)", len(narration))

    scenes = run_retryable_llm_stage(
        llm_client,
        _ScenesRetryable(),
        full_prompt,
        max_tokens=max_tokens,
    )

    logger.info("Planned %d scenes", len(scenes))
    return scenes


def _parse_scenes(raw: str) -> list[Scene]:
    """Parse LLM JSON output into a list of Scene objects."""
    cleaned = _strip_artefacts(raw)

    # Try to extract a JSON array from the response
    json_str = _extract_json_array(cleaned)
    if json_str is None:
        logger.warning("No JSON array found in LLM response (first 500 chars): %.500s", raw)
        return []

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
            return []

    if not isinstance(data, list):
        logger.warning("Expected JSON array, got %s", type(data).__name__)
        return []

    scenes: list[Scene] = []
    for i, item in enumerate(data):
        scene = _validate_scene(item, index=i)
        if scene is not None:
            scenes.append(scene)

    return scenes


def _strip_artefacts(raw: str) -> str:
    """Remove markdown fences and leading/trailing noise."""
    text = raw.strip()
    # Remove ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_array(text: str) -> str | None:
    """Find the first JSON array in the text."""
    start = text.find("[")
    if start == -1:
        return None
    # Find matching closing bracket
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
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


def _validate_scene(item: dict, *, index: int) -> Scene | None:
    """Validate a single scene dict and return a Scene or None."""
    if not isinstance(item, dict):
        return None

    scene_num = item.get("scene")
    text = item.get("text", "")
    visual = item.get("visual_description", "")

    # Validate scene number
    if not isinstance(scene_num, int):
        scene_num = index + 1  # fallback to sequential

    # Validate text
    if not isinstance(text, str) or not text.strip():
        logger.warning("Scene %d has empty text, skipping", scene_num)
        return None

    # Validate visual_description
    if not isinstance(visual, str) or not visual.strip():
        logger.warning("Scene %d has empty visual_description, skipping", scene_num)
        return None

    return Scene(
        scene=scene_num,
        text=text.strip(),
        visual_description=visual.strip(),
    )
