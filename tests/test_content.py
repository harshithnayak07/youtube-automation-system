"""Tests for the content.script module — all use mocks, no real API calls."""

from __future__ import annotations

import pytest

from content.script import generate_narration, _clean_narration
from llm.client import LLMClient, LLMResponse, LLMError
from research.trending import Topic


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_topic(
    title: str = "OpenAI Announces GPT-5",
    url: str = "https://example.com/gpt5",
    summary: str = "OpenAI released GPT-5 with major reasoning improvements.",
    source: str = "TechCrunch",
) -> Topic:
    return Topic(title=title, url=url, summary=summary, source=source)


class StubLLMClient:
    """Stub that returns pre-configured text from ``complete``."""

    def __init__(self, text: str):
        self._text = text
        self._last_prompt: str | None = None

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self._last_prompt = prompt
        return LLMResponse(text=self._text, provider="stub", model="stub-model")


class FailingLLMClient:
    """Stub that always raises LLMError."""

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        raise LLMError("simulated provider failure")


# ---------------------------------------------------------------------------
# generate_narration — success
# ---------------------------------------------------------------------------

class TestGenerateNarrationSuccess:
    def test_returns_clean_text(self):
        narration = (
            "OpenAI just announced GPT-5, their most advanced model yet. "
            "It features significant improvements in reasoning and coding. "
            "This release is expected to reshape how developers build AI applications."
        )
        client = StubLLMClient(narration)
        topic = _make_topic()

        result = generate_narration(client, topic)

        assert result == narration
        assert isinstance(result, str)

    def test_prompt_contains_topic_info(self):
        client = StubLLMClient("Some narration.")
        topic = _make_topic(title="DeepMind's New Protein Model")

        generate_narration(client, topic)

        assert client._last_prompt is not None
        assert "DeepMind's New Protein Model" in client._last_prompt
        assert topic.source in client._last_prompt
        assert topic.summary in client._last_prompt

    def test_strips_markdown_headings(self):
        client = StubLLMClient("## Heading\nActual narration text here.")
        result = generate_narration(client, _make_topic())

        assert "##" not in result
        assert "Heading" in result  # heading text is kept, only syntax removed
        assert "Actual narration text here" in result

    def test_strips_scene_labels(self):
        client = StubLLMClient("[Scene 1] Welcome to the show.\n[Scene 2] Let's dive in.")
        result = generate_narration(client, _make_topic())

        assert "[Scene 1]" not in result
        assert "[Scene 2]" not in result
        assert "Welcome to the show" in result

    def test_strips_timestamps(self):
        client = StubLLMClient("(0:05) Breaking news today.\n(0:15) More details follow.")
        result = generate_narration(client, _make_topic())

        assert "(0:05)" not in result
        assert "(0:15)" not in result

    def test_strips_bold_markers(self):
        client = StubLLMClient("This is **important** news for __everyone__.")
        result = generate_narration(client, _make_topic())

        assert "**" not in result
        assert "__" not in result
        assert "important" in result

    def test_collapses_whitespace(self):
        client = StubLLMClient("Too   many    spaces   here.")
        result = generate_narration(client, _make_topic())

        assert "   " not in result
        assert "  " not in result


# ---------------------------------------------------------------------------
# generate_narration — empty / invalid LLM output
# ---------------------------------------------------------------------------

class TestGenerateNarrationEmpty:
    def test_raises_on_empty_response(self):
        client = StubLLMClient("")
        with pytest.raises(RuntimeError, match="empty narration"):
            generate_narration(client, _make_topic())

    def test_raises_on_whitespace_only_response(self):
        client = StubLLMClient("   \n  \n  ")
        with pytest.raises(RuntimeError, match="empty narration"):
            generate_narration(client, _make_topic())

    def test_raises_on_only_artefacts(self):
        # Input that strips to empty: headings without text, empty scene labels, etc.
        client = StubLLMClient("##\n[Scene 1]\n**__**\n___")
        with pytest.raises(RuntimeError, match="empty narration"):
            generate_narration(client, _make_topic())


# ---------------------------------------------------------------------------
# generate_narration — LLM failure propagation
# ---------------------------------------------------------------------------

class TestGenerateNarrationFailure:
    def test_propagates_llm_error(self):
        client = FailingLLMClient()
        with pytest.raises(LLMError, match="simulated provider failure"):
            generate_narration(client, _make_topic())


# ---------------------------------------------------------------------------
# generate_narration — input validation
# ---------------------------------------------------------------------------

class TestGenerateNarrationValidation:
    def test_rejects_empty_title(self):
        client = StubLLMClient("ok")
        topic = _make_topic(title="")
        with pytest.raises(ValueError, match="title must not be empty"):
            generate_narration(client, topic)

    def test_rejects_whitespace_title(self):
        client = StubLLMClient("ok")
        topic = _make_topic(title="   ")
        with pytest.raises(ValueError, match="title must not be empty"):
            generate_narration(client, topic)

    def test_rejects_empty_summary(self):
        client = StubLLMClient("ok")
        topic = _make_topic(summary="")
        with pytest.raises(ValueError, match="summary must not be empty"):
            generate_narration(client, topic)

    def test_rejects_whitespace_summary(self):
        client = StubLLMClient("ok")
        topic = _make_topic(summary="  \n  ")
        with pytest.raises(ValueError, match="summary must not be empty"):
            generate_narration(client, topic)


# ---------------------------------------------------------------------------
# _clean_narration — unit tests
# ---------------------------------------------------------------------------

class TestCleanNarration:
    def test_strips_heading(self):
        assert _clean_narration("## Hello") == "Hello"

    def test_strips_multiple_headings(self):
        result = _clean_narration("## Title\n### Subtitle\nContent")
        assert "##" not in result
        assert "###" not in result
        assert "Content" in result

    def test_strips_scene_label(self):
        result = _clean_narration("[Scene 1] Hello world")
        assert "Scene" not in result
        assert "Hello world" in result

    def test_strips_parenthetical_timestamp(self):
        result = _clean_narration("(0:05) Hello")
        assert "0:05" not in result
        assert "Hello" in result

    def test_strips_bold(self):
        assert _clean_narration("**bold**") == "bold"

    def test_strips_italic_single_star(self):
        assert _clean_narration("*italic*") == "italic"

    def test_empty_input(self):
        assert _clean_narration("") == ""

    def test_whitespace_normalisation(self):
        assert _clean_narration("  hello   world  ") == "hello world"
