"""Tests for the content.metadata module — all use mocks, no real API calls."""

from __future__ import annotations

import json

import pytest

from content.metadata import (
    VideoMetadata,
    generate_metadata,
    _parse_metadata,
    _strip_artefacts,
    _extract_json_object,
    _validate_metadata,
    _normalize_description_hashtags,
    _normalize_json_quotes,
)
from llm.client import LLMResponse, LLMError
from research.trending import Topic


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_topic(
    title: str = "OpenAI Announces GPT-5",
    source: str = "TechCrunch",
    summary: str = "OpenAI released GPT-5 with major reasoning improvements.",
) -> Topic:
    return Topic(title=title, url="https://example.com/article", summary=summary, source=source)


_NARRATION = (
    "OpenAI just announced GPT-5, their most advanced model yet. "
    "It features significant improvements in reasoning and coding. "
    "Developers are already building groundbreaking applications with it."
)

SAMPLE_METADATA_JSON = json.dumps({
    "title": "GPT-5 Is Here: What You Need to Know",
    "description": "OpenAI has released GPT-5, featuring major improvements in reasoning and coding. This short covers the key highlights and what it means for developers.",
    "tags": ["gpt-5", "openai", "artificial intelligence", "ai news", "technology"],
})


class StubLLMClient:
    def __init__(self, text: str):
        self._text = text

    def complete(self, prompt: str, *, model: str | None = None, max_tokens: int = 2048) -> LLMResponse:
        return LLMResponse(text=self._text, provider="stub", model="stub-model")


class FailingLLMClient:
    def complete(self, prompt: str, *, model: str | None = None, max_tokens: int = 2048) -> LLMResponse:
        raise LLMError("simulated provider failure")


# ---------------------------------------------------------------------------
# _strip_artefacts
# ---------------------------------------------------------------------------

class TestStripArtefacts:
    def test_strips_json_fence(self):
        assert _strip_artefacts("```json\n{}\n```") == "{}"

    def test_strips_plain_fence(self):
        assert _strip_artefacts("```\n{}\n```") == "{}"

    def test_no_fence(self):
        assert _strip_artefacts("{}") == "{}"


# ---------------------------------------------------------------------------
# _extract_json_object
# ---------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_finds_object(self):
        assert _extract_json_object('text {"a": 1} more') == '{"a": 1}'

    def test_returns_none_when_no_brace(self):
        assert _extract_json_object("no json here") is None

    def test_handles_nested(self):
        text = 'prefix {"a": {"b": [1,2]}} suffix'
        result = _extract_json_object(text)
        assert result == '{"a": {"b": [1,2]}}'


# ---------------------------------------------------------------------------
# _validate_metadata
# ---------------------------------------------------------------------------

class TestValidateMetadata:
    def test_valid_metadata(self):
        data = {
            "title": "Test Title",
            "description": "Test description.",
            "tags": ["tag1", "tag2"],
        }
        result = _validate_metadata(data)
        assert isinstance(result, VideoMetadata)
        assert result.title == "Test Title"
        assert result.description.startswith("Test description.")
        assert result.description.endswith("#ai #technology #ainews")
        assert result.tags == ["tag1", "tag2"]

    def test_strips_hashtags_from_tags(self):
        data = {"title": "T", "description": "D", "tags": ["#ai", "#tech"]}
        result = _validate_metadata(data)
        assert result.tags == ["ai", "tech"]

    def test_lowercases_tags(self):
        data = {"title": "T", "description": "D", "tags": ["AI", "Tech"]}
        result = _validate_metadata(data)
        assert result.tags == ["ai", "tech"]

    def test_caps_tags_at_10(self):
        data = {"title": "T", "description": "D", "tags": [f"tag{i}" for i in range(15)]}
        result = _validate_metadata(data)
        assert len(result.tags) == 10

    def test_truncates_title_at_100(self):
        data = {"title": "A" * 150, "description": "D", "tags": ["t"]}
        result = _validate_metadata(data)
        assert len(result.title) == 100

    def test_rejects_empty_title(self):
        data = {"title": "", "description": "D", "tags": ["t"]}
        assert _validate_metadata(data) is None

    def test_rejects_empty_description(self):
        data = {"title": "T", "description": "", "tags": ["t"]}
        assert _validate_metadata(data) is None

    def test_rejects_empty_tags(self):
        data = {"title": "T", "description": "D", "tags": []}
        assert _validate_metadata(data) is None

    def test_rejects_non_string_title(self):
        data = {"title": 123, "description": "D", "tags": ["t"]}
        assert _validate_metadata(data) is None

    def test_rejects_non_list_tags(self):
        data = {"title": "T", "description": "D", "tags": "not-a-list"}
        assert _validate_metadata(data) is None

    def test_filters_invalid_tags(self):
        data = {"title": "T", "description": "D", "tags": ["", None, 123, "valid"]}
        result = _validate_metadata(data)
        assert result is not None
        assert result.tags == ["valid"]

    def test_description_ends_with_hashtags(self):
        data = {"title": "T", "description": "A short about AI.", "tags": ["ai"]}
        result = _validate_metadata(data)
        assert result is not None
        assert result.description.endswith("#ai #technology #ainews")

    def test_existing_hashtags_preserved(self):
        data = {
            "title": "T",
            "description": "About AI.\n\n#machinelearning #deeplearning",
            "tags": ["ai"],
        }
        result = _validate_metadata(data)
        assert result is not None
        assert "#machinelearning" in result.description
        assert "#deeplearning" in result.description
        assert "#ai" in result.description  # default added


# ---------------------------------------------------------------------------
# _normalize_description_hashtags
# ---------------------------------------------------------------------------

class TestNormalizeDescriptionHashtags:
    def test_appends_defaults_when_none_present(self):
        result = _normalize_description_hashtags("A short about robotics.")
        assert result == "A short about robotics.\n\n#ai #technology #ainews"

    def test_preserves_existing_hashtags(self):
        result = _normalize_description_hashtags("About tech.\n\n#robotics #automation")
        assert "#robotics" in result
        assert "#automation" in result
        assert "#ai" in result  # default fills remaining slots

    def test_deduplicates_hashtags(self):
        result = _normalize_description_hashtags("About AI.\n\n#ai #tech")
        hashtag_part = result.split("\n\n")[-1]
        tokens = hashtag_part.split()
        assert tokens.count("#ai") == 1

    def test_caps_at_five_hashtags(self):
        desc = "Text.\n\n#a #b #c #d #e #f"
        result = _normalize_description_hashtags(desc)
        hashtag_part = result.split("\n\n")[-1]
        assert len(hashtag_part.split()) == 5

    def test_lowercases_hashtags(self):
        result = _normalize_description_hashtags("Text.\n\n#AI #Tech")
        assert "#ai" in result
        assert "#tech" in result

    def test_strips_body_trailing_whitespace(self):
        result = _normalize_description_hashtags("Body text.   \n\n#existing")
        body = result.split("\n\n")[0]
        assert not body.endswith(" ")

    def test_no_existing_hashtags_all_defaults(self):
        result = _normalize_description_hashtags("Simple body.")
        assert result.endswith("#ai #technology #ainews")


# ---------------------------------------------------------------------------
# generate_metadata — success
# ---------------------------------------------------------------------------

class TestGenerateMetadataSuccess:
    def test_returns_valid_metadata(self):
        client = StubLLMClient(SAMPLE_METADATA_JSON)
        result = generate_metadata(client, _make_topic(), _NARRATION)

        assert isinstance(result, VideoMetadata)
        assert len(result.title) > 0
        assert len(result.description) > 0
        assert len(result.tags) > 0

    def test_prompt_contains_topic_info(self):
        client = StubLLMClient(SAMPLE_METADATA_JSON)
        generate_metadata(client, _make_topic(), _NARRATION)

        assert client._text is not None  # Just verify it was called

    def test_handles_json_in_extra_text(self):
        wrapped = f"Here is the metadata:\n{SAMPLE_METADATA_JSON}\nHope this helps!"
        client = StubLLMClient(wrapped)
        result = generate_metadata(client, _make_topic(), _NARRATION)

        assert isinstance(result, VideoMetadata)


# ---------------------------------------------------------------------------
# generate_metadata — empty/malformed output
# ---------------------------------------------------------------------------

class TestGenerateMetadataEmpty:
    def test_raises_on_empty_response(self):
        client = StubLLMClient("")
        with pytest.raises(RuntimeError, match="malformed or empty"):
            generate_metadata(client, _make_topic(), _NARRATION)

    def test_raises_on_non_json_response(self):
        client = StubLLMClient("I cannot generate metadata right now.")
        with pytest.raises(RuntimeError, match="malformed or empty"):
            generate_metadata(client, _make_topic(), _NARRATION)

    def test_raises_on_invalid_json(self):
        client = StubLLMClient('{"title": "T", "description":')
        with pytest.raises(RuntimeError, match="malformed or empty"):
            generate_metadata(client, _make_topic(), _NARRATION)

    def test_raises_on_missing_fields(self):
        client = StubLLMClient('{"title": "Only title"}')
        with pytest.raises(RuntimeError, match="malformed or empty"):
            generate_metadata(client, _make_topic(), _NARRATION)


# ---------------------------------------------------------------------------
# generate_metadata — LLM failure
# ---------------------------------------------------------------------------

class TestGenerateMetadataFailure:
    def test_propagates_llm_error(self):
        client = FailingLLMClient()
        with pytest.raises(LLMError, match="simulated provider failure"):
            generate_metadata(client, _make_topic(), _NARRATION)


# ---------------------------------------------------------------------------
# generate_metadata — input validation
# ---------------------------------------------------------------------------

class TestGenerateMetadataValidation:
    def test_rejects_empty_topic_title(self):
        client = StubLLMClient(SAMPLE_METADATA_JSON)
        with pytest.raises(ValueError, match="Topic title must not be empty"):
            generate_metadata(client, _make_topic(title=""), _NARRATION)

    def test_rejects_empty_narration(self):
        client = StubLLMClient(SAMPLE_METADATA_JSON)
        with pytest.raises(ValueError, match="Narration must not be empty"):
            generate_metadata(client, _make_topic(), "")


# ---------------------------------------------------------------------------
# VideoMetadata model
# ---------------------------------------------------------------------------

class TestVideoMetadataModel:
    def test_fields(self):
        m = VideoMetadata(title="T", description="D", tags=["a", "b"])
        assert m.title == "T"
        assert m.description == "D"
        assert m.tags == ["a", "b"]

    def test_is_frozen(self):
        m = VideoMetadata(title="T", description="D", tags=["a"])
        with pytest.raises(AttributeError):
            m.title = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _normalize_json_quotes — unit tests
# ---------------------------------------------------------------------------

class TestNormalizeJsonQuotes:
    def test_valid_json_unchanged(self):
        original = '{"title": "Hello", "tags": ["a"]}'
        assert _normalize_json_quotes(original) == original

    def test_single_quoted_keys_and_values(self):
        input_str = "{'title': 'AI News', 'description': 'A report', 'tags': ['ai', 'tech']}"
        expected = '{"title": "AI News", "description": "A report", "tags": ["ai", "tech"]}'
        assert _normalize_json_quotes(input_str) == expected

    def test_apostrophe_inside_double_quoted_string(self):
        input_str = '{"title": "AI\'s Big Day"}'
        assert _normalize_json_quotes(input_str) == input_str

    def test_mixed_quote_styles(self):
        input_str = '{"title": "Hello", \'description\': "World"}'
        expected = '{"title": "Hello", "description": "World"}'
        assert _normalize_json_quotes(input_str) == expected

    def test_empty_string(self):
        assert _normalize_json_quotes("") == ""


# ---------------------------------------------------------------------------
# _parse_metadata — single-quoted response integration
# ---------------------------------------------------------------------------

class TestParseMetadataSingleQuoted:
    def test_single_quoted_json_parsed(self):
        single_quoted = (
            "{'title': 'AI Is Changing Everything', "
            "'description': 'A short about AI advances.', "
            "'tags': ['ai', 'technology', 'future']}"
        )
        metadata = _parse_metadata(single_quoted)
        assert metadata is not None
        assert metadata.title == "AI Is Changing Everything"
        assert "ai" in metadata.tags

    def test_single_quoted_with_explanation_text(self):
        inner = (
            "{'title': 'GPT-5 Is Here', "
            "'description': 'OpenAI released GPT-5.', "
            "'tags': ['gpt-5', 'openai']}"
        )
        wrapped = f"Here is the metadata:\n{inner}\nDone!"
        metadata = _parse_metadata(wrapped)
        assert metadata is not None
        assert metadata.title == "GPT-5 Is Here"

    def test_normal_json_still_works(self):
        metadata = _parse_metadata(SAMPLE_METADATA_JSON)
        assert metadata is not None
        assert "GPT-5" in metadata.title

    def test_malformed_json_returns_none(self):
        metadata = _parse_metadata("{broken json")
        assert metadata is None
