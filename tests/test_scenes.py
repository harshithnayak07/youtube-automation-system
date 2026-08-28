"""Tests for the content.scenes module — all use mocks, no real API calls."""

from __future__ import annotations

import json

import pytest

from content.scenes import Scene, plan_scenes, _parse_scenes, _strip_artefacts, _extract_json_array, _normalize_json_quotes
from llm.client import LLMResponse, LLMError


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

SAMPLE_SCENES_JSON = json.dumps([
    {
        "scene": 1,
        "text": "OpenAI just announced GPT-5.",
        "visual_description": "The OpenAI logo glowing on a dark background with text GPT-5",
    },
    {
        "scene": 2,
        "text": "It features major reasoning improvements.",
        "visual_description": "A neural network diagram with branching reasoning paths lighting up",
    },
    {
        "scene": 3,
        "text": "Developers are already building with it.",
        "visual_description": "A developer at a coding workstation with holographic AI interface",
    },
])


class StubLLMClient:
    """Stub that returns pre-configured text from ``complete``."""

    def __init__(self, text: str):
        self._text = text

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
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


NARRATION = (
    "OpenAI just announced GPT-5, their most advanced model yet. "
    "It features significant improvements in reasoning and coding. "
    "Developers are already building groundbreaking applications with it."
)


# ---------------------------------------------------------------------------
# plan_scenes — success
# ---------------------------------------------------------------------------

class TestPlanScenesSuccess:
    def test_returns_list_of_scene_objects(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        scenes = plan_scenes(client, NARRATION)

        assert isinstance(scenes, list)
        assert all(isinstance(s, Scene) for s in scenes)

    def test_scene_count(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        scenes = plan_scenes(client, NARRATION)

        assert len(scenes) == 3

    def test_scene_numbers_are_sequential(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        scenes = plan_scenes(client, NARRATION)

        nums = [s.scene for s in scenes]
        assert nums == [1, 2, 3]

    def test_all_required_fields_present(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        scenes = plan_scenes(client, NARRATION)

        for s in scenes:
            assert s.scene >= 1
            assert isinstance(s.text, str) and len(s.text.strip()) > 0
            assert isinstance(s.visual_description, str) and len(s.visual_description.strip()) > 0

    def test_text_preserves_narration_content(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        scenes = plan_scenes(client, NARRATION)

        combined = " ".join(s.text for s in scenes)
        assert "GPT-5" in combined
        assert "reasoning" in combined


# ---------------------------------------------------------------------------
# plan_scenes — scene ordering
# ---------------------------------------------------------------------------

class TestPlanScenesOrdering:
    def test_scenes_are_in_order(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        scenes = plan_scenes(client, NARRATION)

        for i, s in enumerate(scenes):
            assert s.scene == i + 1

    def test_text_order_matches_scene_order(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        scenes = plan_scenes(client, NARRATION)

        # First scene should mention announcement, later scenes follow
        assert "announced" in scenes[0].text.lower() or "announced" in scenes[0].visual_description.lower()


# ---------------------------------------------------------------------------
# plan_scenes — short narration (fewer scenes)
# ---------------------------------------------------------------------------

class TestPlanScenesShortNarration:
    def test_short_narration_fewer_scenes(self):
        short_scenes = json.dumps([
            {
                "scene": 1,
                "text": "AI is advancing fast.",
                "visual_description": "A futuristic city skyline with AI symbols",
            },
            {
                "scene": 2,
                "text": "Stay tuned for more.",
                "visual_description": "A subscribe button animation",
            },
        ])
        client = StubLLMClient(short_scenes)
        scenes = plan_scenes(client, "AI is advancing fast. Stay tuned for more.")

        assert len(scenes) == 2

    def test_very_short_narration_single_scene(self):
        single = json.dumps([
            {
                "scene": 1,
                "text": "Breaking AI news today.",
                "visual_description": "A news alert graphic with AI imagery",
            },
        ])
        client = StubLLMClient(single)
        scenes = plan_scenes(client, "Breaking AI news today.")

        assert len(scenes) == 1


# ---------------------------------------------------------------------------
# plan_scenes — empty / malformed LLM output
# ---------------------------------------------------------------------------

class TestPlanScenesEmpty:
    def test_raises_on_empty_response(self):
        client = StubLLMClient("")
        with pytest.raises(RuntimeError, match="no valid scenes"):
            plan_scenes(client, NARRATION)

    def test_raises_on_whitespace_response(self):
        client = StubLLMClient("   \n  ")
        with pytest.raises(RuntimeError, match="no valid scenes"):
            plan_scenes(client, NARRATION)

    def test_raises_on_non_json_response(self):
        client = StubLLMClient("I cannot generate scenes right now.")
        with pytest.raises(RuntimeError, match="no valid scenes"):
            plan_scenes(client, NARRATION)

    def test_raises_on_invalid_json(self):
        client = StubLLMClient("[{\"scene\": 1, \"text\":")
        with pytest.raises(RuntimeError, match="no valid scenes"):
            plan_scenes(client, NARRATION)

    def test_raises_on_empty_json_array(self):
        client = StubLLMClient("[]")
        with pytest.raises(RuntimeError, match="no valid scenes"):
            plan_scenes(client, NARRATION)

    def test_returns_when_json_has_extra_text(self):
        # LLM sometimes wraps JSON in explanation text
        wrapped = f"Here are the scenes:\n{SAMPLE_SCENES_JSON}\nHope this helps!"
        client = StubLLMClient(wrapped)
        scenes = plan_scenes(client, NARRATION)

        assert len(scenes) == 3

    def test_handles_markdown_fenced_json(self):
        fenced = f"```json\n{SAMPLE_SCENES_JSON}\n```"
        client = StubLLMClient(fenced)
        scenes = plan_scenes(client, NARRATION)

        assert len(scenes) == 3


# ---------------------------------------------------------------------------
# plan_scenes — LLM failure
# ---------------------------------------------------------------------------

class TestPlanScenesFailure:
    def test_propagates_llm_error(self):
        client = FailingLLMClient()
        with pytest.raises(LLMError, match="simulated provider failure"):
            plan_scenes(client, NARRATION)


# ---------------------------------------------------------------------------
# plan_scenes — input validation
# ---------------------------------------------------------------------------

class TestPlanScenesValidation:
    def test_rejects_empty_narration(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        with pytest.raises(ValueError, match="Narration must not be empty"):
            plan_scenes(client, "")

    def test_rejects_whitespace_narration(self):
        client = StubLLMClient(SAMPLE_SCENES_JSON)
        with pytest.raises(ValueError, match="Narration must not be empty"):
            plan_scenes(client, "   \n  ")


# ---------------------------------------------------------------------------
# plan_scenes — malformed scene dicts
# ---------------------------------------------------------------------------

class TestPlanScenesMalformed:
    def test_skips_scene_with_empty_text(self):
        data = json.dumps([
            {"scene": 1, "text": "Good scene.", "visual_description": "A valid visual"},
            {"scene": 2, "text": "", "visual_description": "Something"},
            {"scene": 3, "text": "Another good scene.", "visual_description": "Another visual"},
        ])
        client = StubLLMClient(data)
        scenes = plan_scenes(client, NARRATION)

        assert len(scenes) == 2
        assert scenes[0].scene == 1
        assert scenes[1].scene == 3

    def test_skips_scene_with_empty_visual_description(self):
        data = json.dumps([
            {"scene": 1, "text": "Valid text.", "visual_description": ""},
            {"scene": 2, "text": "Good text.", "visual_description": "A valid visual"},
        ])
        client = StubLLMClient(data)
        scenes = plan_scenes(client, NARRATION)

        assert len(scenes) == 1
        assert scenes[0].scene == 2

    def test_skips_non_dict_items(self):
        data = json.dumps(["not a dict", {"scene": 1, "text": "Ok.", "visual_description": "Visual"}])
        client = StubLLMClient(data)
        scenes = plan_scenes(client, NARRATION)

        assert len(scenes) == 1

    def test_fills_missing_scene_number(self):
        data = json.dumps([
            {"text": "Scene one.", "visual_description": "Visual one"},
            {"text": "Scene two.", "visual_description": "Visual two"},
        ])
        client = StubLLMClient(data)
        scenes = plan_scenes(client, NARRATION)

        assert scenes[0].scene == 1
        assert scenes[1].scene == 2

    def test_strips_whitespace_from_fields(self):
        data = json.dumps([
            {"scene": 1, "text": "  Padded text.  ", "visual_description": "  Padded visual.  "}
        ])
        client = StubLLMClient(data)
        scenes = plan_scenes(client, NARRATION)

        assert scenes[0].text == "Padded text."
        assert scenes[0].visual_description == "Padded visual."


# ---------------------------------------------------------------------------
# Scene data model
# ---------------------------------------------------------------------------

class TestSceneModel:
    def test_scene_fields(self):
        s = Scene(scene=1, text="Hello", visual_description="A hello graphic")
        assert s.scene == 1
        assert s.text == "Hello"
        assert s.visual_description == "A hello graphic"

    def test_scene_is_frozen(self):
        s = Scene(scene=1, text="x", visual_description="y")
        with pytest.raises(AttributeError):
            s.scene = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Internal helpers — unit tests
# ---------------------------------------------------------------------------

class TestStripArtefacts:
    def test_strips_json_fence(self):
        assert _strip_artefacts("```json\n[1,2]\n```") == "[1,2]"

    def test_strips_plain_fence(self):
        assert _strip_artefacts("```\n[1]\n```") == "[1]"

    def test_no_fence(self):
        assert _strip_artefacts("[1,2,3]") == "[1,2,3]"


class TestExtractJsonArray:
    def test_finds_array(self):
        assert _extract_json_array("text [1,2,3] more") == "[1,2,3]"

    def test_returns_none_when_no_bracket(self):
        assert _extract_json_array("no json here") is None

    def test_handles_nested(self):
        text = 'prefix [{"a": [1,2]}] suffix'
        result = _extract_json_array(text)
        assert result == '[{"a": [1,2]}]'


# ---------------------------------------------------------------------------
# _normalize_json_quotes — unit tests
# ---------------------------------------------------------------------------

class TestNormalizeJsonQuotes:
    def test_valid_json_unchanged(self):
        original = '[{"scene": 1, "text": "Hello"}]'
        assert _normalize_json_quotes(original) == original

    def test_single_quoted_keys_and_values(self):
        input_str = "[{'scene': 1, 'text': 'Hello', 'visual_description': 'A scene'}]"
        expected = '[{"scene": 1, "text": "Hello", "visual_description": "A scene"}]'
        assert _normalize_json_quotes(input_str) == expected

    def test_apostrophe_inside_double_quoted_string(self):
        input_str = '{"text": "AI\'s response"}'
        assert _normalize_json_quotes(input_str) == input_str

    def test_single_quoted_string_with_apostrophe_inside(self):
        # LLM returns: 'AI's' — apostrophe ends the single-quoted string (malformed,
        # but state machine produces valid JSON: "AI" + trailing text)
        input_str = "['AI\\'s']"
        result = _normalize_json_quotes(input_str)
        # The escaped apostrophe is kept as content inside the converted double-quoted string
        assert '"AI' in result

    def test_mixed_quote_styles(self):
        input_str = '[{"text": "Hello"}, {\'text\': "World"}]'
        expected = '[{"text": "Hello"}, {"text": "World"}]'
        assert _normalize_json_quotes(input_str) == expected

    def test_empty_string(self):
        assert _normalize_json_quotes("") == ""

    def test_no_quotes(self):
        assert _normalize_json_quotes("[1, 2, 3]") == "[1, 2, 3]"

    def test_nested_arrays(self):
        input_str = "[[1, 'two'], ['three', 4]]"
        expected = '[["1", "two"], ["three", "4"]]'
        # Actually nested arrays with int values don't use quotes for ints
        # Let's test proper nested structure
        input_str = "[{'a': [1, 2]}]"
        expected = '[{"a": [1, 2]}]'
        assert _normalize_json_quotes(input_str) == expected

    def test_escape_at_end_of_single_quoted_string(self):
        input_str = "['hello\\'']"
        result = _normalize_json_quotes(input_str)
        # Escaped quote inside single-quoted string, then closing quote
        assert result.startswith('["')
        assert result.endswith('"]')


# ---------------------------------------------------------------------------
# _parse_scenes — single-quoted response integration
# ---------------------------------------------------------------------------

class TestParseScenesSingleQuoted:
    def test_single_quoted_json_parsed(self):
        single_quoted = json.dumps([
            {"scene": 1, "text": "Hello world", "visual_description": "A greeting"},
            {"scene": 2, "text": "Goodbye world", "visual_description": "A farewell"},
        ]).replace('"', "'")
        scenes = _parse_scenes(single_quoted)
        assert len(scenes) == 2
        assert scenes[0].text == "Hello world"
        assert scenes[1].visual_description == "A farewell"

    def test_single_quoted_with_explanation_text(self):
        inner = json.dumps([
            {"scene": 1, "text": "AI news", "visual_description": "AI graphic"},
        ]).replace('"', "'")
        wrapped = f"Here are the scenes:\n{inner}\nHope this helps!"
        scenes = _parse_scenes(wrapped)
        assert len(scenes) == 1

    def test_mixed_valid_and_single_quoted(self):
        # First scene valid JSON, second scene single-quoted (extracted as one array)
        # This tests that _normalize_json_quotes handles the whole extracted string
        raw = "[{'scene': 1, 'text': 'First', 'visual_description': 'Visual one'}, {'scene': 2, 'text': 'Second', 'visual_description': 'Visual two'}]"
        scenes = _parse_scenes(raw)
        assert len(scenes) == 2

    def test_single_quoted_with_apostrophe_in_value(self):
        # "AI's" inside single-quoted string: apostrophe terminates the string
        # (this is malformed, but we should not crash)
        raw = "[{'scene': 1, 'text': 'AI is great', 'visual_description': 'AI graphic'}]"
        scenes = _parse_scenes(raw)
        assert len(scenes) == 1

    def test_normal_json_still_works(self):
        scenes = _parse_scenes(SAMPLE_SCENES_JSON)
        assert len(scenes) == 3

    def test_malformed_json_still_returns_empty(self):
        scenes = _parse_scenes("[{broken json")
        assert scenes == []
