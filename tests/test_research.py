"""Tests for the research.trending module."""

from __future__ import annotations

import pytest

from research.trending import (
    Topic,
    SearchProvider,
    GoogleNewsSearchProvider,
    find_trending_ai_topic,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class StubSearchProvider:
    """Deterministic search provider for testing."""

    def __init__(self, results: list[dict] | None = None, *, fail: bool = False):
        self._results = results or []
        self._fail = fail
        self.queries_received: list[str] = []

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        self.queries_received.append(query)
        if self._fail:
            raise ConnectionError("simulated failure")
        return self._results[:max_results]


SAMPLE_RESULTS = [
    {
        "title": "OpenAI Announces GPT-5 with Reasoning Breakthroughs",
        "url": "https://example.com/gpt5-announcement",
        "source": "TechCrunch",
    },
    {
        "title": "Google DeepMind Releases New Gemini Model",
        "url": "https://example.com/gemini-release",
        "source": "The Verge",
    },
    {
        "title": "AI Startup Raises $500M for Autonomous Agents",
        "url": "https://example.com/ai-startup-funding",
        "source": "Reuters",
    },
]


# ---------------------------------------------------------------------------
# Topic data model
# ---------------------------------------------------------------------------

class TestTopic:
    def test_topic_fields(self):
        t = Topic(
            title="Test Title",
            url="https://example.com/article",
            summary="A short summary.",
            source="TestSource",
        )
        assert t.title == "Test Title"
        assert t.url == "https://example.com/article"
        assert t.summary == "A short summary."
        assert t.source == "TestSource"

    def test_topic_is_frozen(self):
        t = Topic(title="T", url="U", summary="S", source="Src")
        with pytest.raises(AttributeError):
            t.title = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# find_trending_ai_topic — success
# ---------------------------------------------------------------------------

class TestFindTrendingAITopicSuccess:
    def test_returns_topic_with_valid_results(self):
        provider = StubSearchProvider(SAMPLE_RESULTS)
        topic = find_trending_ai_topic(provider)

        assert isinstance(topic, Topic)
        assert topic.title in [r["title"] for r in SAMPLE_RESULTS]
        assert topic.url.startswith("https://")
        assert len(topic.summary) > 0
        assert topic.source != ""

    def test_uses_all_queries_before_giving_up(self):
        provider = StubSearchProvider(SAMPLE_RESULTS)
        find_trending_ai_topic(provider)

        # Default _AI_QUERIES has 5 entries; provider should be called for each
        assert len(provider.queries_received) == 5

    def test_custom_queries_are_passed_through(self):
        provider = StubSearchProvider(SAMPLE_RESULTS)
        custom = ["custom query one", "custom query two"]
        find_trending_ai_topic(provider, queries=custom)

        assert provider.queries_received == custom

    def test_deduplicates_by_url(self):
        duplicate = [
            {"title": "A", "url": "https://same.com/1", "source": "X"},
            {"title": "B", "url": "https://same.com/1", "source": "Y"},
            {"title": "C", "url": "https://different.com/2", "source": "Z"},
        ]
        provider = StubSearchProvider(duplicate)
        topic = find_trending_ai_topic(provider, queries=["q"])

        assert topic.url in ("https://same.com/1", "https://different.com/2")


# ---------------------------------------------------------------------------
# find_trending_ai_topic — failure / edge cases
# ---------------------------------------------------------------------------

class TestFindTrendingAITopicFailure:
    def test_raises_when_all_queries_fail(self):
        provider = StubSearchProvider(fail=True)
        with pytest.raises(RuntimeError, match="No trending AI topics found"):
            find_trending_ai_topic(provider)

    def test_raises_when_results_are_empty(self):
        provider = StubSearchProvider([])
        with pytest.raises(RuntimeError, match="No trending AI topics found"):
            find_trending_ai_topic(provider)

    def test_partial_failure_still_succeeds(self):
        """If some queries fail but others succeed, a topic is returned."""
        class PartialFailProvider:
            def __init__(self):
                self.call_count = 0
            def search(self, query: str, max_results: int = 10) -> list[dict]:
                self.call_count += 1
                if self.call_count <= 2:
                    raise ConnectionError("fail")
                return SAMPLE_RESULTS

        provider = PartialFailProvider()
        topic = find_trending_ai_topic(provider)
        assert isinstance(topic, Topic)


# ---------------------------------------------------------------------------
# GoogleNewsSearchProvider — unit-level
# ---------------------------------------------------------------------------

class TestGoogleNewsSearchProvider:
    def test_parse_rss_items_extracts_data(self):
        sample_xml = """
        <rss><channel>
        <item><title>AI Breakthrough</title><link>https://example.com/1</link><source>TechNews</source></item>
        <item><title>ML Update</title><link>https://example.com/2</link><source>AIWeekly</source></item>
        </channel></rss>
        """
        from research.trending import _parse_rss_items
        items = _parse_rss_items(sample_xml)
        assert len(items) == 2
        assert items[0]["title"] == "AI Breakthrough"
        assert items[1]["url"] == "https://example.com/2"

    def test_parse_rss_handles_empty_input(self):
        from research.trending import _parse_rss_items
        assert _parse_rss_items("") == []
        assert _parse_rss_items("<rss></rss>") == []

    def test_search_provider_protocol_compliance(self):
        provider = GoogleNewsSearchProvider()
        assert isinstance(provider, SearchProvider)
