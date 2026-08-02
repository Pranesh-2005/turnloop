from __future__ import annotations

import httpx
import pytest

from turnloop.config import SearchConfig
from turnloop.errors import ConfigError
from turnloop.tools.websearch import (
    MAX_OUTPUT_CHARS,
    SNIPPET_CHARS,
    WebSearchArgs,
    WebSearchTool,
    _resolve_backend,
    parse_ddg_html,
)

DDG_FIXTURE = """
<html><body><div class="results">
<div class="result results_links_deep web-result">
  <div class="result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc">
        Example <b>Domain</b>
      </a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">
      This domain is for <b>illustrative</b> examples in documents.
    </a>
  </div>
</div>
<div class="result results_links_deep web-result">
  <div class="result__body">
    <h2 class="result__title">
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fother.com%2F&rut=def">
        Other Site
      </a>
    </h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fother.com%2F">
      A second result with a different snippet.
    </a>
  </div>
</div>
</div></body></html>
"""

RATE_LIMITED_FIXTURE = "<html><body><div class='anomaly'>unusual traffic detected</div></body></html>"


# --------------------------------------------------------------------------
# HTML parsing
# --------------------------------------------------------------------------


def test_ddg_html_parses_titles_urls_and_snippets_from_a_recorded_fixture():
    results = parse_ddg_html(DDG_FIXTURE)
    assert [r.title for r in results] == ["Example Domain", "Other Site"]
    assert results[0].url == "https://example.com/page"
    assert "illustrative" in results[0].snippet
    assert results[1].url == "https://other.com/"


def test_ddg_html_with_no_matching_markup_yields_zero_results_not_an_exception():
    assert parse_ddg_html(RATE_LIMITED_FIXTURE) == []


# --------------------------------------------------------------------------
# tool behavior
# --------------------------------------------------------------------------


def _mock_ddg(body: str, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body.encode("utf-8"))

    return httpx.MockTransport(handler)


async def test_zero_results_is_reported_differently_from_a_network_failure(ctx, monkeypatch):
    import turnloop.tools.websearch as websearch

    async def fake_ddg_empty(query, max_results):
        return []

    monkeypatch.setattr(websearch, "_search_ddg", fake_ddg_empty)
    tool = WebSearchTool()
    out = await tool.run(WebSearchArgs(query="asdkjhasdkjh"), ctx)
    assert not out.is_error
    assert "No results" in out.content

    async def fake_ddg_fails(query, max_results):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(websearch, "_search_ddg", fake_ddg_fails)
    out = await tool.run(WebSearchArgs(query="anything"), ctx)
    assert out.is_error
    assert "search failed" in out.content


async def test_search_output_is_bounded_by_snippet_and_total_caps(ctx, monkeypatch):
    import turnloop.tools.websearch as websearch

    long_snippet = "x" * 10_000

    async def fake_ddg(query, max_results):
        return [
            websearch.SearchResult(title=f"Result {i}", url=f"https://x.com/{i}", snippet=long_snippet)
            for i in range(websearch.MAX_RESULTS)
        ]

    monkeypatch.setattr(websearch, "_search_ddg", fake_ddg)
    tool = WebSearchTool()
    out = await tool.run(WebSearchArgs(query="q", max_results=websearch.MAX_RESULTS), ctx)
    assert not out.is_error
    assert len(out.content) <= MAX_OUTPUT_CHARS
    # each individual snippet in the (untruncated-total) render is bounded too
    assert "x" * (SNIPPET_CHARS + 1) not in out.content


def test_websearch_is_read_only_and_usable_in_plan_mode():
    tool = WebSearchTool()
    assert tool.read_only is True
    assert tool.parallel_safe is True
    assert tool.is_read_only_for(WebSearchArgs(query="q")) is True


# --------------------------------------------------------------------------
# backend selection / config
# --------------------------------------------------------------------------


def test_default_search_config_selects_ddg_with_no_api_key_needed():
    cfg = SearchConfig()
    assert cfg.backend == "ddg"
    backend = _resolve_backend(cfg)  # must not raise: no key required
    import turnloop.tools.websearch as websearch

    assert backend is websearch._search_ddg


def test_selecting_a_keyed_backend_without_its_env_var_raises_config_error(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    cfg = SearchConfig(backend="brave")
    with pytest.raises(ConfigError, match="BRAVE_API_KEY"):
        _resolve_backend(cfg)


def test_selecting_tavily_without_its_env_var_names_that_variable(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    cfg = SearchConfig(backend="tavily")
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        _resolve_backend(cfg)


def test_a_custom_api_key_env_name_is_honored(monkeypatch):
    monkeypatch.setenv("MY_BRAVE_KEY", "secret")
    cfg = SearchConfig(backend="brave", api_key_env="MY_BRAVE_KEY")
    backend = _resolve_backend(cfg)  # should not raise, key is present
    assert callable(backend)
