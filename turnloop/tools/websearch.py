"""WebSearch — find pages, don't fetch them.

The hard requirement is zero-config: an agent talking to any provider, with no
API key set up anywhere, still gets a working search. That is why the default
backend is DuckDuckGo's HTML endpoint (`html.duckduckgo.com/html/`) scraped with
the stdlib `html.parser`, the same approach `webfetch.py` uses for page text —
no API key, no new dependency, no sign-up flow standing between "fresh clone"
and "the model can search the web".

Brave and Tavily are opt-in upgrades for when scraping DuckDuckGo's HTML gets
rate-limited or its markup drifts (it will do both; it is not an API and makes
no compatibility promise). Their keys resolve through `api_key_env`, exactly
like `ProviderConfig` — never a literal key sitting in a settings file.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from pydantic import BaseModel, Field

from turnloop.config import SearchConfig
from turnloop.errors import ConfigError
from turnloop.tools.base import Tool, ToolContext, ToolOutput

# A search result list beyond this is not "more information", it's more context
# spent re-reading titles the model will ignore. 10 ranked results is already
# more than most searches need to pick a URL to WebFetch.
MAX_RESULTS = 10
DEFAULT_RESULTS = 5

# Snippets exist to help pick a result, not to substitute for reading the page.
# 200 characters is enough to tell "this is the right doc" from "this isn't".
SNIPPET_CHARS = 200

# At MAX_RESULTS with full-length everything, output could run past 3-4k
# characters — on GLM's 65k-token window that's a real bite for one tool call.
# This caps a full 10-result search comfortably under 1k tokens.
MAX_OUTPUT_CHARS = 4_000

_DEFAULT_KEY_ENV = {"brave": "BRAVE_API_KEY", "tavily": "TAVILY_API_KEY"}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query.")
    max_results: int = Field(
        default=DEFAULT_RESULTS, ge=1, le=MAX_RESULTS,
        description=f"How many results to return (max {MAX_RESULTS}).",
    )


class WebSearchTool(Tool):
    name = "WebSearch"
    Args = WebSearchArgs
    # Genuinely read-only: every backend is a GET/POST against a public search
    # endpoint with no side effects, not an MCP tool where the protocol gives no
    # trustworthy annotation and read_only would just be a guess.
    read_only = True
    parallel_safe = True
    timeout_s = 30.0

    descriptions = {
        "terse": """
Search the web. Returns ranked title/url/snippet, not page content. Use
WebFetch on a result URL to read it.
""",
        "normal": """
Search the web for `query` and return up to `max_results` ranked results.

- Each result is a title, URL and short snippet — not the page itself.
- Use this to find a URL, then WebFetch that URL to read it.
- Works with no configuration (DuckDuckGo by default); Brave or Tavily can be
  configured if scraping gets rate-limited.
""",
        "verbose": """
Search the web for `query` and return up to `max_results` (default 5, max 10)
ranked results, each a title, URL and a snippet truncated to 200 characters.

This is not a page reader. It tells you what exists and where; call WebFetch on
the URL you want to actually read.

Backend is chosen by the `search.backend` setting:
- `ddg` (default): DuckDuckGo's HTML results page, parsed with the standard
  library. No API key required — this is what makes the tool usable out of the
  box.
- `brave` / `tavily`: hosted search APIs, each needing an API key supplied via
  an environment variable named in `search.api_key_env`. If a keyed backend is
  selected and its variable is unset, search fails with a config error instead
  of silently falling back.

Failures are reported distinctly: a network/HTTP failure is not the same
problem as a query that legitimately returned nothing, and each needs a
different next step from you.
""",
    }

    async def run(self, args: WebSearchArgs, ctx: ToolContext) -> ToolOutput:
        cfg = ctx.settings.search
        try:
            backend = _resolve_backend(cfg)
        except ConfigError as exc:
            return ToolOutput.error(str(exc))

        try:
            results = await backend(args.query, args.max_results)
        except httpx.HTTPError as exc:
            return ToolOutput.error(f"search failed: {type(exc).__name__}: {exc}")
        except _ParseFailure as exc:
            return ToolOutput.error(str(exc))

        if not results:
            return ToolOutput(
                content=f"No results for {args.query!r}.",
                display=f"WebSearch {args.query!r} → 0 results",
                metrics={"results": 0},
            )

        return ToolOutput(
            content=_render(results),
            display=f"WebSearch {args.query!r} → {len(results)} result(s)",
            metrics={"results": len(results), "backend": cfg.backend},
        )

    def summary(self, args: WebSearchArgs) -> str:  # type: ignore[override]
        return f"WebSearch {args.query!r}"


class _ParseFailure(Exception):
    """Distinct from a network error: the request succeeded but the response
    could not be understood — the caller needs a different reaction (report
    it, don't retry the same query expecting a different HTML shape)."""


def _render(results: list[SearchResult]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        snippet = r.snippet[:SNIPPET_CHARS]
        lines.append(f"{i}. {r.title}\n   {r.url}\n   {snippet}")
    body = "\n".join(lines)
    return body[:MAX_OUTPUT_CHARS]


def _require_api_key(backend_name: str, cfg: SearchConfig) -> str:
    import os

    env_name = cfg.api_key_env or _DEFAULT_KEY_ENV[backend_name]
    key = os.environ.get(env_name, "")
    if not key:
        raise ConfigError(
            f"search backend '{backend_name}': environment variable {env_name} is not set"
        )
    return key


def _resolve_backend(cfg: SearchConfig):
    if cfg.backend == "ddg":
        return _search_ddg
    if cfg.backend == "brave":
        key = _require_api_key("brave", cfg)

        async def _brave(query: str, max_results: int) -> list[SearchResult]:
            return await _search_brave(query, max_results, key)

        return _brave
    if cfg.backend == "tavily":
        key = _require_api_key("tavily", cfg)

        async def _tavily(query: str, max_results: int) -> list[SearchResult]:
            return await _search_tavily(query, max_results, key)

        return _tavily
    raise ConfigError(f"search backend '{cfg.backend}' is not recognized")


async def _search_ddg(query: str, max_results: int) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0)) as client:
        response = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"user-agent": "turnloop/0.1 (+coding agent)"},
        )
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code} from DuckDuckGo", request=response.request,
            response=response,
        )
    try:
        results = parse_ddg_html(response.text)
    except Exception as exc:  # malformed HTML is the norm, not a bug
        raise _ParseFailure(
            f"could not parse DuckDuckGo's results page ({exc}); its HTML may have "
            "changed, or this may be a rate-limit page rather than results"
        ) from exc
    return results[:max_results]


async def _search_brave(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0)) as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": api_key, "accept": "application/json"},
        )
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code} from Brave", request=response.request,
            response=response,
        )
    data = response.json()
    items = data.get("web", {}).get("results", [])
    return [
        SearchResult(
            title=item.get("title", ""), url=item.get("url", ""),
            snippet=item.get("description", ""),
        )
        for item in items[:max_results]
    ]


async def _search_tavily(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0)) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
        )
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code} from Tavily", request=response.request,
            response=response,
        )
    data = response.json()
    items = data.get("results", [])
    return [
        SearchResult(
            title=item.get("title", ""), url=item.get("url", ""),
            snippet=item.get("content", ""),
        )
        for item in items[:max_results]
    ]


class _DDGResultParser(HTMLParser):
    """Pulls title/url/snippet out of DuckDuckGo's HTML results page.

    Only the three classes that carry the actual result data are tracked
    (`result__a` for the title+link, `result__snippet` for the blurb); every
    other div, span and ad slot on the page is noise this ignores rather than
    a full DOM this builds.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._mode: str | None = None  # "title" | "snippet" | None
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self._pending_url = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_d = dict(attrs)
        classes = attrs_d.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._mode = "title"
            self._pending_url = _unwrap_ddg_redirect(attrs_d.get("href", ""))
        elif tag == "a" and "result__snippet" in classes:
            self._mode = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._mode == "title":
            self._mode = None
        elif tag == "a" and self._mode == "snippet":
            self._flush()
            self._mode = None

    def handle_data(self, data: str) -> None:
        if self._mode == "title":
            self._title_parts.append(data)
        elif self._mode == "snippet":
            self._snippet_parts.append(data)

    def _flush(self) -> None:
        title = " ".join("".join(self._title_parts).split())
        if title and self._pending_url:
            snippet = " ".join("".join(self._snippet_parts).split())
            self.results.append(SearchResult(title=title, url=self._pending_url, snippet=snippet))
        self._title_parts = []
        self._snippet_parts = []
        self._pending_url = ""


def _unwrap_ddg_redirect(href: str) -> str:
    """DuckDuckGo's HTML endpoint links through `//duckduckgo.com/l/?uddg=...`
    instead of the real URL, so the redirect target has to be unwrapped or every
    result would point back at DuckDuckGo."""
    if "uddg=" not in href:
        return href
    parsed = urlparse(href if "://" in href else f"https:{href}")
    qs = parse_qs(parsed.query)
    target = qs.get("uddg", [""])[0]
    return unquote(target) or href


def parse_ddg_html(html: str) -> list[SearchResult]:
    parser = _DDGResultParser()
    parser.feed(html)
    parser._flush()
    return parser.results
