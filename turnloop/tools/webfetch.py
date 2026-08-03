"""WebFetch — retrieve a URL and answer a question about it.

The design constraint is context, not networking. A documentation page is commonly
50-150k characters; dropping that into a 57k-token window destroys the session. So
the page is never returned raw: it is converted to text, capped, and then passed
through the model with the caller's question, and only that answer enters the
conversation.

Redirects are followed only within the same host by default. A tool that fetches
arbitrary URLs on a model's say-so is an SSRF primitive, and cross-host redirects
are how that gets exploited.
"""

from __future__ import annotations

import time
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from turnloop.tools.base import Tool, ToolContext, ToolOutput

MAX_CHARS = 100_000
CACHE_TTL_S = 15 * 60
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254"}

_cache: dict[str, tuple[float, str]] = {}


class WebFetchArgs(BaseModel):
    url: str = Field(description="The URL to fetch. Must be http or https.")
    prompt: str = Field(description="What to extract or answer from the page.")


class WebFetchTool(Tool):
    name = "WebFetch"
    Args = WebFetchArgs
    read_only = True
    parallel_safe = True
    timeout_s = 120.0

    descriptions = {
        "terse": """
Fetch a URL and answer `prompt` about its content. Returns the answer, not the
page. Use for documentation and specific pages you have a URL for.
""",
        "normal": """
Fetch a web page and extract what you asked for.

- `url` must be http or https.
- `prompt` says what to extract; the page is summarized against it and you receive
  the answer, not the raw HTML.
- Results are cached for 15 minutes.
- Use this when you have a specific URL — documentation, an issue, a spec. It is
  not a search engine.
""",
        "verbose": """
Fetch a web page and extract what you asked for.

Arguments:
- `url`: an http or https URL.
- `prompt`: what you want from the page — "the migration steps for v3", "the
  signature of the retry decorator".

Behavior:
- HTML is converted to text (scripts, styles and navigation removed) and capped at
  100,000 characters.
- That text is then passed through the model together with your prompt, and you
  receive the extracted answer. The raw page never enters the conversation, because
  a single documentation page can be larger than the entire context window.
- Redirects are followed only within the same host. A cross-host redirect is
  reported rather than followed.
- Requests to localhost and link-local addresses are refused.
- Responses are cached for 15 minutes, so re-fetching the same URL in one session
  is free.

This is not a search engine. If you do not have a URL, ask the user for one.
""",
    }

    async def run(self, args: WebFetchArgs, ctx: ToolContext) -> ToolOutput:
        parsed = urlparse(args.url)
        if parsed.scheme not in ("http", "https"):
            return ToolOutput.error(f"unsupported scheme {parsed.scheme!r}; use http or https")
        host = (parsed.hostname or "").lower()
        if host in BLOCKED_HOSTS or host.endswith(".local"):
            return ToolOutput.error(
                f"refusing to fetch {host}: local and link-local addresses are blocked"
            )

        cached = _cache.get(args.url)
        if cached and time.monotonic() - cached[0] < CACHE_TTL_S:
            text = cached[1]
            from_cache = True
        else:
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=False
                ) as client:
                    response = await client.get(
                        args.url, headers={"user-agent": "turnloop/0.1 (+coding agent)"}
                    )
                    if response.is_redirect:
                        location = response.headers.get("location", "")
                        target_host = (urlparse(location).hostname or host).lower()
                        if target_host != host:
                            return ToolOutput(
                                content=(
                                    f"{args.url} redirects to a different host: {location}\n"
                                    "Not following it automatically. Call WebFetch again with "
                                    "that URL if you want it."
                                )
                            )
                        response = await client.get(location)
            except httpx.HTTPError as exc:
                return ToolOutput.error(f"fetch failed: {type(exc).__name__}: {exc}")

            if response.status_code >= 400:
                return ToolOutput.error(f"HTTP {response.status_code} from {args.url}")

            content_type = response.headers.get("content-type", "")
            body = response.text
            text = html_to_text(body) if "html" in content_type else body
            text = text[:MAX_CHARS]
            _cache[args.url] = (time.monotonic(), text)
            from_cache = False

        if not text.strip():
            return ToolOutput.error(f"{args.url} returned no readable text")

        answer = await self._extract(text, args.prompt, ctx)
        return ToolOutput(
            content=answer,
            display=f"WebFetch {args.url} ({len(text):,} chars{' cached' if from_cache else ''})",
            metrics={"chars": len(text), "cached": from_cache},
        )

    async def _extract(self, text: str, prompt: str, ctx: ToolContext) -> str:
        """Summarize the page against the prompt using the session's provider."""
        provider = ctx.extras.get("provider")
        if provider is None:
            excerpt = _readable_excerpt(text, 4000)
            return (
                "(no provider available to summarize this page — the text below is an "
                "unsummarized excerpt, not an answer to the prompt; read it yourself)\n\n"
                f"{excerpt}"
            )

        from turnloop.core.events import MessageDone
        from turnloop.core.messages import Message
        from turnloop.providers.base import CompletionRequest

        request = CompletionRequest(
            messages=[
                Message.user_text(
                    f"{prompt}\n\nAnswer only from the page content below. If it does not "
                    f"contain the answer, say so.\n\n---\n{text[:60_000]}"
                )
            ],
            system=["Extract exactly what was asked from the supplied page. Be concise."],
            tools=[],
            max_tokens=2_000,
            temperature=0.0,
        )
        try:
            async for event in provider.stream(request):
                if isinstance(event, MessageDone):
                    return event.message.text.strip() or "(the model returned nothing)"
        except Exception as exc:  # noqa: BLE001
            return (
                f"(summarization failed: {exc}; unsummarized excerpt below, not an answer)\n\n"
                f"{_readable_excerpt(text, 4000)}"
            )
        return (
            "(summarization returned no events; unsummarized excerpt below)\n\n"
            f"{_readable_excerpt(text, 4000)}"
        )

    def summary(self, args: WebFetchArgs) -> str:  # type: ignore[override]
        return f"WebFetch {args.url}"


class _TextExtractor(HTMLParser):
    """HTML to text using only the standard library.

    Not a full renderer, and it does not need to be: headings, paragraphs, list
    items and code blocks carry essentially all the information in technical
    documentation.
    """

    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
    BLOCK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "pre", "blockquote", "section", "article", "table",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        # Root cause of the empty-bullet chrome: this used to fire unconditionally,
        # so a <li> nested inside a skipped <nav>/<header> still emitted its "- "
        # marker even though handle_data() below would never contribute any text
        # for it. Gating on skip_depth stops the marker at its source instead of
        # relying on the text() filter to clean it up afterward.
        if tag == "li" and not self._skip_depth:
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if data.strip():
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        lines = [" ".join(line.split()) for line in joined.splitlines()]
        out: list[str] = []
        blank = 0
        for line in lines:
            # A "- " list item whose only content was an icon (an <svg>, which SKIP
            # already drops) collapses to a bare "-". GitHub-style pages are full of
            # these — two real fetches in one session returned 6k and 23k characters
            # that were nothing but "Skip to content" plus dozens of empty bullets.
            # Dropping them here, in the shared conversion path, fixes every caller
            # at once instead of patching each place that reads the result.
            if line == "-":
                continue
            if not line:
                blank += 1
                if blank > 1:
                    continue
            else:
                blank = 0
            out.append(line)
        return "\n".join(out).strip()


def _readable_excerpt(text: str, limit: int) -> str:
    """First `limit` chars of already-cleaned text, on a line boundary where possible.

    Cutting mid-line at an arbitrary character offset was never the bug — the bug
    was that the text handed to this function was still full of chrome. Now that
    html_to_text() strips empty bullets, a plain slice is enough; this just avoids
    truncating mid-word at the boundary.
    """
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    return text[: cut if cut > limit // 2 else limit].rstrip() + "\n…(truncated)"


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML is the norm
        pass
    return parser.text()
