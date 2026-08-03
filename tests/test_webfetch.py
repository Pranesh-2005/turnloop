"""WebFetch's HTML-to-text conversion and its no-provider fallback.

`tests/fixtures/github_chrome.html` is not hand-invented tag soup: every element in
it (the "Skip to content" link, the `<li role="presentation" aria-hidden="true"
class="prc-UnderlineNav-WrapSpacer--aLgz">` spacer, the underline-nav `<li>`) is
byte-for-byte copied from a real `curl`'d GitHub repository page. The spacer `<li>`
is repeated 40 times to reach the scale of the real incident this fixture guards
against: two GitHub fetches in one session returned 6,314 and 23,042 characters of
pure navigation chrome — "Skip to content" followed by dozens of empty `-` list
bullets — and that dominated the session's 80,680 input tokens. A tidy invented
snippet would not have caught this: GitHub's real markup produces empty list items
specifically because `<li>` wraps an icon-only `<svg>`, which `_TextExtractor`
already drops via SKIP, leaving a bare `-` behind.
"""

from __future__ import annotations

from pathlib import Path

from turnloop.tools.base import ToolContext
from turnloop.tools.webfetch import WebFetchTool, _readable_excerpt, html_to_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_empty_list_bullets_are_stripped_from_real_chrome_heavy_markup():
    html = (FIXTURES / "github_chrome.html").read_text(encoding="utf-8")
    text = html_to_text(html)

    # This is the regression itself: before the fix, each icon-only <li> survived
    # conversion as a bare "-" line. A test on clean HTML would pass either way.
    assert not any(line.strip() == "-" for line in text.splitlines())
    # Both "Skip to content" and the spacer <li> live inside <nav>, which SKIP
    # already drops entirely — correctly, since it is exactly the chrome the
    # incident complained about. The real content must survive that stripping.
    assert "Skip to content" not in text
    assert "Claude SDK for Python" in text
    # The real content is a couple of sentences; chrome must not dwarf it.
    assert len(text) < 500


def test_readable_excerpt_truncates_on_a_line_boundary():
    text = "\n".join(f"line {i}" for i in range(500))
    excerpt = _readable_excerpt(text, 100)
    assert len(excerpt) <= 130
    assert excerpt.endswith("(truncated)")
    body = excerpt.split("\n…(truncated)")[0]
    # Cut lands exactly on a "\n" the source text had, not mid-number.
    assert body in text


def test_readable_excerpt_is_a_no_op_under_the_limit():
    assert _readable_excerpt("short", 4000) == "short"


async def test_no_provider_fallback_labels_the_excerpt_as_unsummarized(ctx: ToolContext):
    """The old fallback silently handed back raw chrome as if it were the answer.

    `ctx.extras` has no "provider" key by default (see conftest's `ctx` fixture), so
    this exercises the exact branch that shipped the bug: a page fetched with no
    summarizing provider configured.
    """
    html = (FIXTURES / "github_chrome.html").read_text(encoding="utf-8")
    text = html_to_text(html)

    answer = await WebFetchTool()._extract(text, "what does this repo do", ctx)

    assert "no provider available" in answer
    assert "not an answer" in answer or "not answer" in answer.lower()
    assert not any(line.strip() == "-" for line in answer.splitlines())
