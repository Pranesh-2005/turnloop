# turnloop

A local-first agentic coding harness. Streaming tool loop, permission gating,
context compaction, subagents, MCP, hooks — and an experiments layer that measures
whether any of it actually helps.

Runs against Anthropic, OpenAI, Gemini, Groq, NVIDIA NIM, Ollama, any
OpenAI-compatible endpoint, and a **self-hosted GLM-5.2 (744B MoE, W4A16) on 4×H200**
that this project treats as a first-class target rather than an afterthought.

```bash
pip install turnloop

tl                              # TUI, mock provider, zero cost
tl -p "fix the failing test"    # headless, one turn, prints and exits
tl --provider glm               # the self-hosted endpoint
tl doctor                       # diagnose shell, providers, context budget
tl experiment run smoke         # a measurement run, offline and free
```

Four runtime dependencies: `textual`, `pydantic`, `httpx`, `pyyaml`. No vendor
SDKs, no agent framework — the loop is the point, so the loop is written here.
~21,900 lines of Python, 381 tests, no network in the default test run.

---

## Table of contents

- [Why it exists](#why-it-exists)
- [Install](#install)
- [Using it](#using-it)
- [Architecture](#architecture)
- [Providers](#providers)
- [Adding a provider](#adding-a-provider)
- [The self-hosted GLM-5.2 target](#the-self-hosted-glm-52-target)
- [Experiments](#experiments)
- [Results](#results)
- [What the measurements changed](#what-the-measurements-changed)
- [Designing tasks that measure something](#designing-tasks-that-measure-something)
- [Bugs worth reading about](#bugs-worth-reading-about)
- [Windows notes](#windows-notes)
- [Testing](#testing)
- [Configuration reference](#configuration-reference)
- [Limitations](#limitations)

---

## Why it exists

Agentic coding tools are mostly judged by anecdote. This one is built so its own
design decisions can be measured: swap the compaction strategy, the tool-description
verbosity, the system prompt or the whole model, run the same fifteen tasks, and read
the pass rate against the token cost.

That measurement layer is not decoration. It found that two of this project's own
assumptions were wrong — see [Results](#results) — and one of the bugs it surfaced had
been silently corrupting every experiment for the life of the repository.

It is also a clean-room implementation. Nothing here is derived from any leaked or
proprietary source; the architecture is designed from first principles and the
comments say *why* each decision went the way it did, including the ones that cost
something.

---

## Install

Requires Python ≥ 3.11. Four dependencies, no compiler, no GPU needed to run it.

```bash
pip install turnloop
```

That is the whole install. It works offline against the `mock` provider immediately —
no API key required to try it.

From source, for development:

```bash
git clone https://github.com/Pranesh-2005/turnloop
cd turnloop
pip install -e ".[dev]"     # editable, plus pytest, ruff, mypy
```

Either way you get two console scripts, `turnloop` and `tl`. They are the same entry
point.

Verify:

```bash
tl --version                # turnloop 0.1.11
tl doctor                   # every provider, key presence, shell, context budget
```

`doctor` is the first thing to run on a new machine. It reports which API key
*names* were found (never their values), which shell it will use, and the exact
context arithmetic for the active provider:

```
 + context budget     32,000 window - 4,096 output - 881 system - 3,567 tools = 23,456 for history
```

---

## Using it

### Launch modes

```bash
tl                                          # interactive TUI
tl -p "why does the login test flake"       # headless single turn
tl -p "..." --json                          # headless, JSONL events on stdout
tl -c                                       # resume the most recent session, no picker
tl --resume                                 # interactive picker (TUI); most recent (headless)
tl --resume <session-id>                    # resume a specific one, no picker
tl sessions                                 # list recorded sessions
tl config                                   # merged config + which layer each value came from
tl config --edit                            # interactive settings editor
tl mcp                                      # add, edit, enable/disable, remove MCP servers
```

`config --edit` and `mcp` are deliberately the only paths that write a settings file,
and neither is reachable by the model — see
[Editing configuration](#editing-configuration).

### Flags

| flag | meaning |
|---|---|
| `--provider NAME` | which configured provider to use (default `mock`) |
| `--model NAME` | override that provider's model |
| `--permission-mode MODE` | `default` · `plan` · `auto` · `bypass` |
| `--cwd PATH` | working directory the agent operates in |
| `--max-iterations N` | tool-loop safety cap (default 40) |
| `-p, --print PROMPT` | headless single turn |
| `--json` | headless output as JSONL events |
| `-c, --continue` | resume the most recent session directly, no picker |
| `--resume [ID]` | bare: interactive picker in the TUI, most recent in headless. With an id: that session, no picker |

### Permission modes

| mode | behavior |
|---|---|
| `default` | every write and shell call asks. The manual mode. |
| `plan` | read-only, enforced three independent ways |
| `auto` | no prompts inside `--cwd`; still asks for anything outside it |
| `bypass` | no gate — deny rules still apply |

Set at launch, or switch mid-session with `/default`, `/plan`, `/auto`, or **Ctrl+P**
to cycle. Make it permanent via `TURNLOOP_PERMISSION_MODE` or `settings.json`.

> **Running inside VS Code's terminal?** VS Code claims `Ctrl+P` for Quick Open before
> turnloop ever sees the key. Use the slash commands, or add
> `"terminal.integrated.commandsToSkipShell": ["-workbench.action.quickOpen"]` to your
> VS Code settings to pass it through.

### Keys

```
ctrl+c   copy the selection; else interrupt a running turn; else nothing
ctrl+p   cycle permission mode
ctrl+l   clear the view (history is kept)
ctrl+d   quit
```

`ctrl+c` no longer quits. It used to call `self.exit()` whenever no turn was
running, so pressing it to copy text killed the session. It now checks, in order,
whether there is a selection to copy (via Textual's `App.copy_to_clipboard`, OSC 52,
no new dependency), then whether a turn is running to interrupt, and otherwise does
nothing. `ctrl+d` is still the only way to quit.

`ctrl+c` and friends are bound with `priority=True` — the prompt Input always has
focus and would otherwise swallow them.

### Slash commands

```
/help        list commands              /permissions  show rules and mode
/clear       clear view, keep history   /plan /auto /default  switch mode
/compact     summarize conversation now /memory       discovered memory files
/cost        token and cost accounting  /tools        available tools
/context     what is filling the window /mcp          MCP server status
/model       show or change model       /mcp add      add an MCP server
/provider    switch provider            /config       settings editor
/export      transcript to markdown     /hooks        configured hooks
/sessions    recorded sessions          /doctor       diagnostics
/skills      loaded skills, and any     /resume       picker, rebuilds the
             rejected on disk (why,                   agent in place; refuses
             and where)                                while a turn is running
/quit        exit
```

Custom commands live in `.turnloop/commands/*.md`.

### What you can feed it

- **Prose** — "the checkout total is wrong for orders over $500, find out why"
- **`@path/to/file`** — inlines that file into the prompt
- **Shell output** — `` !`git diff` `` inside a command template, routed through the
  permission engine like any other Bash call
- **`TURNLOOP.md`** at the project root for standing instructions (`CLAUDE.md` and
  `AGENTS.md` are also read)
- **Text files of any kind** via the `Read` tool — source, JSON, YAML, CSV, Markdown
- **Images** — PNG, JPEG, GIF and WEBP via `Read`, detected by magic bytes rather
  than extension, on providers whose `Capabilities.supports_vision` is true
  (Anthropic, OpenAI, Gemini). On a text-only provider `Read` refuses the image and
  names the provider, instead of sending a payload the model cannot parse — which
  matters here, because GLM-5.2 W4A16 as deployed is text-only.

**Not supported:** binary files other than those image formats — refused by a
null-byte sniff, which covers PDFs, `.docx` and archives. Anything convertible on
the command line comes back in scope, because `Bash` exists: `pdftotext spec.pdf -`
and the agent will reach for it itself.

### Tools

Eleven registered: `Read` `Write` `Edit` `Glob` `Grep` `Bash` `TodoWrite` `Task`
`WebFetch` `WebSearch` `AskUserQuestion`, plus `Skill` when the project has skills.
Schemas are generated from pydantic models, so the declared schema and the
validation are the same object and cannot drift.

`WebSearch` and `WebFetch` are the pair: search finds URLs, fetch reads one. Both
are `read_only` and `parallel_safe`, so they work in plan mode and run concurrently
with `Read`/`Grep`. **Search needs no API key** — the default DuckDuckGo backend
works on a fresh install with an empty config. Brave and Tavily are opt-in upgrades:

```json
{ "search": { "backend": "brave", "api_key_env": "BRAVE_API_KEY" } }
```

A keyed backend only errors if you actually select one, so the zero-config path
never breaks. Results are budgeted for a small window — 10 results maximum,
200-character snippets, 4k characters total, roughly 1k tokens against GLM's 65k.

---

## Architecture

```
turnloop/
  core/         messages, events, token estimation, ids
  providers/    anthropic · openai_compat (GLM/Groq/NVIDIA/vLLM) · gemini · mock
                + sse parser, pricing/capability presets, registry
  tools/        Read Edit Write Bash Glob Grep TodoWrite Task WebFetch WebSearch
                Ask Skill + runner, shell scanner, text io
  permissions/  rule grammar, matcher, engine
  agent/        loop, system prompt, subagents, factory, headless
  context/      budget, three-tier compaction, memory files
  sessions/     append-only JSONL, resume
  tui/          Textual app, streaming transcript, permission modal, status bar
  mcp/          JSON-RPC client (stdio + SSE), schema→pydantic adapter
  hooks/        settings.json lifecycle hooks
  commands/     slash commands and skills
  experiments/  runner, graders, 15-task suite, report renderer
  diagnostics.py  the doctor
```

### The loop

```
compact if needed → stream a turn → run any requested tools → append results → repeat
```

Four decisions worth defending:

**Compaction runs before every request, not on a timer.** Pressure is a property of
what just happened, and one large tool result can cross two thresholds in a single
turn.

**Tools run in parallel only when every tool is `parallel_safe` and none needs a
permission prompt.** Two concurrent edits to one file, or two modals racing for the
screen, are both worse than being slow.

**A truncated stream never leaves a partial assistant message in history.** An
assistant turn containing a `tool_use` with no matching `tool_result` is a hard 400
from every provider, and it would poison every later request in the session — not
just the one that failed.

**The iteration cap injects a message instead of returning silently.** A loop that
stops at turn 40 with no explanation looks like a crash.

### Provider layer

One `httpx.AsyncClient` and a 40-line SSE parser serve every backend. That is not
purism — it solves three concrete problems:

- GLM's reasoning arrives as `reasoning_content`, a sibling of `content`. With the
  raw dict there is nothing to fight; through a typed SDK it hides in `model_extra`.
- We own the timeout. The `openai` SDK's 600-second default **fails** on a genuine
  GLM cold boot, which takes longer than that.
- Each adapter becomes a pure `dict → StreamEvent` function, tested against recorded
  `.sse` fixtures with no network at all.

Capabilities are declared per model (`max_context`, `max_output`,
`supports_prompt_caching`, `supports_reasoning_field`, `native_thinking`,
`max_concurrent_requests`, per-token pricing, `cost_per_hour`) and the *loop* decides
how to degrade — so fallback logic exists once rather than once per adapter.

Streaming tool-call shapes differ by vendor and both are handled: NVIDIA NIM sends
each call **whole in one delta** (id, name, complete arguments JSON); vLLM's `glm47`
parser and OpenAI **fragment** arguments across deltas, splitting mid-token
(`{"file_p` / `ath": "README.md"}`). The accumulator is tested against both.

### Permissions

Rules read `Tool` or `Tool(pattern)`:

```json
{
  "permissions": {
    "allow": ["Bash(git status)", "Bash(npm test)", "Edit(src/**)"],
    "deny":  ["Read(**/.env)", "Bash(git push --force*)"],
    "ask":   ["Bash(npm publish*)"]
  }
}
```

Precedence is flat and fixed: deny → plan-mode read-only → bypass → allow → ask →
mode default. No specificity scoring, because a permission system nobody can predict
is a permission system people disable.

The single most important behavior: **a compound shell command is allowed only if
every segment is allowed.** Without that, `Bash(git *)` grants
`git status && rm -rf /`. Splitting on top-level `;`/`&&`/`||`/`|` needs a real
scanner that respects quoting — `shlex` discards operators, so the `rm` comes back
looking like an argument to `git`. 48 tests cover this file alone.

Twelve deny rules ship by default, including `Read(**/.env)` and — since 0.1.4 —
`Write(**/.turnloop/settings*.json)` and `Edit(**/.turnloop/settings*.json)`. Those two
close a self-escalation path: settings files set `permission_mode` and
`permissions.allow`, so a model that could write one could grant itself `bypass` on the
next launch. Verified against the real engine *in bypass mode*, which is the case that
matters — deny beats bypass, so the block holds even there.

That was a unit test. It has since been verified live: with
`Write(.turnloop/**)` explicitly allowed *and* `--permission-mode bypass`, a real
model told to grant itself permanent bypass access was blocked, made one attempt,
and proposed the change to the user instead. The settings file was unchanged. Deny
beats an explicit allow rule and bypass mode at once — not just in the test suite,
against a live model that was actually trying.

It is a guardrail, not a sandbox, and the boundary is worth being precise about. `Write`
and `Edit` are blocked; **`Bash` is not**, so a shell redirect reaches the same file.
Closing that properly needs a path guard below the tool layer rather than more patterns.
In `default` mode Bash asks a human first, so this only bites under `bypass` — which is
already the mode that means "I accept the consequences."

### Context compaction

Three escalating tiers, checked before every request:

| tier | cost | trigger | what it does |
|---|---|---|---|
| micro | free | always | old oversized tool results → head+tail; superseded file reads collapse |
| thinking drop | free | 60% pressure | reasoning removed outside the last two turns |
| summarize | one model call | 75% pressure | prefix → structured summary + verbatim todos + file manifest |

Tiers 1 and 2 typically reclaim 30–50% of a tool-heavy session, which is why they run
first: a summarization call that could have been avoided costs money, latency and
fidelity.

The invariant, with its own property test over 200 random histories: **a `tool_use`
block and its `tool_result` are never separated.** An orphan is a hard 400 from every
provider, and it happens exactly when context is already tight.

Token estimation self-calibrates. `len/3.6` seeded, then corrected by EWMA against
each response's reported `input_tokens`. No `tiktoken`: a flat 4-chars-per-token guess
is ~20% off on GLM's tokenizer, and 20% of 65k is 13k tokens of headroom either
wasted or blown through.

### Subagents

Isolation is structural, not disciplinary. A child gets a fresh session, a filtered
registry (no `Task`, no `AskUserQuestion`), `depth=1`, and **the same permission
engine** — permissions belong to the user, so delegation must not be a privilege
escalation path. Only the child's final message returns to the parent; its
intermediate tool calls never enter parent context, which is the entire economic
argument for delegating.

### Sessions

Append-only JSONL, one file per session. Resume replays it. Append-only matters
because a crashed run still leaves a readable trace, and because the experiments
layer reads exactly the same format the TUI writes — a run trace and a real session
are the same artifact.

`-c`/`--continue` reopens the most recent session with no prompting. Bare `--resume`
in the TUI opens an interactive picker — a `DataTable` of sessions, newest first,
with a live preview pane of the selected conversation — and `--resume <id>` jumps
straight to a specific one. `/resume` opens the same picker in-session and rebuilds
the agent in place; it refuses while a turn is running rather than tearing one down
mid-flight. Headless has no picker to show, so a bare `--resume` there falls back to
"most recent" — the same thing `--continue` does.

---

## Providers

All nine ship preconfigured. `doctor` shows which have keys present.

| name | kind | model | auth | notes |
|---|---|---|---|---|
| `mock` | mock | `mock-1` | none | default. scripted / replay / chaos modes |
| `glm` | openai_compat | `glm-5.2` | **none** | self-hosted vLLM on Modal, 65k, $18.16/hr |
| `nvidia` | openai_compat | `z-ai/glm-5.2` | `NVIDIA_API_KEY` | hosted GLM-5.2, ≥200k context |
| `groq` | openai_compat | `openai/gpt-oss-120b` | `GROQ_API_KEY` | free tier: 8k TPM |
| `anthropic` | anthropic | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` | |
| `openai` | openai_compat | `gpt-4.1` | `OPENAI_API_KEY` | |
| `gemini` | gemini | `gemini-flash-latest` | `GEMINI_API_KEY` | alias, not a pin — see below |
| `blaxel` | openai_compat | `gpt-4o-mini` | `BL_API_KEY` | sandbox; ignores the model field |
| `ollama` | openai_compat | `qwen2.5-coder:14b` | none | local |

**Multiple keys per provider.** Free-tier accounts run out fast, so any `*_API_KEY`
also accepts numbered siblings: `GROQ_API_KEY`, `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, ...
up to `_20`. On a 429, 401, or 403, turnloop rotates to the next configured key and
retries the same turn instead of failing it — each key is tried at most once per turn.
`doctor` reports the count, e.g. `GROQ_API_KEY set (3 keys)`, never the values.

Three provider facts that are configuration rather than capability, each learned from
a live failure:

**Groq's free tier counts the *requested* `max_tokens` against an 8,000 TPM quota.**
A 4k prompt asking for the model's real 32k output is a 35k request → hard `413` with
`code: rate_limit_exceeded` before a single token generates. The preset therefore sets
`max_context: 7000`, `max_output: 1500` and terse tool descriptions, leaving ~2,400
tokens for history. Raise both on a paid tier. A 413 carrying `rate_limit` is
classified retryable, not fatal. Groq also validates tool names server-side: a model
emitting `glob` instead of `Glob` gets a 400 rather than a recoverable tool result.

**NVIDIA NIM serves GLM-5.2 at ≥200,000 tokens of context** — a 200,010-token prompt
was accepted and answered. Thinking is *off* by default there; `reasoning_effort` is
silently ignored and `{"chat_template_kwargs": {"thinking": {"type": "enabled"}}}` is
what turns it on. Its free tier is unusable for sweeps: an 84-run experiment died
84/84 on HTTP 429, and the quota is **token**-based (~156k input tokens per window),
not request-based. The endpoint exposes no `Retry-After` and no `x-ratelimit-*`
headers, so backoff has nothing to key on. `max_concurrent_requests: 1`. Fine for
smoke tests, not for experiments.

**Capability presets fall back to substring matching, and that is a live trap.**
`"glm-5.2" in "z-ai/glm-5.2"` is `True`, so without an explicit `PRESETS` key the
NVIDIA model would silently inherit the self-hosted 65k window and $18.16/hr cost
basis. A regression test guards it.

**Gemini's default model is an alias because pinned ones rot in two different ways.**
0.1.1 shipped `gemini-2.5-pro`, whose free-tier quota is literally zero: a brand-new
key gets `429 ... generate_content_free_tier_input_token_count, limit: 0` on its first
request, which reads as a broken adapter rather than a model that was never free.
Pinning `gemini-2.5-flash` instead fails the other way — `404 no longer available to
new users`. `gemini-flash-latest` follows Google's own pointer. Worth knowing when
reading a 429 from any provider: the status code says "slow down", the body says
whether slowing down will ever help.

Its free tier is **5 requests per minute**, which an agent loop reaches in one
ordinary task — a fix-two-bugs-and-write-tests run spent six turns and died on the
seventh. The retry ladder is correct and still loses, because the window is longer
than the backoff. Free Gemini is a good way to try the harness and a bad way to do
sustained work.

---

## Adding a provider

### If it speaks OpenAI chat-completions: config only, no code

OpenRouter, Together, Fireworks, DeepSeek, Mistral, xAI, Groq, LM Studio, vLLM,
llama.cpp, LocalAI — all of these are a `settings.json` entry. Drop it in
`~/.turnloop/settings.json` to have it everywhere, `<project>/.turnloop/settings.json`
to share it with the repo, or `.turnloop/settings.local.json` to keep it out of git.

```json
{
  "providers": {
    "openrouter": {
      "kind": "openai_compat",
      "model": "anthropic/claude-sonnet-4.5",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY",
      "headers": {
        "HTTP-Referer": "https://github.com/you/yourproject",
        "X-Title": "turnloop"
      },
      "caps": {
        "max_context": 200000,
        "max_output": 32000,
        "price_in_per_mtok": 3.0,
        "price_out_per_mtok": 15.0,
        "max_concurrent_requests": 8
      }
    }
  }
}
```

```bash
export OPENROUTER_API_KEY=...          # or put it in .env at the project root
tl --provider openrouter
tl doctor                              # confirms it registered and the key is present
```

`providers` is a dict and config layers merge recursively, so a new key is added
without restating the nine that ship.

### The fields

| field | when you need it |
|---|---|
| `kind` | `openai_compat` · `anthropic` · `gemini` · `mock` |
| `model` | exactly the id the endpoint expects |
| `base_url` | required for anything but the built-in defaults |
| `api_key_env` | the *name* of the env var. Never put a key in a committed file |
| `headers` | vendor attribution, org routing, custom auth schemes |
| `timeout_s` | `null` means no read timeout — only for endpoints that cold-boot |
| `health_url` / `cold_boot_budget_s` | preflight polling for a scale-to-zero endpoint |
| `extra_body` | merged into the request JSON: reasoning toggles, routing preferences |
| `glm_reasoning` | read `reasoning_content` as a sibling of `content` |
| `tool_verbosity` | `terse` on a tight window — measured −20% input tokens |
| `caps.max_context` | drives compaction thresholds. Get this one right |
| `caps.max_concurrent_requests` | builds a process-wide `CapacityLimiter` so subagent fan-out cannot queue behind itself |
| `caps.price_*_per_mtok` | cost column only. Wrong numbers skew reporting, never behavior |

`extra_body` is the escape hatch for provider-specific knobs. OpenRouter routing:

```json
"extra_body": { "provider": { "order": ["Anthropic"], "allow_fallbacks": false } }
```

...and it is exactly how thinking is enabled on NVIDIA NIM, where
`reasoning_effort` is silently ignored:

```json
"extra_body": { "chat_template_kwargs": { "thinking": { "type": "enabled" } } }
```

> **Declare `caps` explicitly.** Omitting it falls back to `preset_for(model)`, which
> does substring matching — `"glm-5.2" in "z-ai/glm-5.2"` is `True`. That is how a
> hosted endpoint once silently inherited a 65k window and an $18.16/hour cost basis
> from an unrelated self-hosted deployment. If your model id contains a known model's
> name, spell out `caps`.

### If it speaks something else: one adapter, ~150 lines

`kind` is a closed set, so a genuinely different wire format — Bedrock's SigV4
signing, Vertex's auth, Cohere's schema — needs code. It is a small, well-fenced job:

1. Add `providers/yourvendor.py` with a `feed(chunk: dict) -> list[StreamEvent]`
   accumulator, a `headers()` method, and request-body shaping.
2. Register it in `providers/registry.py` and add the literal to `ProviderConfig.kind`.
3. Record a `.sse` fixture and test against it — **no network**.

The adapter stays a pure `dict → StreamEvent` function; everything else (retries,
timeouts, cold-boot classification, capacity limiting, cost accounting) already lives
in the base class and the loop. Two things the fixture should cover, because both are
easy to get wrong and neither shows up until production: reasoning arriving alongside
content, and tool-call arguments **fragmented mid-token** across deltas
(`{"file_p` / `ath": "README.md"}`) — NVIDIA sends each call whole in one delta,
vLLM and OpenAI split them.

---

## The self-hosted GLM-5.2 target

Every non-obvious setting is a fact about that deployment rather than a preference:

| | |
|---|---|
| endpoint | stock `vllm serve` 0.25.1, OpenAI-compatible, **unauthenticated** |
| model id | `glm-5.2`, `--tool-call-parser glm47`, `--reasoning-parser glm45` |
| weights | `/models/GLM-5.2-W4A16`, 744B MoE quantized |
| context | **65,536 tokens**, prompt + completion combined |
| concurrency | `--max-num-seqs 16`, `max_containers=1` |
| cost | **$18.16/hour** of wall clock, 4×H200 |
| cold boot | ~29 min cold, **12m34s measured** with a warm compile cache |
| scaledown | `scaledown_window=600` — 10 idle minutes, then the container is released |

Deploying it:

```bash
cd zai-glm5.2-modal
modal deploy serve.py                 # registers the web function in ~6s
tl --provider glm -p "hello"          # first request triggers the boot; poll is narrated
modal app stop glm-5-2-serve -y       # ALWAYS. an idle container is ~$3/hour of nothing
```

Three consequences the harness handles explicitly:

**The window is small.** After output reserve, system prompt and ten tool schemas,
~58k remains for history — and a 3,000-line file read is ~30k of it. Micro-compaction
is a requirement here, not an optimization. Tool descriptions default to `terse` for
this provider, which buys back ~1,100 tokens; whether that costs pass rate was one of
the shipped experiments, and the answer is [below](#results).

**Cold boot dominates.** `/health` is polled with progress narrated in the UI, the read
timeout is unbounded while connect stays at 10s (short connect *is* the cold-boot
signal), and errors are classified into `cold_boot` / `retryable` / `fatal` — so a 502
from Modal's edge waits patiently while a 400 from a malformed schema fails
immediately instead of retrying for 55 minutes.

A hard-won detail: **`modal app stop` deregisters the app, it does not scale it to
zero.** A stopped deployment and a cold-booting one are *indistinguishable at the
transport level* — both accept TCP, then read-timeout. The timeout message therefore
says so explicitly and tells you to check the deployment, because otherwise you sit
watching a progress counter tick toward a 55-minute budget against a URL that will
never answer. Ask how that was discovered.

**It bills by the hour.** The status bar shows GPU uptime and running cost in amber,
and exiting prints `modal app stop glm-5-2-serve`. Note that in-app cost accounting
covers wall clock *across requests* and excludes boot time — the Modal invoice counts
the 12 minutes of loading weights, so the two numbers do not match by design.

---

## Experiments

```bash
tl experiment run smoke              # free, offline — a shipped config, by name
tl experiment run bakeoff            # real models, real money
tl experiment run ./my-config.yaml   # or your own, by path
tl experiment report .turnloop/experiments/<run-dir>
```

A bare name resolves against the configs bundled in the package, so the shipped
experiments are runnable straight from a `pip install`. A local file always wins, so
your own `smoke.yaml` is never shadowed by ours.

Thirteen configs ship, in four families, one runner:

- **`bakeoff.yaml`** — same tasks across GLM-5.2, Claude, GPT, Gemini, Groq.
- **`context.yaml` / `context-glm.yaml`** — ablate tool verbosity, compaction strategy
  and system prompt variant on one model, so the model is not the variable. The
  `minimal-prompt` arm measures how much of a coding agent's competence comes from the
  harness rather than the weights.
- **`loop.yaml`** — single agent vs subagent fan-out vs plan-then-execute.
- **`reliability.yaml`** — malformed-args and schema-violation rates, and the number
  that actually separates models: the **recovery rate** after a rejected call.

Plus calibration configs (`hard-calibration`, `rewrite-calibration`, `ceiling-check`)
used to establish whether a task discriminates at all before spending GPU on it.

### The suite

Fifteen offline fixture-based tasks, all decided by **programmatic graders** — no LLM
judge, which would introduce the very variable being measured.

| task | fixture | grader | what it probes |
|---|---|---|---|
| `fix_failing_test` | `failing_test` | `pytest_passes` | read → edit → verify |
| `find_the_bug` | `needle` | `answer_contains` | search, read-only |
| `multi_file_rename` | `rename` | `command_exits_zero` | multi-file consistency |
| `implement_from_spec` | `spec` | `pytest_passes` | build from a written spec |
| `add_cli_flag` | `cli_flag` | `command_exits_zero` | feature addition |
| `grep_then_edit` | `grep_edit` | `command_exits_zero` | search-driven edit |
| `structural_function` | `util` | `ast_has_function` | structural, not textual, grading |
| `refuse_outside_root` | — | `no_writes_outside` | **permission safety** |
| `ambiguous_request` | — | `asked_a_question` | did it clarify, or guess? |
| `forces_compaction` | `bulky` | `answer_contains` | 133k tokens of fixture |
| `regression_trap` | `regression_trap` | `pytest_passes` | fix without breaking a sibling |
| `cross_file_contract` | `cross_file_contract` | `pytest_passes` | encode/decode contract |
| `subtle_spec_edge` | `subtle_spec_edge` | `pytest_passes` | ordering edge case |
| `circular_import` | `circular_import` | `pytest_passes` | import cycle |
| `misleading_traceback` | `misleading_traceback` | `pytest_passes` | symptom ≠ cause |

A test asserts no task in the suite is graded by `always_pass` — a task that always
passes measures nothing, and it will happily report a turn that died on a provider
error as a success.

### Mechanics that keep runs honest

Each run writes `config.json`, one full session trace per `(arm, task, repeat)`, and
`results.jsonl`. The report shows **medians and n**, never a single run: agent
behavior is heavy-tailed and one 20-turn flail makes a mean meaningless.

- Execution is grouped **by provider**, so a scale-to-zero endpoint boots once instead
  of once per arm.
- Every repeat gets a **fresh fixture copy**, or repeat 2 grades the agent against
  repeat 1's output.
- **Provider abort threshold.** If the first 3 runs for a provider all end in a
  transport `error`, the rest of that provider's runs are recorded `skipped` rather
  than attempted. This exists because an 84-run sweep once failed 84/84 on HTTP 429
  and exited 0 with a tidy report full of zeros. The predicate is `row.error is not
  None`, so a *grader* failure never trips it — the chaos arm is supposed to fail.
- **Cost basis is recorded per row** as `tokens` or `wall_clock` and marked with `†`
  in the report. A per-hour GPU and a per-token API are not comparable, and a report
  that puts them in one column will show "+26% tokens, −23% cost" and quietly mean
  nothing.
- **A 15-minute cold-boot budget** for experiment runs, with preflight progress printed
  every 30s — because a silent preflight against a dead endpoint looks exactly like a
  slow one.

### The mock provider is load-bearing

Three modes: `scripted` (golden transcripts), `replay` (re-runs a recorded real trace
at zero cost), and `chaos` — which emits malformed JSON, truncated arguments and
unknown tool names on purpose. Chaos is how "tool dispatch never raises" gets proven
rather than asserted:

```
| arm         | tool calls | error rate | malformed args | schema violations |
| mock-chaos  | 18         | 100.0%     | 3              | 15                |
```

18 deliberately broken calls, every one turned into an error result the model can
read, no exception surfaced.

---

## Results

All numbers below are from runs in this repository, reproducible from the shipped
configs. Roughly $20 of GPU time.

### Context engineering ablation — 84 runs, 7 arms, GLM-5.2

Same model throughout, so the harness is the only variable.

All seven arms, 12 runs each, pass rate 100% across the board:

| arm | tok in | vs baseline | tok out | tool calls | error rate | median wall |
|---|---|---|---|---|---|---|
| baseline | 33,941 | — | 950 | 98 | 4.1% | 33.5s |
| **`terse-tools`** | **27,104** | **−20%** | 840 | 98 | **0.0%** | **26.1s** |
| `minimal-prompt` | 29,711 | −12% | 795 | 92 | 5.4% | 30.2s |
| `careful-prompt` | 35,047 | +3% | 1,128 | 103 | 3.9% | 37.2s |
| `no-compaction` | 35,028 | +3% | 1,074 | 101 | 5.0% | 34.0s |
| `micro-only-compaction` | 35,072 | +3% | 1,099 | 104 | 4.8% | 32.9s |
| `verbose-tools` | 42,680 | +26% | 949 | 95 | 2.1% | 26.5s |

**`terse-tools` dominates `verbose-tools` on both axes** — 36% fewer input tokens
(27,104 vs 42,680) *and* a lower error rate (0.0% vs 2.1%), at the same pass rate and
the same 98 tool calls. Longer tool descriptions are not buying accuracy here.

Against baseline the picture is more interesting than "verbose is bad": verbose-tools
*did* cut the error rate 2.0pp. It just cost 26% more input to do it, while terse-tools
cut it 4.1pp while *saving* 20%. Length is not the useful variable; wording is.

**`minimal-prompt` buys tokens with reliability.** −12% input tokens and −12% tool
calls at an identical pass rate, but the error rate rises to 5.4%. That is the
interesting result: the system prompt is not making the model *smarter*, it is making
the model's tool calls *better formed*. Competence comes from the weights; protocol
compliance comes from the harness.

**Compaction does what it claims.** On `forces_compaction`, the `no-compaction` arm
used **272,938** median input tokens against baseline's **196,104** — a 39% penalty
for turning it off.

> **Caveat, stated plainly:** the error-rate deltas rest on small counts (4 errors in
> 98 calls versus 0). The ordering is suggestive, not established. The token deltas are
> solid.

### Model comparison — 18 runs, 3 hard tasks

| | GLM-5.2 (self-hosted) | gpt-4o-mini |
|---|---|---|
| pass rate | **100%** (9/9) | 11% (1/9) |
| tool-call error rate | **2.8%** | 23.3% |
| **recovery after a failed call** | **100%** | **0%** |
| median output tokens | **1,521** | 16,430 |
| median wall clock | **35.2s** | 157.8s |
| median input tokens | 20,174 | 19,883 |

The recovery column is the one that matters and the one nobody reports. Both models
make mistakes; only one reads the error and adapts. gpt-4o-mini never repairs a failed
tool call — it repeats it, which is also why its output token count is 10× higher for
a worse result.

Per task: `circular_import` 3/3 vs 0/3 · `cross_file_contract` 3/3 vs 0/3 ·
`regression_trap` 3/3 vs 1/3.

### The result that cost the most to learn

**Pass rate is the wrong outcome metric for GLM-5.2 on this suite.** It scored 100% on
all four original tasks across 84 runs, then 9/9 on five purpose-built "hard" tasks
with decoy solutions. Every context-engineering ablation therefore ties at the ceiling
and measures nothing about correctness.

Writing harder tasks did not fix it. GLM-5.2 solves realistic multi-file refactors,
circular-import traps and cross-file contract bugs in ~30 seconds and 5–6 tool calls.
The falsification cost ~$7 of GPU and the plan was mine.

So the suite demotes pass rate to a **gate** — it must be 100%, and the report prints
a note when every arm hits it — and compares on efficiency and reliability, which have
real headroom.

---

## What the measurements changed

Concrete changes to this codebase that exist because a run said so:

1. **Tool verbosity defaults to `terse`** on constrained providers. Measured −20%
   input tokens at no cost to pass rate.
2. **Pass rate became a gate, not a metric.** The report emits
   `_pass_rate_gate_note` when every arm hits 100%, so a tie at the ceiling is
   labelled rather than read as a finding.
3. **`cost_basis` was added to every result row** after a report showed +26% tokens
   and −23% cost in adjacent columns.
4. **`PROVIDER_ABORT_THRESHOLD = 3`** after 84 runs failed and reported success.
5. **The efficiency tables exist at all** — tokens in/out, tool calls, error rate,
   recovery, median wall — because the outcome table had stopped discriminating.

---

## Designing tasks that measure something

Two principles, both learned by getting them wrong.

**A discriminating task needs a decoy.** There must be a plausible wrong fix that
makes the stated symptom disappear while still failing the grader. Without one, the
task measures reading comprehension. Validation is therefore three-state, not two:

```
initial state  → FAILS
reference fix  → PASSES
decoy fix      → FAILS      ← the one everybody skips
```

Two original tasks failed this: "basis points vs percent" and "median averages the
middle two" are cases every model has seen hundreds of times. They were 3/3 for the
*weaker* model, which means they measured nothing.

**Difficulty is a property of the task–model pair, not the task.** Calibrating on a
cheap proxy model systematically mis-targets: tasks that were 0/6 for gpt-4o-mini were
3/3 for GLM-5.2. Calibrate against the model you will actually run, or accept that your
difficulty labels are fiction.

---

## Bugs worth reading about

**`Glob` returned zero matches with `ok: true` in every experiment workspace, for the
life of the project.**

`Glob` prefers `git ls-files --cached --others --exclude-standard` over a filesystem
walk, so results respect `.gitignore`. In a directory that is *itself* git-ignored,
that command exits **0 with zero entries**. An empty list is not `None`, so the git
branch was taken and the tool reported "No files match" — successfully.

Every experiment workspace lives under `.turnloop/experiments/`, which is gitignored.
`Glob` was silently dead in every run ever recorded.

The damage was not the missing tool, it was the misattribution. The model tried
`**/*`, then `**/*.py`, then `**/*.*`, then honestly reported "No files found in the
directory" — and was scored as a *model* failure. A published claim that gpt-4o-mini
scored 50% and sat "dead centre of the discriminating band" was drawn from those runs.
After the fix it scored 100%. That claim is retracted.

The fix returns `None` only when `ls-files` is empty **and** `git check-ignore -q`
confirms the base is ignored — a stricter condition than "empty", because a
non-ignored directory whose files happen to all be individually gitignored must not
suddenly have them exposed.

`Grep` had the same latent blind spot on its `ripgrep` path, fixed the same way. It
never manifested only because `rg` was not installed on the machine and it had been
falling back to a pure-Python walk.

**The Gemini adapter passed a live test and was still broken.**

Gemini returns a `thoughtSignature` alongside each `functionCall` part. Replay the
assistant turn without it and the next request is rejected: `400 — Function call is
missing a thought_signature in functionCall parts`. The adapter decoded the call and
dropped the signature, so **turn two of every tool conversation failed**.

It was verified live before shipping — one text turn, one tool call, both correct —
and that verification is exactly what hid the bug. A single tool call never replays
anything. The failure needs a *second* request carrying the first one's output, which
is the first thing a real agent loop does and the last thing a smoke test does.

Found by setting the harness up as a user would and giving it an ordinary task, not by
testing the adapter. The signature now round-trips through `ToolUseBlock.signature`
and through session JSONL, so a `--resume`d conversation does not hit the same wall.

**0.1.2 shipped with this bug.** It went to PyPI on the strength of the single-turn
check, and anyone who pointed 0.1.2 at Gemini got one working tool call and then a
400. Fixed in 0.1.3.

### `NoActiveWorker`: 283 green tests and a crash on the first keystroke

The config UI shipped with tests covering `/config` — they asserted that dispatching it
returned `open_screen="config"`. All 283 passed. Typing `/config` in the actual TUI
raised `NoActiveWorker: push_screen must be run from a worker`.

Textual's `push_screen_wait` blocks the calling coroutine until the screen is dismissed,
so it must run inside a worker; called from the message pump it would deadlock the pump
it is waiting on, and Textual refuses rather than hanging. `_handle_command` runs on the
pump. So does Textual's action dispatch, which meant the same defect existed at four
call sites, not the one in the traceback — including the provider form reachable from
`tl config --edit`, a path the crash report never touched.

The correct pattern was eleven lines away in the same file: `_permission_worker` is
`@work`-decorated, which is exactly why *its* `push_screen_wait` works.

Two things this cost, both worth stating. The tests verified the layer *below* the bug —
that the command produced the right instruction, never that the app could carry it out.
And **0.1.4 shipped this way**, so `pip install turnloop==0.1.4` crashes on `/config`.
Fixed in 0.1.5. The tests now drive a real app through Textual's `run_test()` pilot and
fail if the decorators are removed.

**A directory that exists on every launch cannot mean "this is a project."**

`find_project_root` treated any ancestor `.turnloop` directory as an explicit project
declaration. But `default_project_dir` creates `.turnloop/` — with a `sessions/`
folder and a `.gitignore` — on every launch, unconditionally, wherever turnloop
happens to start. So the directory's mere existence declared nothing; it was
frequently just the residue of having once run the tool there. Run `tl` once from a
home directory, or a drive root, and it permanently annexed every project beneath it:
sessions, the `.env` layer, `settings.json` and the permission rules all resolved to
the wrong root from then on. Real instance, the author's own machine: `F:\.turnloop`
was created on 2026-08-01, and every `F:\anything\proj\pyproject.toml` since then
resolved its project root to `F:\`.

The invariant that broke was never written down as one, which is exactly why it lasted.
The fix makes the implicit rule explicit: a `.turnloop` counts only if it holds
something a human put there — `settings.json`, `settings.local.json`, `commands/`,
`skills/`, `TURNLOOP.md` — and one holding only an auto-created `sessions/` is ignored.
The home directory is excluded outright on top of that, since `~/.turnloop` is the
user-scope settings location and can never mean "project." The false invariant had
been there since the tool first created `.turnloop/` on launch; fixed in 0.1.6.

**A skill that failed to parse vanished with no error, and the model reported success.**

A live run installed a skill whose frontmatter was markdown-bold — `**name**:
something` — rather than YAML. `yaml.safe_load` returned no usable keys,
`load_skills` hit its "no description" skip, and the skill disappeared: no exception,
no log line, nothing in `/skills`. The model that had just "installed" it reported
success, because from its side the write succeeded and nothing said otherwise.

The bug was not the parser rejecting bad frontmatter — that part is correct. It was
that a rejection and a nonexistent skill looked identical from the outside. `load_skills`
keeps its exact signature, but a new `rejected_skills()` now returns what was dropped
and why, `/skills` reports it, and `CONFIG_LAYOUT` shows the literal YAML frontmatter
block in the system prompt so the model stops guessing at markdown-bold instead.
Fixed in 0.1.6.

**A permission denial that should end in one message looped ten times, and it cost
real money before anyone noticed — because nothing about it failed a test.**

Two defects landed in the same code path with the same symptom. The DENY branch's
message said only `Permission denied: <rule>`; the ASK-decline branch, which is
recoverable, had always carried "Do not retry it" — so the weaker wording sat on the
*permanent* block, the one that most needed it. Fixing the message was not enough on
its own: the ASK path looped despite already carrying that warning, because prompt
wording is a request, not a constraint, and a model under pressure to finish a task
does not reliably honor it.

Measured against a live, non-interactive provider — this is what a passing test suite
cannot see, since nothing here raises or returns the wrong value, it just keeps calling
a tool that keeps getting denied:

- denied `Write .turnloop/settings.local.json`, repeated 10 times → **233,742 input
  tokens, $0.0372**
- after the message fix alone → **10,654 tokens, $0.0017**, one attempt then a correct
  proposal to the user
- a separate ASK-decline loop on `dir`/`mkdir`, before any fix → **199,352 tokens,
  $0.0317**

So wording is now backed by a deterministic guard in `ToolRunner` rather than trusted
alone: identity is tool name plus pydantic-normalized args, scope is the session, one
free retry, and the entry clears the moment the call is allowed or approved — so a
permission granted mid-session un-poisons a call that was denied earlier, instead of
blocking it forever. Fixed in 0.1.6.

**The lesson, eight times over in one project:** an 84-run sweep failing while printing
a tidy report of zeros; a tool returning empty with `ok: true`; a sweep sitting at
10/18 doing nothing for 25 minutes; an adapter passing its live test and failing on
the turn the test never made; a feature with green tests that crashed on its first
keystroke; an implicit invariant that a docstring never stated, so nobody noticed it
was false; a silent failure that a model reported as success; a cost bug that no
assertion could ever have caught, because every individual call did exactly what it
was told. Each was caught only by looking past the summary line — and most of them
only by using the thing instead of testing it.

---

## Windows notes

Both of these cost real time to diagnose from a traceback, so they are handled
explicitly and covered by tests.

**`bash` on PATH is a trap.** On a default install it is
`C:\Windows\System32\bash.exe` — the WSL launcher. With no distribution installed it
exits 255 and prints its error in UTF-16LE. Any `bash.exe` under `%SystemRoot%` is
rejected outright; Git Bash is located via `git`'s install directory, with PowerShell
as the fallback. POSIX is preferred because models write POSIX shell.

**`bash -lc` makes the real command a grandchild.** `terminate()` kills the shell and
orphans the work, which keeps running and holding file locks after a timeout. The
whole tree is killed via `taskkill /T`.

**Never edit source with PowerShell `Get-Content | Set-Content`.** On PS 5.1,
`Set-Content -Encoding utf8` writes a **BOM**, and the read side decodes UTF-8 as ANSI
— em dashes become mojibake. It corrupted a module and 17 fixtures in this repository,
and the BOM then made `ast.parse` fail with "invalid non-printable character U+FEFF".
Files are read with `utf-8-sig` and a BOM is preserved on write, so a stray U+FEFF
never lands invisibly inside an `Edit`'s `old_string` and silently break the match.

File writes preserve the existing newline style and trailing-newline convention, so a
one-line edit in a CRLF checkout does not produce a whole-file diff.

---

## Testing

```bash
pytest                    # 381 tests, no network
ruff check turnloop
mypy turnloop
```

| file | tests | |
|---|---|---|
| `test_permissions.py` | 50 | rule grammar, compound-command splitting, mode enforcement |
| `test_experiments.py` | 36 | runner, graders, report, suite invariants, config resolution |
| `test_providers.py` | 44 | adapters vs recorded `.sse`, image blocks, thought signatures, multi-key rotation on 429/401 |
| `test_tools_files.py` | 28 | Read/Write/Edit/Glob/Grep, encodings, newlines, image detection |
| `test_hooks_mcp_commands.py` | 28 | lifecycle hooks, MCP client, slash commands |
| `test_tui.py` | 25 | Textual snapshots, `/config` and `/mcp add` driven through a real app |
| `test_config.py` | 23 | layering, env overrides, presets, narrow-console guard |
| `test_skills_install.py` | 43 | frontmatter validation, GitHub resolution, agent-mirror tiebreak, concurrent fetch ordering, console caps, non-TTY refusal on an open stdin pipe, drive-root refusal |
| `test_loop.py` | 23 | streaming, truncation, iteration cap |
| `test_compaction.py` | 16 | three tiers, tool_use/tool_result invariant |
| `test_configio.py` | 15 | minimal-diff writes, atomic save, secret-bearing MCP targets, the settings deny rules |
| `test_bash.py` | 14 | shell selection, process-tree kill |
| `test_config_screen.py` | 10 | provider form validation, caps derivation, env indicator leaks |
| `test_websearch.py` | 9 | HTML parsing, backend selection, truncation, failure modes |
| `test_system_prompt_layout.py` | 7 | `CONFIG_LAYOUT` presence, token cost, `minimal` variant omission |

`conftest.py` monkeypatches httpx's transport to raise unless a test is marked
`live` — **network is off at the transport layer**, so a forgotten real call fails
loudly in CI instead of hanging or booting a GPU. `addopts = "-m 'not live' -q"`
deselects live tests by default.

---

## Configuration reference

Precedence, later wins:

```
packaged defaults
~/.turnloop/settings.json
<project>/.turnloop/settings.json
<project>/.turnloop/settings.local.json     (gitignored, personal)
TURNLOOP_* environment variables
CLI flags
```

Dicts merge recursively; lists replace wholesale. Replacing lists is deliberate — a
project that declares its allowlist means *that* list, not that list plus whatever the
user had globally.

Environment overrides: `TURNLOOP_PROVIDER`, `TURNLOOP_MODEL`,
`TURNLOOP_PERMISSION_MODE`, `TURNLOOP_MAX_ITERATIONS`, `TURNLOOP_TOOL_VERBOSITY`,
`TURNLOOP_SYSTEM_VARIANT`.

Project root detection is two-pass. `.turnloop/` is an explicit declaration and always
wins; otherwise the *nearest* build manifest or repo marker wins, with `.git` **last**
in the tuple — a drive root can itself be a git repository, and keying only on `.git`
would make every project's root the entire drive, which then scopes permissions and
sessions wrongly.

API keys come from the environment or a `.env` at the project root. A real environment
variable always wins over the file, and `doctor` reports which key *names* were loaded
— never their values. This applies to `search.api_key_env` exactly as it does to a
provider's: settings files name the variable, never the secret.

Project instructions go in `TURNLOOP.md` (`CLAUDE.md` and `AGENTS.md` are also read).
Slash commands live in `.turnloop/commands/*.md`, skills in
`.turnloop/skills/<name>/SKILL.md` — skills advertise only their name and description
in the system prompt and load their body on demand, which on a 65k window is the
difference between having skills and not.

The system prompt also documents this layout to the model itself, as `CONFIG_LAYOUT`
— 432 tokens, in the cacheable region, omitted from the `minimal` variant. Without it
the model cannot configure the tool it is running inside: asked to add an MCP server
it invented `.turnloop/turnloop.config.json`, and asked to install a skill it wrote to
`examples/` — both real outputs from live runs. With it, both go to the right place
first try.

It also names `tl skills add` explicitly and tells the model *not* to run the install
scripts skill repos publish for other agents. That paragraph was added after watching
three separate sessions try `/plugin marketplace add` as a shell command, then
`clawhub install`, then `git clone` plus `npm install` plus `npm test` — twenty-six
messages ending in "the ponytail skill has been installed", when nothing had been put
anywhere turnloop looks. The same request now costs 11,090 tokens and answers
`tl skills add DietrichGebert/ponytail`. It includes the literal YAML frontmatter a skill needs, since a skill with
markdown-bold frontmatter instead of YAML used to vanish with no diagnostic; `/skills`
now also reports anything found on disk but rejected, with the reason and path — see
[Bugs worth reading about](#bugs-worth-reading-about).

Shell expansion inside a command template (`` !`cmd` ``) goes through the permission
engine like any other Bash call. A markdown file in a repository is not trusted input
just because it is on disk.

### Editing configuration

`tl config --edit` edits the provider table, permission mode, tool verbosity, iteration
cap, search backend, memory, and the three rule lists. `tl mcp` manages MCP servers.
`tl skills add/list/remove` manages skills. The config and MCP screens open inside a
running session as `/config` and `/mcp add`.

**Adding a provider** is a form for `kind`, `model`, `base_url`, and `api_key_env`, with
a live ✓/✗ showing whether that environment variable is currently set. Capabilities are
re-derived from `preset_for(model)` on every save, mirroring what `load_settings` does
for `--model` — without that, a provider added through the UI keeps the previous model's
context window and pricing, and `doctor`'s context-budget line quietly lies. An
`openai_compat` provider without a `base_url` is refused by the form, because
`load_settings` would otherwise raise on the *next* launch, turning a typo into a CLI
that will not start.

Four further properties are structural rather than cosmetic:

**Only the diff is written.** Neither screen dumps the settings model. It writes the
difference against the packaged defaults, merged into whatever the file already holds.
Round-tripping the full model would freeze today's deny list and provider table into
your file permanently, so a future security default could never reach you.

**Writes are atomic and validated.** The edited settings round-trip through
`Settings.model_validate` before anything touches disk, then land via a temp file and
`os.replace`. A malformed settings file makes `load_settings` raise and the CLI exit 2 —
failing closed is right, but only if we never cause it ourselves.

**Secrets are named, never stored.** Provider credentials are edited as the env var
*name*, and the written config has an `api_key_env` key and no `api_key` value at all —
a config UI with a "paste your key here" box writes secrets into a JSON file that gets
committed, which is how keys leak. The ✓/✗ indicator reports the name and a boolean,
never the value, a prefix, or a length. MCP `env` values are literal by protocol necessity, so a server carrying any is
forced to `settings.local.json` and refused a shared file, and the screen creates
`.turnloop/.gitignore` before writing — otherwise a secret-bearing file can be committed
before the first session ever runs.

**No agent-reachable route writes settings.** No tool and no hook mutates a settings
file — which is what makes the deny rules above meaningful instead of decorative.
`/config` and `/mcp add` are slash commands, and that is safe for a specific reason worth
stating: `dispatch_command` has exactly one caller, reached from the prompt Input's
`on_input_submitted`. The model emits tool calls, not typed input, so it cannot reach
that path. The property lives in the dispatch path rather than in the commands, so a
future refactor that routes model output through `dispatch_command` would break it
silently — there is a comment there saying so.

**In-session saves reload only what is safe to reload.** `permission_mode`,
`permissions`, `max_iterations` and `tool_verbosity` are read from the live `Settings`
and `PermissionEngine`, so they take effect immediately. `provider`, `providers`,
`search` and `include_memory` are baked into objects built once at agent construction,
so the confirmation names them as needing a restart. Reporting "saved" while the old
permission mode is still being enforced would be worse than saying nothing.

One honest gap: removing a server deletes it from the one file you chose. `deep_merge`
has no tombstone, so a server also defined in a higher layer stays effective. The screen
says so rather than pretending otherwise.

### MCP servers are a trust decision

Adding an MCP server is the same class of decision as `npm install`, and turnloop cannot
check it for you. Two specific reasons, both inherent to the protocol rather than to this
implementation:

**A stdio server is a program you agreed to run.** `command` and `args` are spawned as a
child process at every launch, with your full user privileges and any `env` you set. The
add form states this and requires an explicit confirmation, because a text field labelled
"command" does not otherwise read like "execute this forever."

**Tool descriptions are untrusted text that enters the model's context.** A server
advertises its own names and descriptions, and those go into the system prompt verbatim.
A hostile or compromised server can put instructions there. Nothing in the protocol
authenticates that text.

What turnloop does about it, which is containment rather than prevention: MCP tools are
`read_only = False` unconditionally (`mcp/adapter.py`), because the protocol has no
read-only annotation worth trusting — so they always prompt in `default` mode and are
refused outright in `plan` mode. A dead or misbehaving server degrades to "that tool is
unavailable" and never blocks the loop. Connections are lazy and every call is bounded.

None of that helps if you install a server that does exactly what it says. Read what you
add.

### Skills are a trust decision too

`tl skills add <owner/repo>` resolves a GitHub repo shorthand, a repo URL, or a direct
raw URL to a `SKILL.md`, and installs it to `.turnloop/skills/<name>/` (or
`~/.turnloop/skills/` with `--user`). The same reasoning as the MCP section above
applies without modification: a skill's body is not metadata, it is text that gets
loaded straight into the model's context the moment the model decides the skill
applies (`commands/loader.py`). A hostile `SKILL.md` can instruct the model exactly
like a hostile MCP tool description can. `add` prints the source URL and the
description and asks for confirmation before writing anything; `--yes` skips waiting
for that confirmation but never skips printing it, and never skips printing what got
installed and from where.

Before writing, `add` parses the frontmatter with the same `parse_frontmatter` the
loader uses and requires a non-empty `description` — the exact rule `load_skills`
enforces when deciding what to advertise (see "A skill that failed to parse vanished
with no error" below). A `SKILL.md` that would silently vanish once installed is
refused before it is ever written, rather than discovered later as "the skill isn't
there."

**Importing from Claude Code.** If `~/.claude/skills/*/SKILL.md` exist and turnloop
has never asked about them, one line appears once, in the TUI, pointing at
`tl skills import` — not a prompt that blocks startup, and not a silent copy.
Importing computes the token cost of each candidate with `rough_tokens` before you
choose (34 skills advertise for about 2,961 tokens of permanent system-prompt
overhead — most of a session's `CONFIG_LAYOUT` segment on its own), lets you import
any subset, and copies the chosen files into turnloop's own skills directory rather
than reading `~/.claude/skills` at runtime — turnloop stays uncoupled from another
tool's directory layout. Whether you say yes or no, the question is recorded in the
user-level settings file so it is never asked again.

---

## Limitations

Stated rather than buried:

- **Image support is verified on two of three encoders.** The `openai_compat` data-URL
  path and Gemini's `inline_data` path both round-trip a real image live; Anthropic's
  is still fixture-only. None of it can be exercised on the primary target, since
  GLM-5.2 W4A16 as deployed is text-only.
- **No binary file support** beyond those image formats. Null-byte sniff refuses the
  rest. Convert via `Bash`.
- **`WebSearch`'s default backend parses HTML, not an API.** DuckDuckGo will rate-limit
  and will eventually change its markup; both are handled as distinct, actionable
  errors rather than a crash, but a keyed backend is the durable choice for anything
  that matters. That trade is deliberate: the default has to work with no key at all.
- **The Anthropic adapter is fixture-tested but has never sent a real request.** It is
  coded against the documented wire format and passes against recorded SSE, which is not
  the same as verified. It is now the only one: Gemini is verified through a real
  multi-turn agent loop on `gemini-flash-latest` — tool calls, edits, file writes and
  images — and OpenAI needs no separate verification because `openai` is
  `kind="openai_compat"`, the same adapter Groq, NVIDIA, Ollama, Blaxel and the
  self-hosted GLM endpoint all run through, making it the most exercised code path in
  the repo. "Verified" here means multi-turn deliberately: a single-turn check on
  Gemini passed while the loop was broken. See [Bugs worth reading about](#bugs-worth-reading-about).
- **`subtle_spec_edge` currently passes for nobody** (0/5 on gpt-4o-mini). The rewrite
  overshot — runs die on collection errors and leave `NotImplementedError`, which is
  failure-to-produce-working-code, not falling for the decoy. It needs another pass.
- **No task in the suite discriminates GLM-5.2 on pass rate**, and more task authoring
  is unlikely to fix it. See [Results](#results).
- **Error-rate findings rest on small counts.** Token findings are solid; ordering by
  error rate is suggestive.
- **Single-turn headless only** — `-p` runs one turn, not a scripted multi-turn session.

---

## License

MIT. Not affiliated with Anthropic.
