# tcode

Want Claude Code's file editing, shell access, and persistent memory without
the subscription? tcode is a terminal coding agent that gives you the same
core capabilities — workspace-rooted file read/write/edit/search, a real
shell, memory that carries across projects — running on free-tier inference
instead.

## Install

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [Groq API key](https://console.groq.com/keys) (free tier available)

```bash
git clone https://github.com/Stevealila/tcode.git
cd tcode
uv tool install --editable .
```

This puts a `tcode` command on your `PATH`, usable from **any** directory.
It operates on whatever directory you launch it from, the same way `claude`
does.

To upgrade after editing the source, no reinstall is needed (`--editable`
points straight at this checkout).

## Configure

Put your Groq API key in a `.env` file, either in a specific project
(`./.env`) or globally (`~/.tcode/.env`) so every project picks it up:

```bash
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b     # optional, this is the default
GROQ_REQUEST_LIMIT=50              # optional, caps tool-call round-trips per turn
TCODE_ALLOWED_COMMANDS=git,pytest  # optional, see "Shell access" below
TCODE_MAX_RPM=30                   # optional, client-side requests/min throttle (0 disables)
TCODE_WEB_SEARCH=1                 # optional, web search/fetch tools (default on, 0 disables)
TAVILY_API_KEY=tvly-...            # optional, better web search — see "Web search" below
```

Get a key at <https://console.groq.com/keys>. Real environment variables
exported in your shell always take precedence over both `.env` files;
a project's `.env` takes precedence over the global one.

Good Groq models for this kind of agentic/tool-calling work: `openai/gpt-oss-120b`
(default, 131K context) or `qwen/qwen3.6-27b` (an alternative worth
comparing on your workload). Check `GET /openai/v1/models` against your own
key before picking one — Groq's catalog changes, and a model that's current
today may be gone in a few months.

## Use

```bash
cd ~/some/project
tcode                       # interactive session, rooted at this directory
tcode "fix the failing test in tests/test_x.py"   # one-shot, prints and exits
tcode -c                    # continue the last session in this directory
tcode --sessions            # list saved sessions for this directory
tcode --model qwen/qwen3.6-27b
```

Slash commands inside an interactive session:

- `/help` — list commands
- `/clear` — clear the in-memory conversation (keeps the saved session file until you send another message)
- `/memory` — show what's currently in the global memory notebook
- `/sessions` — list saved sessions for this project
- `/exit`, `/quit` — leave (Ctrl-D also works)

## How state is laid out

Global state lives under `~/.tcode`, keyed per-project by the absolute
path you ran `tcode` from.

```
~/.tcode/
  memory/                       # global, persists across every project
  projects/
    -home-alice-some-project/
      sessions/                 # this project's conversation history
      steps/                    # harness StepPersistence execution log
```

Memory is deliberately **global**, not per-project: it's where the agent
keeps what it has learned about *you* (conventions you prefer, corrections
you've given it), so that carries into every project. Conversation history
and step logs are per-project, so switching directories switches context
the way switching repos should.

## Shell access

By default the agent's shell is unrestricted except for a short denylist of
genuinely destructive commands (`rm`, `rmdir`, `dd`, `mkfs`, `format`,
`shutdown`, `reboot`, `halt`, `poweroff`, `init`). Everything else,
including things like `gh`, `npm`, or `docker`, works with no setup.

If you'd rather lock a project down to a specific, small set of commands,
set `TCODE_ALLOWED_COMMANDS` (comma-separated) in that project's `.env`.
This flips to strict allowlist mode, blocking everything not listed.

Either way, this is a guardrail against accidents, not a sandbox boundary:
allowed commands can still spawn arbitrary subprocesses. Don't point this at
a directory you don't trust an agent to have shell access to.

For a task that processes untrusted content and has no legitimate reason to
run commands at all (summarizing scraped web pages, say), an allowlist
still leaves a shell for a prompt-injected page to try to abuse. Set
`TCODE_SHELL=0` to omit the Shell capability entirely — no `run_command`
tool exists for the model to be tricked into calling, not even a
restricted one.

## Web search

On by default, no API key needed: a search tool (DuckDuckGo) and a fetch
tool that reads a URL and distills it down to whatever you actually asked
for, rather than dumping the raw page. Set `TCODE_WEB_SEARCH=0` to turn both
off for a project that shouldn't touch the live web.

DuckDuckGo's free-text search is the zero-setup default, but its snippets
are often stale for "what's the current X" questions — static page copy with
an old example number or date baked in, not what the page shows live. Set
`TAVILY_API_KEY` to switch the search tool to [Tavily](https://app.tavily.com)
instead: a search API built for LLM/agent use (cleaner extracted content,
a finance/news topic mode), with a free tier (1,000 searches/month, no card)
that fits the same "works without a subscription" goal as everything else
here. This mirrors how other agent harnesses (OpenClaw, for one) pick a
search backend: prefer a real search API when a key for one is configured,
fall back to a scrape when it isn't.

The fetch tool always distills, regardless of which search backend found the
URL: the page is fetched and converted to markdown, then a second, cheap
Groq call (`openai/gpt-oss-20b`) reads it against the specific question the
model asked for and returns just that — the same two-stage shape Claude
Code's own `WebFetch` uses (a full fetch, then a small model extracting the
answer), reverse-engineered from its public tool description since this
harness has no equivalent server-side infrastructure to lean on. It also
means the model doesn't get to hallucinate a citation from a page that
failed to load or a JS-only page that fetched as boilerplate — that model
is instructed to say plainly when the answer isn't on the page, rather than
pass along noise for the (weaker, more confident) main model to run with.

## Rate limits & context management

Groq's free tier is tight: 8,000 tokens/minute and 30 requests/minute for
`openai/gpt-oss-120b` (and gpt-oss-20b; switching models doesn't buy more
TPM. `llama-3.1-8b-instant` is actually lower, at 6,000). Three things keep
a session working within that:

- **`ToolOutputLimits`** caps any single tool result (a big `ls -R`, a
  large file `cat`) as it enters the conversation. `TCODE_TOOL_OUTPUT_LIMIT`
  (default 2000 characters).
- **`ClearToolResults`** prunes older tool results once the conversation
  has grown past a token budget, keeping the last few intact.
  `TCODE_CLEAR_AFTER_TOKENS` (default 3000) and `TCODE_KEEP_TOOL_PAIRS`
  (default 3).
- **A client-side RPM throttle** (`TCODE_MAX_RPM`, default 30, shared
  across every `tcode` process via `~/.tcode/rpm_state` since Groq enforces
  this per account, not per process) waits out the window instead of
  firing a request that's just going to get a 429.

Groq seems to cache the system prompt and tool schemas server-side; repeat
turns showed most `input_tokens` coming back as `cache_read_tokens`, which
don't count against the TPM budget. The real pressure is the new content
each turn (conversation growth, tool output), which is what the two
compaction knobs above target. `ClearToolResults` rewriting old content
does invalidate the cache from that point on, so the turn it fires on pays
full price once.

If you're on a higher tier, raise or disable these (`TCODE_MAX_RPM=0`,
larger `TCODE_CLEAR_AFTER_TOKENS`). The defaults are tuned for the free
tier, not a hard ceiling.

## Large files and batch tasks

`read_file` truncates a large file from the *middle* past a few thousand
characters — real content just isn't there for the model to see, with
nothing telling it that. For a file where you need a specific answer
rather than its literal bytes (a long log, a research note, a data dump),
use `read_and_distill(path, prompt)` instead: it reads the file in full,
then a second, cheap model extracts just what's relevant to `prompt`. Keep
using `read_file` for anything you need to see or edit verbatim.

For a task that means processing *many* such files — read everything in a
directory and reduce it to one summary, say — don't ask a single tcode
turn to do it, even with `read_and_distill` available. Tested this
exhaustively: a turn given a list of items and told "process all of them"
reliably stops early, past a surprisingly small N, and confabulates a
plausible-sounding reason (almost always some form of "out of token
budget," which checking the actual usage never bears out). This isn't a
context-window problem — it reproduces even when each individual item is
small. It's the model deciding a multi-item task looks too big and bailing
before actually running out of anything.

The fix is external, not a flag: drive it with a script that calls `tcode`
once per item — a plain shell loop, one headless call per file — rather
than one call handling a list. A three-stage version scales to any input
size:

```bash
# 1. map — one tcode call per source file, never a batch
for f in data/*.md; do
  tcode "Use read_and_distill on $f to extract <what you need>. \
    Write ONLY that to summaries/$(basename "$f") via write_file."
done

# 2. combine — plain shell, no model call, can't fail
cat summaries/*.md > combined.md

# 3. reduce — one tcode call, one file in
tcode "Read combined.md and write the final summary to output.md."
```

If a single combine step is still too large for one reduce call, repeat
the map step one level up (per-group digests, then combine those) rather
than asking one turn to read everything at once — validated up to 39 files
across 5 groups this way. Each map-step call is independent, so a script
driving this should check the expected output file exists after each call
and retry once if not, rather than assume success; individual calls are
reliable but not perfect.

## Extending

Everything is one file each in `src/`:

- `config.py` — `.env` loading, on-disk layout
- `agent.py` — which harness capabilities are wired in
- `web.py` — the web search/fetch tools (see "Web search" above)
- `guardrails.py` — technical backstops for instructions the model doesn't
  reliably follow on its own
- `ratelimit.py` — shared cross-process RPM throttle
- `sessions.py` — conversation save/resume
- `ui.py` — terminal rendering
- `cli.py` — argument parsing and the REPL loop

To add a capability (a Playwright browser, spend limits, skills), import it
from `pydantic_ai_harness` (or `pydantic_ai.capabilities`, like the built-in
web search) in `agent.py` and add it to the `capabilities=[...]` list.
