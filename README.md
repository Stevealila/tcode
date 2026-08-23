# tcode

Want Claude Code's file editing, shell access, and persistent memory without
the subscription? tcode is a terminal coding agent that gives you the same
core capabilities — workspace-rooted file read/write/edit/search, a real
shell, memory that carries across projects — running on free-tier inference
instead. It currently runs on [Groq](https://groq.com), built with
[Pydantic AI](https://ai.pydantic.dev)'s [harness](https://ai.pydantic.dev/harness/)
capability library — early days, so the underlying stack may change as the
project grows.

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
```

Get a key at <https://console.groq.com/keys>. Real environment variables
exported in your shell always take precedence over both `.env` files;
a project's `.env` takes precedence over the global one.

Good Groq models for this kind of agentic/tool-calling work: `openai/gpt-oss-120b`
(default, 131K context, fast, cheap), `moonshotai/kimi-k2-instruct-0905`
(bigger, stronger at long agentic tool-use, 256K context), or
`llama-3.3-70b-versatile`.

## Use

```bash
cd ~/some/project
tcode                       # interactive session, rooted at this directory
tcode "fix the failing test in tests/test_x.py"   # one-shot, prints and exits
tcode -c                    # continue the last session in this directory
tcode --sessions            # list saved sessions for this directory
tcode --model llama-3.3-70b-versatile
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

## Extending

Everything is one file each in `src/`:

- `config.py` — `.env` loading, on-disk layout
- `agent.py` — which harness capabilities are wired in
- `ratelimit.py` — shared cross-process RPM throttle
- `sessions.py` — conversation save/resume
- `ui.py` — terminal rendering
- `cli.py` — argument parsing and the REPL loop

To add a capability (web search, a Playwright browser, spend limits,
skills), import it from `pydantic_ai_harness` in `agent.py` and add it to
the `capabilities=[...]` list.
