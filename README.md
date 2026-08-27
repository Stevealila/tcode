# tcode

tcode is a fast, small terminal coding agent that wraps free-tier LLMs. It
gives you workspace-rooted file read/write/edit/search, a real shell, web
search, and memory that carries across projects, all running on inference
you can get without a subscription. Groq is the zero-setup default; Google
Gemini also works today, and other backends are a small change to add.

It runs two ways:

- **Interactive**, as a lightweight coding assistant in the terminal.
- **Scripted and non-interactive**, invoked as a subprocess that reads
  files, logs, or data under a workspace and returns a parseable answer or
  a written report on stdout. Most of the harder work here (a clean stdout
  contract, read-only and write-scope guardrails, an independent
  verification pass, batch map-reduce over many files) exists to make that
  second mode trustworthy when no human reads every run. A representative
  use: point it at a directory of notes and data with a standing prompt,
  run it on a schedule, and parse a structured decision out of stdout.
  Nothing about the tool is tied to any one domain.

Honest status: tcode has not yet been pushed hard on deep, sprawling
coding work, and getting it there is likely future effort. What is solid
today is the scripted single-task and batch-reduce path, plus everyday
interactive edits.

## Install

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [Groq API key](https://console.groq.com/keys) (free tier available)

Built and run on Linux. It has never been tested or packaged for Windows,
and the `--sandbox` path is Linux-only by design. macOS is untested but
likely close; Windows will probably need work.

```bash
git clone https://github.com/Stevealila/tcode.git && cd tcode && uv tool install --editable .
```

This puts a `tcode` command on your `PATH`, usable from **any** directory.
It operates on whatever directory you launch it from.

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
TAVILY_API_KEY=tvly-...            # required for web search/fetch, see "Web search" below
TCODE_CHECK_CITATIONS=1            # optional, reject a write citing a source file that doesn't exist (default off)
TCODE_DISTILL_MODEL=openai/gpt-oss-120b  # optional, a stronger Groq model for the internal distillation pass (default gpt-oss-20b)
TCODE_REQUIRE_CITATION_FOR="[CONFIRMED],[MEASURED]"  # optional, reject a write using one of these tags with no citation on that line
TCODE_EXPERT_MODEL=google:gemini-2.5-pro  # optional, a stronger model the agent can delegate one hard sub-problem to, see "Escalating to a stronger model" below
```

Get a key at <https://console.groq.com/keys>. Real environment variables
exported in your shell always take precedence over both `.env` files;
a project's `.env` takes precedence over the global one.

Good Groq models for this kind of agentic/tool-calling work:
`openai/gpt-oss-120b` (default, 131K context) or the smaller, faster
`openai/gpt-oss-20b`. Check `GET /openai/v1/models` against your own key
before picking one: Groq's catalog changes, and a model that's current
today may be gone in a few months. tcode refuses to run Qwen models
(unreliable structured output in practice); see
`_BANNED_MODEL_SUBSTRINGS` in `src/config.py`.

### Groq free tier vs. developer tier

The free tier is enough to try tcode and to run short tasks, but its
tokens-per-minute (TPM) ceiling and per-request limits are low. A longer
interactive session, a large file, or a big `--reduce` batch will reach
them: requests get throttled hard by the RPM/TPM pacing, and a turn whose
history has grown past the free per-request token budget can fail outright
even when it is still well under the model's context window. If you use
tcode for real work on Groq, upgrade the Groq account to the paid
developer tier. It raises the TPM and request ceilings by a wide margin
and clears most of these stalls. Nothing in tcode's configuration changes,
the same key just has more headroom. The context-management pipeline below
helps regardless, but it cannot manufacture throughput the account tier
doesn't grant.

### Other providers

Groq stays the zero-setup default, but `--model` also accepts
`provider:model_id` to run a turn on a different backend when you have a
key for one and Groq's own current catalog isn't the right fit for a given
task: `tcode --model google:gemini-2.5-flash "..."`. Currently wired up:

| provider prefix | env var          | example model         |
|------------------|-------------------|------------------------|
| *(none)*          | `GROQ_API_KEY`    | `openai/gpt-oss-120b`  |
| `google`          | `GOOGLE_API_KEY`  | `gemini-2.5-flash`     |

`GROQ_API_KEY` stays required either way: the large-file distillation
pass (see "Large files and batch tasks" below) and `--verify`'s
default verifier/compare passes are Groq-only by design, independent of
whichever provider the main turn runs on. Each provider paces its own
requests against its own account-wide rate limit (Groq: 30 RPM default;
Google AI Studio's free tier: 15 RPM default), tracked separately per
provider under `~/.tcode/`, and `TCODE_MAX_RPM` still overrides Groq's
specifically. Adding another provider is a small, self-contained change;
see `_build_model` in `src/agent.py`.

### Escalating to a stronger model

Set `TCODE_EXPERT_MODEL` (same `provider:model_id` format as `--model`) to
give the agent a second, stronger model it can reach for mid-run without
switching the whole session over to it. This adds an optional `model`
argument to its existing `delegate_task` tool. Normally every delegated
sub-task runs on that sub-agent's own default model, but with
`TCODE_EXPERT_MODEL` set, the agent can pass `model='expert'` to route one
specific delegation (a genuinely hard sub-problem, not routine exploration)
to it instead, at a higher reasoning effort. Unset (the default), the
`model` argument doesn't exist at all and every delegation behaves exactly
as before. This is opt-in, not a second model client running by default.

### Reasoning effort for the main loop

`--effort {low,medium,high}` (or `--think`, shorthand for `--effort high`,
or `TCODE_EFFORT`) sets the reasoning effort of the primary model for that
run: a per-task knob, dial it up for a hard problem, leave it off for a
lookup. This is separate from `TCODE_EXPERT_MODEL` above, which only
affects sub-agent delegations; `--effort` affects the main loop itself.
More effort means deeper reasoning but more tokens and more wall-clock for
the same work. Currently applied on Groq models
(`groq_reasoning_effort`); a no-op on other providers until their
equivalent knobs are wired.

## Use

```bash
cd ~/some/project
tcode                       # interactive session, rooted at this directory
tcode "fix the failing test in tests/test_x.py"   # one-shot, prints and exits
tcode -c                    # continue the last session in this directory
tcode --sessions            # list saved sessions for this directory
tcode --model openai/gpt-oss-20b
tcode --quiet "summarize today's log" > answer.txt   # one-shot, clean stdout
```

Slash commands inside an interactive session:

- `/help` list commands
- `/init` explore the project and write a `TCODE.md` instruction file for it
- `/clear` start fresh: drop the conversation and the `-c` pointer (per-turn archives stay in `/sessions`)
- `/context` show how full the model's context window is (a gauge, the live token count, the compaction thresholds)
- `/compact [focus]` summarise the conversation now to reclaim window; an optional `focus` steers what the summary keeps
- `/memory` show what's currently in the global memory notebook
- `/sessions` list saved sessions for this project
- `/model` show the current model and the Groq catalogue (live, with a
  static fallback when offline); `/model <id>` switches for the rest of the
  session, fuzzy match ok (`/model 120b`, `/model mini`), or
  `/model google:<id>` / `/model zai:<id>` for another backend. Session-only;
  `--model` / `GROQ_MODEL` still set the startup default. A rejected switch
  (banned model, missing provider key, missing extra) leaves the session on
  its current model.
- `/skills` list skills found in `~/.tcode/skills`
- `/skill <name>` load `~/.tcode/skills/<name>.md`; its content is
  prepended to your *next* message, not sent immediately. Write reusable
  prompt snippets there once and pull one in whenever it's relevant.
- `/exit`, `/quit` leave (Ctrl-D also works)

## Scripted / non-interactive use

One-shot mode normally interleaves tool-call activity, a usage footer, and
notices with the model's actual answer on stdout: fine to watch, awkward
to parse. `--quiet` (`-q`) routes all of that to stderr instead, so stdout
is exactly the model's text output and nothing else. This is for a caller
that runs `tcode` as a subprocess and parses stdout (extracting a JSON
decision, say), not a human watching the terminal.

Pair it with `TCODE_READONLY=1` for a task that should look but never
touch: it removes `write_file`/`edit_file`/`create_directory` from the
model's tool list entirely (same mechanism as `TCODE_SHELL=0` for Shell,
see "Shell access" below), leaving `read_file`/`list_directory`/
`find_files`/`read_and_distill` untouched. Combine with `TCODE_SHELL=0`
and `TCODE_WEB_SEARCH=0` for a fully read-only, no-side-effects session:
diagnose, decide, report, touch nothing.

For a task that reads broadly but should only ever write in one place
(the model decides *where* within the workspace, not just *whether*, and
prompt convention alone, "only write under X", is a request an untrusted
input like a fetched page could talk it out of), set
`TCODE_WRITE_SCOPE=path/prefix` (comma-separated for more than one). Every
`write_file`/`edit_file`/`create_directory` call is checked against it
before the write happens and rejected (with a message the model can act
on) if it isn't; reads are unaffected. This is a technical backstop, not a
convention: it doesn't consult `.gitignore` and isn't fooled by `../`, so
it holds even for a target a caller-side `git status` diff would never see
(a gitignored directory) or shouldn't trust diffing at all (a path some
other process legitimately rewrites concurrently).

Two more write-time checks, for a caller whose whole job is producing a
report that cites source files it read this run (a research rollup, say):

- Always on: any write whose text contains a URL that looks like a garbled
  copy of one a tool result actually returned this run (truncated
  mid-word, trailing an ellipsis) is rejected and the model is asked to
  copy it in full or drop it, rather than silently landing a broken link.
- `TCODE_CHECK_CITATIONS=1` (opt-in, since a normal coding session
  legitimately proposes paths that don't exist *yet*, which this can't
  tell apart from a fabricated citation): any write whose text cites a
  backtick-quoted path (`` `profiles/example/daily/2026-08-23.md` ``) that
  doesn't actually exist under the workspace is rejected the same way.
  Catches a model inventing a plausible-sounding source it never read.
- `TCODE_REQUIRE_CITATION_FOR="[TAG1],[TAG2]"` (opt-in, no built-in
  vocabulary of its own): for a caller whose own prompt defines confidence
  tags and asks the model to use them, any write where a configured tag
  appears on a line with no URL or cited file path anywhere on that same
  line is rejected. Catches a model stating a fabricated fact under its
  own highest-confidence label with nothing backing it up at all, not just
  a wrong or mangled citation the two checks above already catch.

The prompt can come via stdin instead of an argument too (`echo "..." |
tcode`, or a subprocess `input=` from another language), useful once the
prompt is long enough that shell quoting gets awkward, or the caller
already has it as a string in memory rather than something to shell-escape.

### `--verify`: an independent second opinion before you trust the answer

Clean output and correct scoping don't do anything about a *confidently
wrong* answer: a model can call every tool correctly and still misread
its own inputs, and nothing about `--quiet`/`TCODE_READONLY` catches that.
For an answer something else is going to act on without a human reading it
critically first (a script parsing a decision out of stdout), that's the
failure mode that actually matters.

`--verify` runs three passes instead of one: the real answer, an
independent re-derivation from the same prompt (a different Groq model by
default, deliberately not just another sample from the same model, which
research on this specifically shows doesn't help when the error is a
systematic blind spot rather than random noise, or `TCODE_VERIFY_CMD`, an
external command for a cross-provider verifier when one's available), and
a cheap pass judging whether they actually agree on the substance. Agree:
stdout gets the real answer, same contract as `--quiet`. Disagree: stdout
is empty and the exit code is 2, deliberately not an error message or a
hedge, so a caller already treating "no parseable answer" as "this attempt
failed, try the next tier" (a retry, a fallback provider) handles
disagreement correctly with no extra logic of its own.

```bash
echo "$PROMPT" | tcode --verify --model openai/gpt-oss-120b
TCODE_VERIFY_MODEL=openai/gpt-oss-20b tcode --verify "..."  # explicit verifier model
TCODE_VERIFY_ADVISORY=1 tcode --verify "..."                # always print primary; verdict to stderr
TCODE_VERIFY_CMD="<external llm CLI, read-only tools only>" \
  tcode --verify "..."                                     # external/cross-provider verifier
```

Best suited to a narrow, structured decision (classify this, decide this)
where "agreement" has a clear meaning: two valid answers to an open-ended
prompt can differ completely in wording and both be fine, which reads as
disagreement to the comparison pass. `TCODE_VERIFY_TIMEOUT` (seconds,
default 150) bounds the external-command path; `-c`/`--continue` is
ignored with `--verify` since verification needs an independent
re-derivation, not a continued conversation.

## How state is laid out

Global state lives under `~/.tcode`, keyed per-project by the absolute
path you ran `tcode` from.

```
~/.tcode/
  memory/                       # global, persists across every project
  TCODE.md                      # global instructions, loaded in every project
  projects/
    home-alice-some-project/
      sessions/                 # this project's conversation history
      steps/                    # harness StepPersistence execution log
```

Memory is deliberately **global**, not per-project: it's where the agent
keeps what it has learned about *you* (conventions you prefer, corrections
you've given it), so that carries into every project. Conversation history
and step logs are per-project, so switching directories switches context
the way switching repos should. `TCODE_MEMORY=0` omits the notebook
entirely (no `write_memory`/`read_memory`, nothing injected), right for a
short headless run with no durable facts to keep, and one that a weak model
would otherwise pollute by pasting its own injected context back in. When
memory *is* on, a guardrail (`memory_write_is_sane`) blocks exactly that
paste-back and over-long dumps.

Separately, tcode also auto-loads Markdown instruction files as static
context. Its own file is `TCODE.md`. For compatibility with other coding
agents it also reads `CLAUDE.md` and `AGENTS.md` when present. Where these
are looked for:

- **Global:** `~/.tcode/TCODE.md`, applied in every project, the place for
  standing preferences that aren't tied to one repo.
- **Walk-up:** every directory from the one you launched `tcode` from up
  through your home directory, so a monorepo-root file is picked up even
  when you run `tcode` from a subdirectory.
- **Nested:** a subdirectory's instruction file is pulled in the first time
  the agent reads or lists that directory during the session.
- **Personal, uncommitted:** `TCODE.local.md` alongside `TCODE.md`, for
  overrides you don't want to check in (add it to `.gitignore`).

When more than one applies, precedence runs global < ancestor < workspace,
and within a single directory `TCODE.local.md` > `TCODE.md` >
`CLAUDE.md`/`AGENTS.md`: the file closest to where you're working, and
most specific to tcode, wins.

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
`TCODE_SHELL=0` to omit the Shell capability entirely: no `run_command`
tool exists for the model to be tricked into calling, not even a
restricted one.

Why this gap matters here specifically: tcode ships shell access on by
default and runs on open-weight models that are, in general, less tested
against resisting injected instructions than frontier models. The
denylist/allowlist guardrail is prefix-based, not a kernel sandbox. A page
fetched mid-session that tries to talk the agent into running something is
the live threat this guards against, and it's a threat a weaker model is
plausibly more susceptible to, not less. If you're pointing tcode at
genuinely untrusted content, use `TCODE_SHELL=0` or `TCODE_WRITE_SCOPE`
above, or run tcode inside a container or a bubblewrap/firejail wrapper for
an actual OS-level boundary. None of this guardrail layer is a substitute
for one.

For the last of those, `--sandbox` (or `TCODE_SANDBOX=1`) is a built-in
version: on Linux, tcode re-execs itself inside `bwrap` (preferred) or
`firejail` with only the workspace and `~/.tcode` writable and the rest of
the filesystem read-only, so a write outside those trees fails in the
kernel no matter what the model was talked into. It's a no-op with a
warning on non-Linux or when neither tool is installed. Network stays
reachable (the model API needs it), so this contains filesystem writes,
not exfiltration; `TCODE_SHELL=0` is still the right choice when the
content itself is untrusted.

## Web search

On by default. Both the search tool and the fetch tool go through
[Tavily](https://app.tavily.com), a search/extract API built for LLM/agent
use (cleaner extracted content than SERP snippets, a finance/news topic
mode), with a free tier (1,000 credits/month, no card) that fits the same
"works without a subscription" goal as everything else here. `web_fetch`
adds a second stage on top: Tavily's extracted markdown plus your specific
question go to a cheap model that returns just the answer, or says plainly
when the page doesn't have it, rather than dumping the raw page for a weak
model to guess from.

`TAVILY_API_KEY` is **required for web**: without it both tools are simply
not registered (the banner says so), the same state as `TCODE_WEB_SEARCH=0`.
The earlier zero-setup DuckDuckGo search and markdownify fetch are gone:
DDG's snippets were routinely stale for "what's the current X" questions
(old example numbers baked into static copy), and raw markdownified HTML
gave a weak model boilerplate to hallucinate from.

The fetch tool always distills, regardless of which search backend found the
URL: the page is fetched and converted to markdown, then a second, cheap
Groq call (`openai/gpt-oss-20b`) reads it against the specific question the
model asked for and returns just that, a two-stage shape (a full fetch,
then a small model extracting the answer) rather than handing a weak model
a whole page to guess from. It also means the model doesn't get to
hallucinate a citation from a page that failed to load or a JS-only page
that fetched as boilerplate: that small model is instructed to say plainly
when the answer isn't on the page, rather than pass along noise for the
(weaker, more confident) main model to run with.

## Rate limits & context management

Two different limits get called "context" and tcode keeps them apart:

- the model's context **window**, a hard per-request size ceiling
  (`openai/gpt-oss-*` on Groq is 131K). Exceeding it is a hard error.
- Groq's tokens-per-**minute** rate limit, a throughput ceiling over time,
  handled by the **RPM throttle** (`TCODE_MAX_RPM`, default 30, shared
  across every `tcode` process via `~/.tcode/rpm_state` since Groq enforces
  it per account). Groq also caches the system prompt and tool schemas
  server-side, so repeat turns re-pay only for new content.

On the Groq free tier both of those bite sooner than the numbers above
suggest (see "Groq free tier vs. developer tier"); the pipeline here
manages the window, not the account's throughput.

The compaction pipeline (`src/context.py`) targets the *window*, on a
**fraction** of it resolved per request, so it stays correct after a
`/model` switch to any size of model. Cheapest first, escalating only as far
as it must:

- **`ToolOutputLimits`** head/tail-truncates any single tool result over
  `TCODE_TOOL_OUTPUT_LIMIT` characters (default 8000) as it enters history.
- **`ClampOversizedMessages`** truncates one runaway message part (a giant
  tool call emitted as text), `TCODE_MAX_PART_TOKENS` (default 24000).
- **`DeduplicateFileReads`** blanks a file read superseded by an identical
  later read of the same range. Near-lossless, every request.
- **`ClearToolResults`** blanks older tool results once history passes
  `TCODE_CONTEXT_COMPACT_FRACTION` of the window (default 0.80), keeping the
  last `TCODE_KEEP_TOOL_PAIRS` (default 6).
- **`SummarizingCompaction`**, only if clearing wasn't enough: older turns
  summarised by a small Groq model (`openai/gpt-oss-20b`), the last
  `TCODE_CONTEXT_KEEP_MESSAGES` (default 12) and user turns kept verbatim.

`WarnNearLimits` tells the model to wrap up near
`TCODE_CONTEXT_WARN_FRACTION` (default 0.92). `/context` shows how full the
window is; `/compact [focus]` summarises now; `/clear` starts fresh.

Set `TCODE_CONTEXT_WINDOW` for a model genai-prices doesn't have a window
for (some google/zai ids), otherwise it falls back to a 200K assumption.
All fractions floor at 0.25 (below that, fixed overhead alone exceeds the
target and compaction just thrashes).

tcode's system prompt deliberately does **not** encourage the model to run
independent tool calls in parallel. On a generous per-minute budget,
parallel calls trade idle capacity for latency. Here the binding
constraint is the shared RPM ceiling above: parallel calls would just burn
through that same scarce budget faster, not use up idle headroom. Don't add
this back without accounting for the throttle it would be racing against.

## Large files and batch tasks

`read_file` truncates a large file from the *middle* past a few thousand
characters: real content just isn't there for the model to see, with
nothing telling it that. For a file where you need a specific answer
rather than its literal bytes (a long log, a research note, a data dump),
use `read_and_distill(path, prompt)` instead: it reads the file in full,
then a second, cheap model extracts just what's relevant to `prompt`. Keep
using `read_file` for anything you need to see or edit verbatim.

For a task that means processing *many* such files (read everything in a
directory and reduce it to one summary, say), don't ask a single tcode
turn to do it, even with `read_and_distill` available. Tested this
exhaustively: a turn given a list of items and told "process all of them"
reliably stops early, past a surprisingly small N, and confabulates a
plausible-sounding reason (almost always some form of "out of token
budget," which checking the actual usage never bears out). This isn't a
context-window problem: it reproduces even when each individual item is
small. It's the model deciding a multi-item task looks too big and bailing
before actually running out of anything.

`--reduce PATTERN` handles this internally rather than pushing the
workaround onto whatever's calling tcode. A caller with a batch task
should be able to invoke tcode the same way it would for a single file and
let tcode sort out how many internal steps that actually takes, the same
principle `--verify` follows for a different problem:

```bash
tcode --reduce "data/*.md" "Summarize each file's key points, then write one combined report."
tcode --reduce "logs/**/*.log" "Find every ERROR line and group them by cause."
```

Internally: one distillation call per matched file, run concurrently
(`throttle()` still paces the underlying requests, this isn't a way
around the rate limit, just not waiting on each file before starting the
next); grouped by parent directory and digested first if there are more
files than one reduce turn has been tested reliable for; one final,
ordinary tcode turn (full capabilities, can `write_file` if the prompt
asks for that) over the collected result. Validated end-to-end on 56 real
files across 5 directories, both the per-directory grouping and a
same-directory group large enough to need further chunking.

`@listfile` (a path prefixed with `@`, read as one file/glob per line,
`#`-comments and blank lines ignored) takes an explicit file list instead
of a glob, for selection logic a glob can't express (a caller's own
date-range filter, say). The caller keeps owning that logic and just hands
tcode the resulting list; tcode doesn't need to learn what "the last 14
days" means for someone else's files.

Every distillation call in this pipeline (the per-file/per-group map and
digest passes above, plus `read_and_distill` and `web_fetch`'s own
page-summary step) runs on the same small, cheap Groq model by default
(`openai/gpt-oss-20b`), regardless of which model `--model` points the
primary turn at. That's fine for most tasks, but for one whose whole point
is precise multi-document extraction and citation, this is a hidden
bottleneck: everything the primary model ever sees about a large file or a
`--reduce` batch went through this smaller model first, so a misread there
poisons every downstream turn no matter how good the primary is.
`TCODE_DISTILL_MODEL=<groq-model-id>` points it at something stronger
instead (a bare Groq model id, not `provider:model_id`, this pass is
Groq-only by design, same constraint `TCODE_VERIFY_MODEL` already has).

After the final reduce turn, a cheap heuristic checks whether every input
group (files grouped by parent directory, same signal used above) actually
surfaces in the final text at all, by its own distinguishing path
keywords, not just any mention. If a whole group is missing, a notice
flags it rather than staying silent: it can't tell "genuinely nothing new
to report" from "silently dropped during synthesis," but a caller or a
human watching the logs can at least see the omission and go check, rather
than trusting an answer that reads as complete when it isn't.

### `--write PATH`: guaranteeing the answer actually lands on disk

At large output sizes, an open-weight tool-calling model can fail to make
its final `write_file` call cleanly: a garbled tool name from malformed
function-call encoding, or the model printing the call as JSON prose
instead of actually issuing it, usually truncated by the same length
pressure that caused it. tcode retries the whole turn a bounded number of
times when this happens (distinct from a single tool call's own retry
budget, this restarts the turn itself) and, when a turn does produce real
text, never discards it just because the tool call built on top of it
broke. Neither of those guarantees the write happens at all, though: a
model that fails the same way on every retry still won't produce a file.

For a caller that already knows the one path a run is supposed to write
(a report at a fixed location, say), `--write PATH` closes that gap:

```bash
tcode --reduce "data/*.md" --write out/summary.md "Summarize each file, then write one combined report to out/summary.md."
```

If `PATH`'s mtime hasn't changed by the time the turn ends, tcode writes
the model's own final answer there directly, so the caller's file exists
either way (via a real `write_file` call, or via this fallback) without
needing to detect and paper over the failure itself. Works the same way
for a plain one-shot call, not just `--reduce`.

## Extending

Everything is one file each in `src/`:

- `config.py` `.env` loading, on-disk layout
- `models.py` the Groq model catalogue lookup behind `/model`
- `context.py` the compaction pipeline plus the `/context` / `/compact` helpers
- `agent.py` which harness capabilities are wired in
- `web.py` the web search/fetch tools (see "Web search" above)
- `files.py` / `distill.py` the large-file distillation tool and its
  shared model, also used by `web.py`
- `verify.py` `--verify`'s independent-verifier decision mode
- `reduce.py` `--reduce`'s internal map-reduce over many files
- `guardrails.py` technical backstops for instructions the model doesn't
  reliably follow on its own
- `model_quirks.py` wraps the model to paper over provider glitches (today:
  gpt-oss on Groq emitting `functions/read_file` for a tool named `read_file`)
- `ratelimit.py` shared cross-process RPM throttle
- `sessions.py` conversation save/resume
- `skills.py` `/skill <name>` (see "Use" above)
- `ui.py` terminal rendering
- `runner.py` the single-turn execution loop (`cli.py` and `reduce.py`
  both use it)
- `cli.py` argument parsing and the REPL loop

To add a capability (a Playwright browser, spend limits, skills), import it
from `pydantic_ai_harness` (or `pydantic_ai.capabilities`, like the built-in
web search) in `agent.py` and add it to the `capabilities=[...]` list.
