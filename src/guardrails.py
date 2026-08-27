"""Tool guardrails enforcing prompted behavior, not just asking for it.

agent.py's own instructions already ask for each of these; a smaller model
doesn't reliably follow them unprompted, so each guard gives one of them
technical teeth:

- `scope_shell_exploration` blocks an unscoped recursive `ls` (workspace
  root or home directory) in favor of the scoped FileSystem tools, leaving
  scoped recursion (`ls -R src/`) and other commands (`find`, `grep -r`)
  alone — those are used narrowly often enough that blocking them outright
  would be mostly false positives.
- `prefer_web_fetch_tool` blocks `curl`/`wget` against http(s) URLs, forcing
  a retry through `web_fetch` instead, which returns clean markdown rather
  than the raw HTML/JS a model tends to hallucinate an answer from.
- `scope_writes_to` restricts `write_file`/`edit_file`/`create_directory` to
  one or more path prefixes, for a caller whose model decides WHERE to
  write, not just whether. This has to live at the tool-call layer rather
  than as a caller-side post-hoc `git status` diff: a path under a
  gitignored directory never appears in `git status --porcelain` output at
  all, matched or collapsed — and widening that diff into a plain
  filesystem walk isn't safe either, since a file that's gitignored
  *because* something else keeps rewriting it in place would look
  "changed by this run" and get wrongly reverted. Checking the tool call's
  own path argument up front avoids both: same allow-list-not-convention
  approach `protected_patterns` already uses for `.env`/`*.pem`/secrets.
- `UrlLedger` catches a model writing a URL it mangled in transcription —
  truncated mid-word, trailing an ellipsis — even though a tool result
  handed it the real one intact this same run. Two ToolGuardrails share one
  instance: `record` (a result_guard on every tool) remembers every URL a
  tool result actually contained, `check_write` (an argument guard on the
  write tools) rejects a write whose text contains something that looks
  like a mangled copy of one. Always on — see tcode_improvements.txt's
  Finding 5, the reproducible case this closes.
- `citation_paths_exist` rejects a write that cites (backtick-quoted, e.g.
  `` `profiles/bessent/daily/2026-08-23.md` ``) a source file that doesn't
  exist under the workspace — the single most repeated failure shape in
  tcode_improvements.txt's audit (Finding 1: citing a file that was never
  written). Opt-in (`TCODE_CHECK_CITATIONS`, see config.py), not always-on
  like the others here: a generic coding session legitimately proposes
  paths that don't exist *yet* (a file to create next), which this can't
  tell apart from a fabricated citation.
- `confidence_tags_need_citation` rejects a write where a caller-configured
  confidence tag (e.g. `[CONFIRMED]`) appears on a line with no URL or
  cited file path anywhere on it — tcode_improvements.txt's Finding 10:
  a model stating a fabricated number or an invented source under its own
  highest-confidence label, with no citation at all (not a fake path, not
  a mangled URL — the two guards above catch a *wrong* citation, not a
  *missing* one). Opt-in (`TCODE_REQUIRE_CITATION_FOR`, a caller-supplied
  tag list) rather than hardcoding one caller's own tagging vocabulary —
  most tcode tasks don't tag confidence at all, so this only exists once a
  caller's own prompt asks for and defines that convention.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo, ToolResultInfo

_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|]")
_BROAD_TARGETS = {"", ".", "~", "$HOME"}


def _is_unscoped_recursive_ls(command: str) -> bool:
    for segment in _SEGMENT_SPLIT.split(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        if not tokens or tokens[0] != "ls":
            continue
        flags = [t for t in tokens[1:] if t.startswith("-")]
        targets = [t for t in tokens[1:] if not t.startswith("-")]
        if not any("R" in f for f in flags):
            continue
        target = targets[0] if targets else ""
        if target in _BROAD_TARGETS or target == str(Path.home()):
            return True
    return False


def scope_shell_exploration(call: ToolCallInfo) -> GuardrailResult:
    if call.name == "run_command" and _is_unscoped_recursive_ls(str(call.args.get("command", ""))):
        return GuardrailResult.retry(
            "Recursive listing of the whole workspace/home directory is blocked. "
            "Use list_directory/find_files scoped to the specific subdirectory "
            "you need, or run `ls -R` against a narrower path (e.g. a repo you "
            "just cloned), not the workspace root or the home directory."
        )
    return GuardrailResult.allow()


_URL = re.compile(r"https?://\S+")


def _is_curl_or_wget_fetch(command: str) -> bool:
    for segment in _SEGMENT_SPLIT.split(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        if not tokens or tokens[0] not in ("curl", "wget"):
            continue
        if any(_URL.match(t) for t in tokens[1:]):
            return True
    return False


def prefer_web_fetch_tool(call: ToolCallInfo) -> GuardrailResult:
    if call.name == "run_command" and _is_curl_or_wget_fetch(str(call.args.get("command", ""))):
        return GuardrailResult.retry(
            "Fetching a URL with curl/wget is blocked; use the web_fetch tool "
            "instead — it returns clean text/markdown instead of raw HTML/JS. "
            "Use duckduckgo_search first if you don't have a URL yet. If "
            "web_fetch fails or a result looks stale, try a different search "
            "result rather than falling back to shell."
        )
    return GuardrailResult.allow()


WRITE_TOOLS = ("write_file", "edit_file", "create_directory")
"""The three write-capable FileSystem tools `scope_writes_to` gates.

Exposed so the `ToolGuardrail(..., tools=...)` restriction in agent.py names
the same list this guard is meant for, rather than a second hand-copied
tuple drifting out of sync with it.
"""


def scope_writes_to(workspace: str | Path, scopes: Sequence[str]):
    """Build a guard restricting write_file/edit_file/create_directory to `scopes`.

    `scopes` are paths relative to `workspace` (e.g. "src/research/view/
    profiles"); a write is allowed only if its resolved target is that path
    itself or falls under it. Reads are untouched — this only gates the
    three write-capable FileSystem tools, same restriction-of-writes-not-
    reads shape as `protected_patterns`, just an allow-list instead of a
    deny-list. Resolves before comparing (the docstring's own
    `no_writes_outside_workspace` example does the same) so a prefix string
    match can't be fooled by `../`.
    """
    root = Path(workspace).resolve()
    allowed = [(root / s).resolve() for s in scopes]

    def guard(call: ToolCallInfo) -> GuardrailResult:
        raw_path = str(call.args.get("path", ""))
        target = (root / raw_path).resolve()
        if any(target == a or target.is_relative_to(a) for a in allowed):
            return GuardrailResult.allow()
        return GuardrailResult.retry(
            f"Writing to {raw_path!r} is out of scope this session — only "
            f"{' or '.join(s.rstrip('/') + '/' for s in scopes)} may be "
            "created or modified. If the task doesn't fit there, say so "
            "instead of writing elsewhere."
        )

    return guard


def _write_text(call: ToolCallInfo) -> str:
    """The text a write-capable tool call is about to put on disk.

    `write_file(path, content)` and `edit_file(path, old_text, new_text)`
    name it differently; `create_directory(path)` has neither, so this
    reads as an empty string for it — nothing worth scanning there anyway.
    """
    return str(call.args.get("content", call.args.get("new_text", "")))


_URL_IN_TEXT = re.compile(r"https?://[^\s\"'<>\)\]]+")


def _looks_truncated(url: str, seen: set[str]) -> bool:
    """Whether `url` looks like a mangled copy of something already seen.

    Two independent signals, either enough on its own: it visibly trails
    off (an ellipsis or literal "...", the exact shape tcode_improvements
    .txt's Finding 5 reproduced — Tavily/web_fetch handed the model an
    intact URL and it still wrote a cut-off one to the file), or it's a
    strict prefix of a longer URL this run's own tool results actually
    contained — the mechanical case a post-write check can catch with no
    model judgment involved at all.
    """
    # "." is deliberately excluded from this strip set: it's what "..."
    # itself is made of, so stripping it first would swallow the very
    # ellipsis the endswith check below is looking for.
    if url.rstrip("),;:!?]}").endswith(("…", "...")):
        return True
    return any(seen_url != url and seen_url.startswith(url) for seen_url in seen)


class UrlLedger:
    """Remembers every URL a tool result actually returned this run, so a
    later write can be checked against it — see `_looks_truncated` above.

    One instance per agent (built once in agent.py's build_agent), shared
    between two `ToolGuardrail`s: `record` (a `result_guard`, every tool)
    fills the ledger in as results come back; `check_write` (an argument
    guard, write tools only) reads it back before a write lands. Sharing
    state this way — rather than each guard reaching into run history
    itself — is what lets `check_write` compare against everything seen
    so far without needing its own copy of the conversation.
    """

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def record(self, result: ToolResultInfo) -> GuardrailResult:
        self.seen.update(_URL_IN_TEXT.findall(str(result.result)))
        return GuardrailResult.allow()

    def check_write(self, call: ToolCallInfo) -> GuardrailResult:
        bad = sorted({u for u in _URL_IN_TEXT.findall(_write_text(call)) if _looks_truncated(u, self.seen)})
        if not bad:
            return GuardrailResult.allow()
        return GuardrailResult.retry(
            "This content includes a URL that looks truncated or garbled "
            f"partway through ({'; '.join(bad)}) rather than copied in full "
            "from the search/fetch result it came from. Copy it exactly as "
            "given, or leave it out rather than writing a partial one."
        )


_CITED_PATH = re.compile(r"`([\w][\w./-]*(?:/[\w][\w./-]*)+\.[A-Za-z0-9]{1,5})`")


def citation_paths_exist(workspace: str | Path):
    """Build a guard rejecting a write that cites a source file which
    doesn't actually exist under `workspace`.

    Scoped deliberately narrow: only a backtick-quoted, slash-containing,
    extensioned path — the shape a model uses when citing a file it read
    (`` `profiles/bessent/daily/2026-08-23.md` ``) — not any bare filename
    mentioned in prose. `TCODE_CHECK_CITATIONS` opts in (see config.py)
    rather than this running always: a generic coding session legitimately
    proposes paths that don't exist yet (a file to create next), which
    this has no way to tell apart from a fabricated citation — the failure
    mode it exists for (tcode_improvements.txt's Finding 1) is specific to
    a caller whose output is citations of files it was actually handed.
    """
    root = Path(workspace).resolve()

    def _exists(rel_path: str) -> bool:
        resolved = (root / rel_path).resolve()
        return resolved.is_relative_to(root) and resolved.is_file()

    def guard(call: ToolCallInfo) -> GuardrailResult:
        cited = set(_CITED_PATH.findall(_write_text(call)))
        missing = sorted(p for p in cited if not _exists(p))
        if not missing:
            return GuardrailResult.allow()
        return GuardrailResult.retry(
            "This content cites a source file that doesn't exist under "
            f"this workspace: {', '.join(missing)}. Only cite a file you "
            "actually read this run — remove the citation or fix the path "
            "instead of leaving a reference to something that isn't there."
        )

    return guard


def confidence_tags_need_citation(tags: Sequence[str]):
    """Build a guard rejecting a write where any of `tags` appears on a
    line with no URL or cited file path anywhere on that same line.

    `tags` are literal strings a caller's own prompt defines and asks the
    model to use (e.g. `["[CONFIRMED]"]`) — this has no built-in vocabulary
    of its own, so an empty/unset `tags` (the default, via
    TCODE_REQUIRE_CITATION_FOR) makes the returned guard a no-op rather
    than matching every write. Proximity is deliberately just "the same
    line": every real example of this failure (tcode_improvements.txt's
    Finding 10) was a whole bullet or sentence carrying the tag with no
    source anywhere in it, not a citation placed elsewhere in the
    paragraph — a wider search window could credit a tag with a citation
    that's actually backing up a different claim nearby.
    """
    escaped = [re.escape(t) for t in tags if t.strip()]
    if not escaped:
        return lambda call: GuardrailResult.allow()
    tag_pattern = re.compile("|".join(escaped))

    def guard(call: ToolCallInfo) -> GuardrailResult:
        bad_lines = [
            line.strip()[:120]
            for line in _write_text(call).splitlines()
            if tag_pattern.search(line) and not _URL_IN_TEXT.search(line) and not _CITED_PATH.search(line)
        ]
        if not bad_lines:
            return GuardrailResult.allow()
        return GuardrailResult.retry(
            f"This content uses a confidence tag ({', '.join(tags)}) on a "
            f"line with no URL or cited file path backing it up: "
            f"{'; '.join(bad_lines)}. Add a real source on that line, or "
            "use a lower-confidence tag if you don't actually have one."
        )

    return guard
