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
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo

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
