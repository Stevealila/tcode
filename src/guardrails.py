"""Tool guardrails enforcing prompted behavior, not just asking for it.

CODER_INSTRUCTIONS (agent.py) already asks the model to prefer the scoped
FileSystem tools over raw shell recon and to target the specific thing the
user asked about rather than dumping the whole workspace. Observed behavior
against Groq's smaller models shows that instruction alone doesn't always
hold: a real session ran `ls -R .` against the user's home directory to
answer a question about one specific repo, walking past unrelated personal
files it never needed to see. `scope_shell_exploration` gives that
instruction technical teeth for the one shape of command that caused it: an
unscoped recursive `ls` against the workspace root or the user's home
directory. It intentionally does not touch scoped recursion (`ls -R src/`,
`ls -R ./some_repo`) or other commands like `find`/`grep -r`, which are far
more often used narrowly and would produce too many false positives to
block outright.

`prefer_web_fetch_tool` is the same shape of backstop for the web tools
added alongside WebSearch/WebFetch in agent.py. A real session asked for the
current gold price: `duckduckgo_search` returned a snippet with a stale
dated figure, the model reached for `curl`/raw shell to get a live number
instead of trying another search result or `web_fetch`, that curl call hit
JS-rendered markup it couldn't parse, and the run ended in a hallucinated
price range presented as current fact. Blocking `curl`/`wget` against
http(s) URLs forces the retry through `web_fetch` instead, which returns
clean markdown rather than a page's raw HTML/JS.
"""

from __future__ import annotations

import re
import shlex
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
