"""Tool guardrail enforcing scoped exploration, not just prompted for it.

CODER_INSTRUCTIONS (agent.py) already asks the model to prefer the scoped
FileSystem tools over raw shell recon and to target the specific thing the
user asked about rather than dumping the whole workspace. Observed behavior
against Groq's smaller models shows that instruction alone doesn't always
hold: a real session ran `ls -R .` against the user's home directory to
answer a question about one specific repo, walking past unrelated personal
files it never needed to see. This gives that instruction technical teeth
for the one shape of command that caused it: an unscoped recursive `ls`
against the workspace root or the user's home directory. It intentionally
does not touch scoped recursion (`ls -R src/`, `ls -R ./some_repo`) or other
commands like `find`/`grep -r`, which are far more often used narrowly and
would produce too many false positives to block outright.
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
