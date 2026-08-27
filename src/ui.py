"""Terminal rendering: banner, help, tool-call/result lines, memory view."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .config import Config

# The harness's Planning capability's tool result text always ends in a
# `Summary: N completed, M in progress, K pending[, ...]` line and, for the
# in-progress item, a `[~] [id] <content>` line — see the installed
# pydantic_ai_harness's planning/_toolset.py render_summary/status_icon.
# Parsed here instead of reconstructing plan state independently, since the
# harness already renders it once per call.
PLAN_TOOL_NAMES = frozenset(
    {
        "write_plan",
        "read_plan",
        "add_task",
        "update_task_status",
        "update_task_statuses",
        "remove_task",
    }
)
_PLAN_SUMMARY = re.compile(r"Summary:\s*(.+)")
_PLAN_COUNT = re.compile(r"(\d+)\s+(\w+)")
_PLAN_CURRENT = re.compile(r"\[~\]\s*\[[^\]]+\]\s*(.+)")

console = Console(highlight=False)
# Tool-call activity, the usage footer, and notices are diagnostics, not the
# answer — a scripted caller (parsing stdout for a clean model response, the
# way brain_call.call() parses a JSON decision out of headless claude's
# stdout) needs a way to keep them off stdout without losing them entirely.
# --quiet routes them here instead of dropping them; the model's actual text
# output is untouched either way. Rich auto-detects a non-tty file() and
# skips ANSI codes on its own, so no extra config needed for piped output.
console_err = Console(highlight=False, file=sys.stderr)


def _out(quiet: bool) -> Console:
    return console_err if quiet else console

BANNER = """\
[bold cyan]tcode[/bold cyan] [dim]— an alternative to Claude Code, running on Groq[/dim]
  [dim]dir[/dim]     {cwd}
  [dim]model[/dim]   {model}
  [dim]memory[/dim]  {memory_dir}
  [dim]scratch[/dim] {scratch_dir}
[dim]/help for commands, Ctrl-D or /exit to leave[/dim]
"""

HELP = """\
[bold]Commands[/bold]
  /help          show this help
  /clear         clear the conversation in this session
  /memory        show the global memory notebook
  /sessions      list saved sessions for this project
  /skills        list skills in ~/.tcode/skills
  /skill <name>  load a skill, added to your next message
  /exit          leave (also: /quit, Ctrl-D)
"""


def print_banner(cfg: Config) -> None:
    console.print(
        BANNER.format(
            cwd=cfg.cwd,
            model=cfg.model,
            memory_dir=cfg.memory_dir,
            scratch_dir=cfg.scratch_dir,
        )
    )


def print_help() -> None:
    console.print(HELP)


def print_error(message: str) -> None:
    # Always stderr, quiet or not: an error is never the answer a scripted
    # caller is parsing for, and an interactive user still sees stderr by
    # default without redirection.
    console_err.print(f"[bold red]error:[/bold red] {message}")


def print_notice(message: str, *, quiet: bool = False) -> None:
    _out(quiet).print(f"[dim]{message}[/dim]")


def render_tool_call(tool_name: str, args: dict, *, quiet: bool = False) -> None:
    arg_str = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
    _out(quiet).print(f"[cyan]›[/cyan] [bold]{tool_name}[/bold]({arg_str})")


def render_tool_result(tool_name: str, content: object, is_error: bool, *, quiet: bool = False) -> None:
    text = _short(content, limit=240)
    out = _out(quiet)
    if is_error:
        out.print(f"  [red]✗ {tool_name}: {text}[/red]")
    else:
        out.print(f"  [dim green]✓ {text}[/dim green]")


def render_plan_update(content: object, *, quiet: bool = False) -> None:
    """Compact "plan: N/M done · current: <task>" line for a Planning tool result.

    A weak model's "no tool calls left" can't be fully trusted to mean
    "actually done" — this is the cheapest available signal for a human
    watching a long unattended run to catch it wandering off task early,
    rather than finding out at the end.
    """
    text = str(content)
    summary_match = _PLAN_SUMMARY.search(text)
    if not summary_match:
        _out(quiet).print(f"[bold cyan]plan:[/bold cyan] [dim]{_short(text, limit=100)}[/dim]")
        return
    counts = {label: int(n) for n, label in _PLAN_COUNT.findall(summary_match.group(1))}
    total = sum(counts.values())
    done = counts.get("completed", 0)
    current_match = _PLAN_CURRENT.search(text)
    current = f" · current: {_short(current_match.group(1), limit=60)}" if current_match else ""
    _out(quiet).print(f"[bold cyan]plan:[/bold cyan] [dim]{done}/{total} done{current}[/dim]")


def render_usage(cost: object, total_tokens: int, elapsed: float, *, quiet: bool = False) -> None:
    cost_str = f"${cost:.4f}" if cost is not None else "n/a"
    _out(quiet).print(
        f"[dim]{total_tokens} tokens · {cost_str} · {elapsed:.1f}s[/dim]"
    )


def show_memory(cfg: Config) -> None:
    files = sorted(cfg.memory_dir.glob("**/*.md"))
    if not files:
        console.print("[dim]memory notebook is empty[/dim]")
        return
    for f in files:
        rel = f.relative_to(cfg.memory_dir)
        console.rule(str(rel))
        console.print(Markdown(f.read_text()))


def _fmt_smell(record: dict | None) -> str:
    if record is None:
        return "n/a"
    tools = sum(record.get("tool_counts", {}).values())
    retries = record.get("retry_count", 0)
    elapsed = record.get("elapsed_s")
    tokens = record.get("total_tokens")
    elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "?s"
    tokens_str = f"{tokens}tok" if tokens is not None else "?tok"
    return f"{tools} tools, {retries} retries, {elapsed_str}, {tokens_str}"


def render_backtest_table(rows: list[tuple[str, dict | None, dict | None]]) -> None:
    """One row per replayed prompt: before (recorded at the time) vs after
    (this run, current model) — flagged when tool-call count or retries
    jumped materially. See cli.py's backtest_mode."""
    from rich.table import Table

    table = Table(title="backtest")
    table.add_column("prompt")
    table.add_column("before")
    table.add_column("after")
    table.add_column("")
    for prompt, old, new in rows:
        flagged = False
        if old is not None and new is not None:
            old_tools = sum(old.get("tool_counts", {}).values())
            new_tools = sum(new.get("tool_counts", {}).values())
            if new_tools > old_tools + 2 or new.get("retry_count", 0) > old.get("retry_count", 0):
                flagged = True
        table.add_row(
            _short(prompt, limit=60),
            _fmt_smell(old),
            _fmt_smell(new),
            "[bold red]⚠[/bold red]" if flagged else "",
        )
    console.print(table)


def show_sessions(paths: list[Path]) -> None:
    if not paths:
        console.print("[dim]no saved sessions for this project yet[/dim]")
        return
    for p in paths:
        console.print(f"  {p.stem}")


def show_skills(names: list[str]) -> None:
    if not names:
        console.print("[dim]no skills yet — add a .md file to ~/.tcode/skills[/dim]")
        return
    for name in names:
        console.print(f"  {name}")


def _short(value: object, limit: int = 80) -> str:
    s = str(value)
    s = s.replace("\n", "\\n")
    if len(s) > limit:
        s = s[:limit] + "…"
    return s
