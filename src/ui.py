"""Terminal rendering: banner, help, tool-call/result lines, memory view."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .config import Config

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
  /help       show this help
  /clear      clear the conversation in this session
  /memory     show the global memory notebook
  /sessions   list saved sessions for this project
  /exit       leave (also: /quit, Ctrl-D)
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


def show_sessions(paths: list[Path]) -> None:
    if not paths:
        console.print("[dim]no saved sessions for this project yet[/dim]")
        return
    for p in paths:
        console.print(f"  {p.stem}")


def _short(value: object, limit: int = 80) -> str:
    s = str(value)
    s = s.replace("\n", "\\n")
    if len(s) > limit:
        s = s[:limit] + "…"
    return s
