"""Argument parsing and the interactive REPL loop."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from pydantic_ai import ModelHTTPError, UsageLimits
from pydantic_ai.messages import ModelMessage

from . import ui
from .agent import build_agent
from .config import Config, ConfigError, load_config
from .runner import run_turn
from .sessions import list_sessions, load_latest_session, save_session


def _resolve_write_path(cfg: Config, raw: str) -> Path:
    """Resolve --write's target relative to cfg.cwd, refusing to escape it.

    Same sandboxing intent as FileSystem's own root_dir, applied to a path
    this process writes directly rather than one a tool call resolves.
    """
    p = (cfg.cwd / raw).resolve()
    try:
        p.relative_to(cfg.cwd)
    except ValueError:
        raise ConfigError(
            f"--write path must stay inside the workspace ({cfg.cwd}), got {raw!r}"
        ) from None
    return p


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _apply_write_fallback(
    write_path: Path | None, before_mtime: float | None, final_text: str, *, quiet: bool
) -> None:
    """A Python-level safety net for --write, see build_parser()'s help text.

    Mtime-based rather than "did a write_file call to this exact path
    happen": that's what the caller actually cares about (is the file on
    disk different now?), and it's the same check profile_scout.sh/
    story_intake.sh/state_of_market.sh already had to write themselves in
    bash before this existed — moved here once so no caller has to
    reimplement it.
    """
    if write_path is None:
        return
    if _mtime(write_path) != before_mtime:
        return
    if not final_text.strip():
        return
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(final_text)
    ui.print_notice(
        f"--write: the model's own write didn't land this turn — wrote its "
        f"answer to {write_path} directly",
        quiet=quiet,
    )


def _friendly_error(e: Exception) -> str:
    """Turn a raw exception into something a terminal user can act on.

    Groq's rate-limit responses (HTTP 413/429) already carry a precise,
    well-formed message naming exactly which budget was hit — requests or
    tokens, per minute or per day — and how long to wait. Surface that
    directly rather than guessing from the status code alone: a 429 can
    mean requests-per-minute (our own throttle() should prevent that one),
    but it can just as easily mean the *daily* token quota is exhausted,
    which is a completely different situation with a completely different
    fix (wait, it's unrelated to conversation size, /clear won't help).
    """
    if isinstance(e, ModelHTTPError) and e.status_code in (413, 429):
        detail = e.body.get("error", {}).get("message") if isinstance(e.body, dict) else None
        if detail:
            return f"{e.model_name} rejected the request (HTTP {e.status_code}): {detail}"
        return (
            f"{e.model_name} rejected the request (HTTP {e.status_code}) — "
            "likely a Groq rate limit. Wait and try again, or switch "
            "models with --model."
        )
    return str(e)


async def interactive(cfg: Config, message_history: list[ModelMessage]) -> None:
    agent = build_agent(cfg)
    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    ui.print_banner(cfg)

    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(cfg.history_file))
    )

    while True:
        try:
            user_input = await prompt_session.prompt_async("you › ")
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/help":
            ui.print_help()
            continue
        if user_input == "/clear":
            message_history = []
            ui.print_notice("conversation cleared")
            continue
        if user_input == "/memory":
            ui.show_memory(cfg)
            continue
        if user_input == "/sessions":
            ui.show_sessions(list_sessions(cfg))
            continue

        try:
            message_history, _ = await run_turn(agent, user_input, message_history, usage_limits, cfg)
        except KeyboardInterrupt:
            ui.print_notice("interrupted")
            continue
        except Exception as e:  # noqa: BLE001 - keep the REPL alive on turn failures
            ui.print_error(_friendly_error(e))
            continue

        save_session(cfg, message_history)

    ui.print_notice("bye")


async def one_shot(
    cfg: Config,
    prompt: str,
    message_history: list[ModelMessage],
    *,
    quiet: bool = False,
    write_path: Path | None = None,
) -> None:
    agent = build_agent(cfg)
    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    before_mtime = _mtime(write_path) if write_path else None
    try:
        message_history, final_text = await run_turn(
            agent, prompt, message_history, usage_limits, cfg, quiet=quiet
        )
    except Exception as e:  # noqa: BLE001
        ui.print_error(_friendly_error(e))
        raise SystemExit(1) from e
    _apply_write_fallback(write_path, before_mtime, final_text, quiet=quiet)
    save_session(cfg, message_history)


async def verify_mode(cfg: Config, prompt: str) -> None:
    """Independent-verifier decision mode — see verify.py's module docstring.

    On agreement, prints the primary's answer to stdout, same contract as
    `--quiet`. On disagreement, prints *nothing* to stdout and exits 2 —
    deliberately not a hedge, a pick-one, or an error message on stdout: a
    caller built around "stdout has a clean answer or it doesn't" (e.g.
    scanning for the first parseable JSON object) already treats empty/
    unparseable stdout as "this attempt produced nothing usable" and moves
    to its own fallback — which is exactly the right response to a verifier
    disagreement too. Inventing a different signal would need every such
    caller to learn a second failure shape for what is, to them, the same
    situation: no trustworthy answer this attempt.
    """
    from . import verify as verify_mod

    agent = build_agent(cfg)
    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    timeout = int(os.environ.get("TCODE_VERIFY_TIMEOUT", "150"))

    try:
        _, primary_text = await run_turn(agent, prompt, [], usage_limits, cfg, capture=True)
        verifier_text = await verify_mod.get_verifier_answer(cfg, prompt, usage_limits, timeout)
        agreed, verdict = await verify_mod.compare(cfg, primary_text, verifier_text)
    except Exception as e:  # noqa: BLE001
        ui.print_error(_friendly_error(e))
        raise SystemExit(1) from e

    ui.print_notice(f"verify: primary={primary_text!r}", quiet=True)
    ui.print_notice(f"verify: verifier={verifier_text!r}", quiet=True)
    ui.print_notice(f"verify: verdict={verdict!r}", quiet=True)

    if agreed:
        sys.stdout.write(primary_text)
        if not primary_text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        return

    ui.print_notice("verify: DISAGREEMENT — withholding the answer (stdout empty, exit 2)", quiet=True)
    raise SystemExit(2)


async def reduce_mode(
    cfg: Config,
    prompt: str,
    pattern: str,
    message_history: list[ModelMessage],
    *,
    quiet: bool,
    write_path: Path | None = None,
) -> None:
    """Map-reduce over many files in one call — see reduce.py's module docstring."""
    from . import reduce as reduce_mod

    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    before_mtime = _mtime(write_path) if write_path else None
    try:
        message_history, final_text = await reduce_mod.run_reduce(
            cfg, prompt, pattern, message_history, usage_limits, quiet=quiet
        )
    except Exception as e:  # noqa: BLE001
        ui.print_error(_friendly_error(e))
        raise SystemExit(1) from e
    _apply_write_fallback(write_path, before_mtime, final_text, quiet=quiet)
    save_session(cfg, message_history)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcode",
        description="A system-wide coding-agent harness powered by Groq models.",
    )
    parser.add_argument(
        "prompt", nargs="*", help="run a single prompt non-interactively and exit"
    )
    parser.add_argument(
        "-c", "--continue", dest="cont", action="store_true",
        help="continue the last session in this directory",
    )
    parser.add_argument("--model", default=None, help="override the Groq model for this run")
    parser.add_argument(
        "--sessions", action="store_true", help="list saved sessions for this directory and exit"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="one-shot mode only: send tool-call activity, the usage footer, and "
        "notices to stderr instead of stdout, so stdout is just the model's answer "
        "— for a script parsing the output (a JSON decision, say), not a human",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="one-shot mode only: get the answer, independently re-derive it with a "
        "different model (or TCODE_VERIFY_CMD, an external command) and only print "
        "it if they agree — stdout is empty and the exit code is 2 on disagreement. "
        "For a caller that shouldn't act on a confidently-wrong-but-clean answer. "
        "Implies --quiet's stdout contract; ignored with -c/--continue (verification "
        "needs an independent re-derivation, not a continued conversation).",
    )
    parser.add_argument(
        "--write", metavar="PATH", default=None,
        help="one-shot/--reduce only: PATH the run is expected to write via its own "
        "write_file call. If PATH's mtime is unchanged when the turn ends — the "
        "model answered in text instead of actually calling the tool, or the "
        "tool call itself failed — tcode writes the model's final answer there "
        "directly, so a caller with one known output path doesn't depend on the "
        "model's own write succeeding. Relative to this directory.",
    )
    parser.add_argument(
        "--reduce", metavar="PATTERN", default=None,
        help="one-shot mode only: PATTERN is a glob (relative to this directory, ** "
        "allowed) matching many files to read and reduce to one answer — the prompt "
        "describes what to extract per file and how to synthesize the result. "
        "@listfile reads an explicit newline-separated file/glob list instead, for "
        "selection logic a bare glob can't express (e.g. a caller's own date-range "
        "filter). Chunks internally (map each file, group/digest if there are more "
        "than a handful, then one final turn) rather than asking a single turn to "
        "read a long file list, which is unreliable regardless of file size — see "
        "reduce.py.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cfg = load_config(Path.cwd(), model_override=args.model)
    except ConfigError as e:
        ui.print_error(str(e))
        raise SystemExit(1) from e

    if args.sessions:
        ui.show_sessions(list_sessions(cfg))
        return

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        # No positional prompt, and stdin isn't a terminal: something's
        # piped in (`echo "..." | tcode`, or a caller passing the prompt as
        # subprocess `input=` — Claude Code's own `-p` accepts a prompt
        # either way, and brain_call.py-shaped Python callers already use
        # `input=` uniformly for every provider they invoke). Falling
        # through to the interactive REPL here would hang forever reading
        # from a pipe that's never going to send REPL commands.
        prompt = sys.stdin.read().strip()

    if prompt and args.verify:
        asyncio.run(verify_mode(cfg, prompt))
        return

    try:
        write_path = _resolve_write_path(cfg, args.write) if args.write else None
    except ConfigError as e:
        ui.print_error(str(e))
        raise SystemExit(1) from e

    message_history = load_latest_session(cfg) if args.cont else []

    if prompt and args.reduce:
        asyncio.run(
            reduce_mode(cfg, prompt, args.reduce, message_history, quiet=args.quiet, write_path=write_path)
        )
        return

    if prompt:
        asyncio.run(one_shot(cfg, prompt, message_history, quiet=args.quiet, write_path=write_path))
    else:
        asyncio.run(interactive(cfg, message_history))


if __name__ == "__main__":
    main()
