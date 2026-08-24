"""Argument parsing and the interactive REPL loop."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from pydantic_ai import Agent, ModelHTTPError, UsageLimits
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
)

from . import ui
from .agent import build_agent
from .config import Config, ConfigError, load_config
from .ratelimit import throttle
from .sessions import list_sessions, load_latest_session, save_session


_MARKDOWN_TABLE_ROW = re.compile(r"\|\s*-{2,}\s*\|")
_SUBSTANTIVE_ANSWER_CHARS = 800


def _skipped_reading_files(tool_names: set[str], final_text: str) -> bool:
    """Whether this turn produced a substantial answer without reading any file.

    A `ToolGuardrail` can force the *scoping* of exploration (see
    guardrails.py) because that fires per tool call, before the call runs.
    There's no equivalent lever for "the model wrote a long analysis without
    reading any source" — that's a property of the whole turn, only visible
    after it's over, and `OutputGuardrail`'s retry/block verdicts aren't
    supported on the streaming run this CLI uses (they'd surface as
    UnexpectedModelBehavior instead of a graceful re-prompt). So instead of
    trying to force a retry, this only tells the user the truth: a
    review-shaped answer built entirely from directory listings is a
    directory-level impression, not a code review, however confident it
    reads.
    """
    if not tool_names or "read_file" in tool_names:
        return False
    return len(final_text) > _SUBSTANTIVE_ANSWER_CHARS or bool(_MARKDOWN_TABLE_ROW.search(final_text))


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


async def run_turn(
    agent: Agent,
    prompt: str,
    message_history: list[ModelMessage],
    usage_limits: UsageLimits,
    cfg: Config,
    *,
    quiet: bool = False,
    capture: bool = False,
) -> tuple[list[ModelMessage], str]:
    """Run one turn, streaming assistant text and tool activity live.

    Returns `(message_history, final_text)`.

    `quiet` routes tool-call/result lines, the usage footer, and notices to
    stderr instead of stdout — the model's actual text output is unaffected
    either way. For a scripted, non-interactive caller parsing stdout for a
    clean answer (a JSON decision, say), this is the difference between a
    parseable result and one interleaved with rendering it never asked for.

    `capture` additionally holds the answer text back from stdout entirely
    (still returned in `final_text`) — implies `quiet`-style routing for
    diagnostics too. For a caller (verify.py) that needs to see the answer
    *before* deciding whether it should reach stdout at all — printing it
    live and then "unprinting" it isn't possible.
    """
    quiet = quiet or capture
    start = time.monotonic()
    streaming_text = False
    final_text_parts: list[str] = []
    tool_names: set[str] = set()

    async with agent.iter(
        prompt, message_history=message_history, usage_limits=usage_limits
    ) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                await throttle(
                    cfg.rpm_state_file,
                    cfg.max_rpm,
                    on_wait=lambda w: ui.print_notice(
                        f"rate limit: waiting {w:.0f}s ({cfg.max_rpm} requests/min budget)",
                        quiet=quiet,
                    ),
                )
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        text = None
                        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                            text = event.part.content
                        elif isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            text = event.delta.content_delta
                        if text:
                            streaming_text = True
                            final_text_parts.append(text)
                            if not capture:
                                sys.stdout.write(text)
                                sys.stdout.flush()
            elif Agent.is_call_tools_node(node):
                if streaming_text:
                    if not capture:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    streaming_text = False
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, FunctionToolCallEvent):
                            part = event.part
                            tool_names.add(part.tool_name)
                            try:
                                args = part.args_as_dict()
                            except Exception:
                                args = {}
                            ui.render_tool_call(part.tool_name, args, quiet=quiet)
                        elif isinstance(event, FunctionToolResultEvent):
                            is_error = isinstance(event.part, RetryPromptPart)
                            content = (
                                event.content
                                if event.content is not None
                                else event.part.content
                            )
                            ui.render_tool_result(event.part.tool_name, content, is_error, quiet=quiet)

        if streaming_text and not capture:
            sys.stdout.write("\n")
            sys.stdout.flush()

        result = run.result

    elapsed = time.monotonic() - start
    usage = result.usage
    ui.render_usage(usage.cost, usage.total_tokens, elapsed, quiet=quiet)

    final_text = "".join(final_text_parts)
    if _skipped_reading_files(tool_names, final_text):
        ui.print_notice(
            "note: that answer didn't read any file contents "
            f"(only used: {', '.join(sorted(tool_names))}) — treat it as a "
            "directory-level impression, not a code review.",
            quiet=quiet,
        )

    return result.all_messages(), final_text


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
    cfg: Config, prompt: str, message_history: list[ModelMessage], *, quiet: bool = False
) -> None:
    agent = build_agent(cfg)
    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    try:
        message_history, _ = await run_turn(agent, prompt, message_history, usage_limits, cfg, quiet=quiet)
    except Exception as e:  # noqa: BLE001
        ui.print_error(_friendly_error(e))
        raise SystemExit(1) from e
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

    message_history = load_latest_session(cfg) if args.cont else []
    if prompt:
        asyncio.run(one_shot(cfg, prompt, message_history, quiet=args.quiet))
    else:
        asyncio.run(interactive(cfg, message_history))


if __name__ == "__main__":
    main()
