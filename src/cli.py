"""Argument parsing and the interactive REPL loop."""

from __future__ import annotations

import argparse
import asyncio
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
) -> list[ModelMessage]:
    """Run one turn, streaming assistant text and tool activity live."""
    start = time.monotonic()
    streaming_text = False

    async with agent.iter(
        prompt, message_history=message_history, usage_limits=usage_limits
    ) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                await throttle(
                    cfg.rpm_state_file,
                    cfg.max_rpm,
                    on_wait=lambda w: ui.print_notice(
                        f"rate limit: waiting {w:.0f}s ({cfg.max_rpm} requests/min budget)"
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
                            sys.stdout.write(text)
                            sys.stdout.flush()
            elif Agent.is_call_tools_node(node):
                if streaming_text:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    streaming_text = False
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, FunctionToolCallEvent):
                            part = event.part
                            try:
                                args = part.args_as_dict()
                            except Exception:
                                args = {}
                            ui.render_tool_call(part.tool_name, args)
                        elif isinstance(event, FunctionToolResultEvent):
                            is_error = isinstance(event.part, RetryPromptPart)
                            content = (
                                event.content
                                if event.content is not None
                                else event.part.content
                            )
                            ui.render_tool_result(event.part.tool_name, content, is_error)

        if streaming_text:
            sys.stdout.write("\n")
            sys.stdout.flush()

        result = run.result

    elapsed = time.monotonic() - start
    usage = result.usage
    ui.render_usage(usage.cost, usage.total_tokens, elapsed)

    return result.all_messages()


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
            message_history = await run_turn(agent, user_input, message_history, usage_limits, cfg)
        except KeyboardInterrupt:
            ui.print_notice("interrupted")
            continue
        except Exception as e:  # noqa: BLE001 - keep the REPL alive on turn failures
            ui.print_error(_friendly_error(e))
            continue

        save_session(cfg, message_history)

    ui.print_notice("bye")


async def one_shot(cfg: Config, prompt: str, message_history: list[ModelMessage]) -> None:
    agent = build_agent(cfg)
    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    try:
        message_history = await run_turn(agent, prompt, message_history, usage_limits, cfg)
    except Exception as e:  # noqa: BLE001
        ui.print_error(_friendly_error(e))
        raise SystemExit(1) from e
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

    message_history = load_latest_session(cfg) if args.cont else []

    prompt = " ".join(args.prompt).strip()
    if prompt:
        asyncio.run(one_shot(cfg, prompt, message_history))
    else:
        asyncio.run(interactive(cfg, message_history))


if __name__ == "__main__":
    main()
