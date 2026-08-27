"""The single-turn execution loop, shared by cli.py and reduce.py.

Split out from cli.py specifically so reduce.py can reuse it for its final
reduce stage (the same full-capability, `--quiet`-aware turn a plain
one-shot invocation gets) without cli.py and reduce.py importing each other.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter

from pydantic_ai import Agent, UnexpectedModelBehavior, UsageLimitExceeded, UsageLimits
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
from .config import Config
from .ratelimit import throttle
from .telemetry import record_smell

_MARKDOWN_TABLE_ROW = re.compile(r"\|\s*-{2,}\s*\|")
_SUBSTANTIVE_ANSWER_CHARS = 800

# Extra whole-turn attempts on top of pydantic_ai's own per-tool retry
# budget (which only covers one call, not the turn itself) — for a turn
# whose tool-calling breaks with no partial answer to salvage. See the
# UnexpectedModelBehavior handler below.
_MAX_EMPTY_TURN_RETRIES = 2

# A model can print a tool call as JSON prose instead of actually making it
# (`{"name": "write_file", "arguments": {...`) — never a valid answer on its
# own, so it's treated as a retry case. See _looks_like_faked_tool_call's
# call site.
_FAKE_TOOL_CALL_PREFIX = re.compile(r'^\{\s*"name"\s*:\s*"[a-zA-Z_][a-zA-Z0-9_]*"\s*,\s*"arguments"\s*:')


def _looks_like_faked_tool_call(text: str) -> bool:
    return bool(_FAKE_TOOL_CALL_PREFIX.match(text.strip()))


def _classify_exception(e: BaseException) -> str:
    """A short slug for a turn-ending exception, stored on the smell
    record's `outcome` field so --backtest can tell a run that looped into
    the request limit apart from one that crashed on a garbled tool call."""
    if isinstance(e, UsageLimitExceeded):
        return "usage_limit"
    if isinstance(e, UnexpectedModelBehavior):
        return "model_behavior_error"
    return "error"


def _skipped_reading_files(tool_names: Counter[str], final_text: str) -> bool:
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
    # Render the answer as live Markdown only for an interactive terminal
    # user. Piped/redirected output (not a tty) and every scripted path
    # (--quiet, capture) keep getting raw text, or nothing, as before.
    render_md = not quiet and ui.console.is_terminal
    start = time.monotonic()

    attempt = 0
    result = None
    final_text_parts: list[str] = []
    tool_names: Counter[str] = Counter()
    outcome = "ok"
    error: str | None = None
    # Held across the retry loop so a `finally` can always stop a live
    # Markdown render — most importantly on a KeyboardInterrupt, which is a
    # BaseException the `except` handlers below never see, and which would
    # otherwise leave Rich's Live thread running and the cursor hidden.
    md_stream: ui.MarkdownStream | None = None

    def _finalize() -> None:
        """Render the usage footer and write the smell record — exactly
        once, on whichever path the turn exits by (success, salvage,
        raise). record_smell must see the failing runs too; they're the
        whole point of the telemetry."""
        elapsed = time.monotonic() - start
        usage = getattr(result, "usage", None)
        if usage is not None:
            ui.render_usage(usage.cost, usage.total_tokens, elapsed, quiet=quiet)
        record_smell(
            cfg, prompt, dict(tool_names), attempt, elapsed, usage,
            outcome=outcome, error=error,
        )

    for attempt in range(_MAX_EMPTY_TURN_RETRIES + 1):
        streaming_text = False
        final_text_parts = []
        tool_names = Counter()
        md_stream = ui.MarkdownStream() if render_md else None

        try:
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
                                    first_chunk = not streaming_text
                                    streaming_text = True
                                    final_text_parts.append(text)
                                    if md_stream is not None:
                                        md_stream.feed(text)
                                    elif not capture:
                                        if first_chunk and not quiet:
                                            ui.begin_assistant_message()
                                        sys.stdout.write(text)
                                        sys.stdout.flush()
                    elif Agent.is_call_tools_node(node):
                        if streaming_text:
                            if md_stream is not None:
                                md_stream.close()
                            elif not capture:
                                sys.stdout.write("\n")
                                sys.stdout.flush()
                            streaming_text = False
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    part = event.part
                                    tool_names[part.tool_name] += 1
                                    if part.tool_name not in ui.PLAN_TOOL_NAMES:
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
                                    if event.part.tool_name in ui.PLAN_TOOL_NAMES and not is_error:
                                        ui.render_plan_update(content, quiet=quiet)
                                    else:
                                        ui.render_tool_result(event.part.tool_name, content, is_error, quiet=quiet)

                if streaming_text:
                    if md_stream is not None:
                        md_stream.close()
                    elif not capture:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    streaming_text = False

                result = run.result

            if _looks_like_faked_tool_call("".join(final_text_parts)):
                if attempt < _MAX_EMPTY_TURN_RETRIES:
                    ui.print_notice(
                        "note: the model printed a tool call as text instead of "
                        f"actually making it — retrying the whole turn "
                        f"({attempt + 1}/{_MAX_EMPTY_TURN_RETRIES})",
                        quiet=quiet,
                    )
                    continue
                # Retry budget exhausted and it's still doing this. Falling
                # through silently here was the bug: the loop below would
                # treat this known-bad text exactly like a normal answer —
                # printing it to stdout, letting --write persist it as if it
                # were the model's real output — with nothing in the logs to
                # tell a human (or a caller like profile_scout.sh, watching
                # only the process exit code) that it's a faked tool call,
                # not prose. Still returned rather than raised: it's
                # sometimes a truncated-but-genuine write_file payload (see
                # --write's recovery of this exact shape), so discarding it
                # outright would lose more than flagging it loudly does.
                outcome = "faked_tool_call"
                ui.print_notice(
                    "note: the model printed a tool call as text instead of "
                    "actually making it, and kept doing so through "
                    f"{_MAX_EMPTY_TURN_RETRIES} retries — giving up and "
                    "returning it as-is, but treat this as an unconfirmed, "
                    "likely-malformed answer, not a normal one.",
                    quiet=quiet,
                )
            break
        except UnexpectedModelBehavior as e:
            # Seen in practice on Groq's gpt-oss models at large output
            # sizes: a garbled tool name exhausts the retry budget before
            # the model gets a real chance at the call. Salvage whatever
            # text it already produced rather than losing a complete answer
            # to a last-step formatting failure.
            if md_stream is not None:
                md_stream.close()
            final_text = "".join(final_text_parts)
            if final_text.strip():
                outcome = "salvaged_after_tool_failure"
                error = str(e)
                ui.print_notice(
                    f"note: the model's own tool call failed after this "
                    f"answer was already produced ({e}) — returning the "
                    "text, but nothing this turn was confirmed written via "
                    "a tool.",
                    quiet=quiet,
                )
                _finalize()
                return message_history, final_text
            # Nothing to salvage — retry the whole turn.
            if attempt >= _MAX_EMPTY_TURN_RETRIES:
                outcome = "model_behavior_error"
                error = str(e)
                _finalize()
                raise
            ui.print_notice(
                f"note: the model's tool call failed before producing any "
                f"answer ({e}) — retrying the whole turn "
                f"({attempt + 1}/{_MAX_EMPTY_TURN_RETRIES})",
                quiet=quiet,
            )
        except Exception as e:  # noqa: BLE001
            if md_stream is not None:
                md_stream.close()
            # Any other failure exiting the turn — most importantly
            # UsageLimitExceeded (the model looped until it hit
            # request_limit), but also a raw provider error. Previously
            # these propagated straight out with no smell line, so
            # --backtest was blind to exactly the runs that regressed
            # worst. Record, then re-raise unchanged so cli.py's handler
            # still does its job.
            outcome = _classify_exception(e)
            error = f"{type(e).__name__}: {e}"
            _finalize()
            raise
        finally:
            # Safety net for KeyboardInterrupt (a BaseException the handlers
            # above never catch): stop the Live render so the cursor comes
            # back and the thread ends. Idempotent — a no-op on every path
            # that already closed it.
            if md_stream is not None:
                md_stream.close()

    _finalize()

    final_text = "".join(final_text_parts)
    if _skipped_reading_files(tool_names, final_text):
        ui.print_notice(
            "note: that answer didn't read any file contents "
            f"(only used: {', '.join(sorted(tool_names))}) — treat it as a "
            "directory-level impression, not a code review.",
            quiet=quiet,
        )

    return result.all_messages(), final_text
