"""The single-turn execution loop, shared by cli.py and reduce.py.

Split out from cli.py specifically so reduce.py can reuse it for its final
reduce stage (the same full-capability, `--quiet`-aware turn a plain
one-shot invocation gets) without cli.py and reduce.py importing each other.
"""

from __future__ import annotations

import re
import sys
import time

from pydantic_ai import Agent, UsageLimits
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
