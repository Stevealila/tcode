"""Context-window management: the compaction pipeline every agent runs, plus
the helpers the `/context` and `/compact` REPL commands use.

Two different limits get called "context" and this module keeps them apart:

  * the model's context **window** — a hard per-request size ceiling.
    `openai/gpt-oss-*` on Groq is 131K; google/zai models vary. Exceeding it
    is a hard 400 from the provider.
  * Groq's tokens-per-**minute** rate limit — a throughput ceiling across
    requests over time, handled entirely by `ratelimit.throttle()`, not
    here. A large history makes each request cost more but does not by
    itself trip the RPM/TPM limiter (Groq also caches the system prompt and
    tool schemas server-side, so repeat turns re-pay only for new content).

The previous pipeline clamped history to a flat ~3K tokens to stay clear of
the minute limit, which also threw away almost all of the model's working
memory after two tool calls. This one triggers on a **fraction of the real
window**, resolved per request — so it is correct after a `/model` switch to
any size of model, and only compacts when the window is genuinely filling.

The ladder, cheapest first (`TieredCompaction` escalates only as far as it
must to get back under `context_compact_fraction`):

  1. `ClampOversizedMessages` — one runaway part (a giant tool call printed
     as text) truncated in place; nothing else can reach the newest message.
  2. `DeduplicateFileReads` — a file read superseded by an identical later
     read of the same range is blanked. Near-lossless, runs every request.
  3. `ClearToolResults` — older tool *results* blanked, keeping the last
     `keep_tool_pairs`. The model can re-fetch what it needs.
  4. `SummarizingCompaction` — last resort: older turns summarised by a
     cheap Groq model, recent turns and user messages kept verbatim.

`ReportContextUsage` feeds the live gauge `/context` reads; `WarnNearLimits`
injects a wrap-up note as the window fills.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    ContextUsage,
    DeduplicateFileReads,
    ReportContextUsage,
    SummarizingCompaction,
    TieredCompaction,
    WarnNearLimits,
    compact_now,
    estimate_token_count,
    resolve_context_window,
)
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW

from .config import Config

# Older tool results and older turns are cheap to summarise but the summary
# itself is a real model call — keep it on the small, reliable Groq model the
# rest of tcode's helper calls use (distill.py, verify.py), never the
# possibly-large or non-Groq primary.
SUMMARY_MODEL = "openai/gpt-oss-20b"

# WarnNearLimits starts warning at `warning_threshold * context_warn_fraction`
# of the window. Kept high on purpose: compaction already targets
# `context_compact_fraction` (0.80), so the history only climbs past ~0.87
# when compaction genuinely can't reclaim enough — which is exactly when the
# model should hear "wrap up", and not a turn sooner.
_WARN_THRESHOLD = 0.95


class ContextGauge:
    """Holds the most recent `ContextUsage` reading so `/context` can show it
    between turns. Process-global (tcode is one session per process); updated
    by `ReportContextUsage` before every model request."""

    def __init__(self) -> None:
        self.latest: ContextUsage | None = None

    def update(self, usage: ContextUsage) -> None:
        # ReportContextUsage propagates any exception raised here into the
        # run, so this stays a plain assignment — nothing that can raise.
        self.latest = usage

    def reset(self) -> None:
        self.latest = None


GAUGE = ContextGauge()


def _read_file_key(call: ToolCallPart) -> str | None:
    """Identify a `read_file` call for `DeduplicateFileReads`.

    Keyed on path **and** the (offset, limit) range: two reads only dedupe
    when they cover the same slice, so blanking the earlier one can never
    drop lines a later partial read didn't re-fetch.
    """
    if call.tool_name != "read_file":
        return None
    try:
        args = call.args_as_dict()
    except Exception:  # noqa: BLE001 - a malformed call is simply not a dedupe target
        return None
    path = args.get("path")
    if not path:
        return None
    return f"{path}@{args.get('offset', 0)}:{args.get('limit')}"


def _window_kwargs(cfg: Config) -> dict:
    """The window-resolution kwargs every fraction-taking capability accepts.

    `context_window` overrides resolution outright; `fallback_context_window`
    is only consulted when resolution fails (a google/zai id genai-prices
    doesn't record). One or the other, never a bare 200K surprise.
    """
    if cfg.context_window_override is not None:
        return {"context_window": cfg.context_window_override}
    return {"fallback_context_window": DEFAULT_CONTEXT_WINDOW}


def _summary_model(cfg: Config) -> GroqModel:
    return GroqModel(SUMMARY_MODEL, provider=GroqProvider(api_key=cfg.api_key))


def _summarizer(cfg: Config) -> SummarizingCompaction:
    return SummarizingCompaction(
        model=_summary_model(cfg),
        # Trigger fields are ignored inside TieredCompaction / compact_now,
        # but the constructor still requires one to be set.
        max_messages=1,
        keep_messages=cfg.context_keep_messages,
        keep_user_messages=True,
        receipts=True,
    )


def compaction_capabilities(cfg: Config) -> list:
    """The ordered context-management capabilities for one agent.

    Appended to `build_agent`'s capability list in place of the old flat
    ClearToolResults / WarnNearLimits / (absolute) trigger trio.
    """
    wk = _window_kwargs(cfg)
    return [
        ClampOversizedMessages(max_part_tokens=cfg.context_max_part_tokens),
        DeduplicateFileReads(file_key=_read_file_key),
        TieredCompaction(
            tiers=[
                ClearToolResults(max_tokens=1, keep_pairs=cfg.keep_tool_pairs),
                _summarizer(cfg),
            ],
            target_fraction=cfg.context_compact_fraction,
            **wk,
        ),
        # After TieredCompaction: the gauge reflects what the model will
        # actually be sent, post-compaction.
        ReportContextUsage(on_usage=GAUGE.update, **wk),
        WarnNearLimits(
            max_context_fraction=cfg.context_warn_fraction,
            warning_threshold=_WARN_THRESHOLD,
            **wk,
        ),
    ]


# --- /context and /compact helpers ---------------------------------------


@dataclass(frozen=True)
class ContextReport:
    """What `/context` renders — see `ui.show_context`."""

    model_label: str
    used_tokens: int
    window_tokens: int
    resolved: bool
    live: bool
    """True when `used_tokens` is the gauge's real last-request reading;
    False when it's a between-turns history-only estimate (no tool schemas,
    no system prompt — a floor)."""
    message_count: int
    tool_result_count: int
    compact_fraction: float
    warn_fraction: float

    @property
    def fraction(self) -> float:
        return self.used_tokens / self.window_tokens if self.window_tokens else 0.0


def _resolved_window(cfg: Config, model_label: str) -> tuple[int, bool]:
    if cfg.context_window_override is not None:
        return cfg.context_window_override, True
    window = resolve_context_window(model_label)
    if window is not None:
        return window, True
    return DEFAULT_CONTEXT_WINDOW, False


def _tool_result_count(messages: list[ModelMessage]) -> int:
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    n = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            n += sum(1 for p in msg.parts if isinstance(p, ToolReturnPart))
    return n


def describe_context(cfg: Config, messages: list[ModelMessage]) -> ContextReport:
    """Snapshot of how full the window is, for `/context`.

    Prefers the gauge's last real reading (the tokens the most recent request
    actually cost, tool schemas and system prompt included). Falls back to a
    history-only estimate when no turn has run yet this session.
    """
    model_label = f"{cfg.provider}:{cfg.model}" if cfg.provider != "groq" else cfg.model
    reading = GAUGE.latest
    if reading is not None:
        used, window, resolved, live = (
            reading.used_tokens,
            reading.window_tokens,
            reading.resolved,
            True,
        )
    else:
        window, resolved = _resolved_window(cfg, f"groq:{cfg.model}" if cfg.provider == "groq" else model_label)
        used, live = estimate_token_count(messages), False
    return ContextReport(
        model_label=model_label,
        used_tokens=used,
        window_tokens=window,
        resolved=resolved,
        live=live,
        message_count=len(messages),
        tool_result_count=_tool_result_count(messages),
        compact_fraction=cfg.context_compact_fraction,
        warn_fraction=cfg.context_warn_fraction,
    )


@dataclass(frozen=True)
class CompactionResult:
    changed: bool
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int


async def compact_history(
    cfg: Config, messages: list[ModelMessage], *, focus: str | None = None
) -> tuple[list[ModelMessage], CompactionResult]:
    """Force a summarising compaction now, for `/compact`.

    Unlike the in-run `TieredCompaction`, this always summarises regardless
    of how full the window is (that's what a user typing `/compact` is
    asking for) — unless the history is already shorter than the keep
    window, in which case `SummarizingCompaction` no-ops and `changed` is
    False. `focus` steers what the summary keeps.
    """
    if not messages:
        return messages, CompactionResult(False, 0, 0, 0, 0)

    before_n = len(messages)
    before_t = estimate_token_count(messages)
    strategy = _summarizer(cfg)
    new_messages = await compact_now(
        strategy,
        list(messages),
        model=_summary_model(cfg),
        focus=focus,
    )
    after_n = len(new_messages)
    after_t = estimate_token_count(new_messages)
    return new_messages, CompactionResult(
        changed=after_n != before_n or after_t != before_t,
        messages_before=before_n,
        messages_after=after_n,
        tokens_before=before_t,
        tokens_after=after_t,
    )
