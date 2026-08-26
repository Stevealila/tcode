"""Internal map-reduce over many files (`tcode --reduce`).

A single turn asked to process a whole list of files — read N raw files,
reduce them to one output — reliably invents a fake "out of budget" excuse
and stops early, regardless of how small the list or the items are. So
`--reduce PATTERN` chunks the work internally rather than asking the caller
to chunk it externally, the same shape `--verify` (verify.py) already uses
to hide its three internal model calls behind one answer:

  1. map    — one internal distillation per matched file (reusing
     `distill.py`, the same mechanism `read_and_distill` already uses),
     run concurrently — `throttle()` still paces the underlying requests,
     so concurrency here is "issue them all and let the shared limiter
     serialize what needs serializing," not a way around the rate limit.
  2. group  — only past `GROUP_THRESHOLD` files: grouped by parent
     directory, a generic structural signal (files a filesystem already
     put together are presumably related) rather than anything specific
     to one caller's domain, then each group gets its own concurrent
     digest pass before the final reduce ever sees it. If that pass still
     leaves more than `GROUP_THRESHOLD` digests (many small groups, not
     a few oversized ones — directory count, not file count, is what's
     past the cap), it folds again in further stages, batching digests
     into digests, until the count is small enough for the final reduce
     to see reliably — same digest mechanism, just applied to its own
     output as many times as the fan-in actually needs, not raising the
     per-call ceiling to match whatever showed up.
  3. reduce — one ordinary tcode turn (full capabilities, respects
     `--quiet`, can write_file if the prompt asks it to) over the
     collected map/digest output plus the caller's original prompt.
  4. coverage check — after reduce, a cheap heuristic (`_uncovered_groups`)
     flags any input group whose distinguishing keywords never surface in
     the final text at all. Advisory only (a notice, not a retry): it
     can't tell "genuinely nothing new to say" from "silently dropped,"
     but tcode_improvements.txt's Finding 3 is a real case of the latter
     reading exactly like the former with nothing to tell them apart —
     this at least makes the omission visible instead of silent.

A caller with a batch task looks exactly like a caller with a single-file
task: one `tcode --reduce PATTERN "..."` invocation, one answer out.
Matched-file count, grouping, and retries are tcode's problem, not the
caller's.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pydantic_ai import AgentRunError, ModelHTTPError, UsageLimits
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai_harness import FileSystem

from . import ui
from .agent import build_agent
from .config import Config
from .distill import make_distill_agent
from .ratelimit import throttle
from .runner import run_turn

# Conservative under the ~16-item ceiling TODO 3 found reliable for a single
# reduce turn — leaves headroom rather than sitting right at the edge.
GROUP_THRESHOLD = 12

_MAX_READ_LINES = 20_000

# What a degraded map/digest item reads as downstream (the final reduce
# prompt, or a later digest pass) when its own distillation call fails —
# see _map_one/_digest_group below.
_NO_SIGNAL = "no signal extracted (distillation call failed)"

# A single unconditional wait before retrying a distillation call that hit
# a 429 — see _distill_with_retry. Real Groq TPM 429s observed in
# production (TCODE_DISTILL_MODEL pointed at the same model already doing
# double duty as the primary/fallback reduce turn, so both drew on one
# shared per-model token budget) named a suggested wait of well under this,
# typically 1-3s — the per-minute window clears fast, so a short blind
# pause recovers the call far more often than it doesn't.
_RATE_LIMIT_RETRY_DELAY = 3.0


def discover_files(pattern: str, root: Path) -> list[Path]:
    """Resolve `pattern` relative to `root`.

    A plain glob (`**` included) matches files directly. `@listfile` — the
    `@file` convention several CLIs use for "read arguments from a file"
    (curl, gcc's `@file`, etc.) — instead reads `listfile` (any path readable
    from where tcode runs, not sandboxed to the workspace: this is tcode's
    own process constructing its file list, not the model reading it via a
    tool) as one path-or-glob per line, blank lines and `#`-comments
    ignored, each line resolved the same way a bare pattern would be. This
    is how a caller with selection logic a glob can't express — "the last
    14 days of these," not "all of these" — hands tcode an exact file list
    without tcode needing to know what "the last 14 days" means for that
    caller's files; the caller keeps owning that logic, same as it always
    did, and just tells tcode the answer.

    Returns paths relative to `root` (matching what the model's own file
    tools expect), sorted for a deterministic map order.
    """
    if pattern.startswith("@"):
        lines = Path(pattern[1:]).expanduser().read_text().splitlines()
        sub_patterns = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        seen: dict[Path, None] = {}
        for p in sub_patterns:
            for f in discover_files(p, root):
                seen[f] = None
        return sorted(seen)

    matches = sorted(root.glob(pattern)) if "*" in pattern or "?" in pattern else [root / pattern]
    return [p.relative_to(root) for p in matches if p.is_file()]


def _group_key(path: Path) -> str:
    """Parent directory as the grouping signal — see module docstring."""
    return str(path.parent)


_WORD_SPLIT = re.compile(r"[/_\-]+")
_MIN_KEYWORD_LEN = 3


def _group_keywords(group_keys: list[str]) -> dict[str, set[str]]:
    """Path segments that distinguish one group's key from the others.

    A segment shared by every group ("profiles", "daily", a common parent
    directory) is structural, not identifying content, so it's subtracted
    out before comparing against the final answer below — otherwise every
    group would trivially "match" on words the final answer was always
    going to contain regardless of what it actually covered.
    """
    seg_sets = {k: {s for s in _WORD_SPLIT.split(k.lower()) if s} for k in group_keys}
    common = set.intersection(*seg_sets.values()) if len(seg_sets) > 1 else set()
    return {k: (segs - common) or segs for k, segs in seg_sets.items()}


def _last_write_content(message_history: list[ModelMessage]) -> str:
    """The content of the most recent write_file/edit_file call, if any.

    The reduce prompt explicitly asks the model to call write_file rather
    than print the rollup as its reply text (see run_reduce's final_prompt)
    — and a model that follows that instruction leaves a chat reply that's
    just a short acknowledgment ("Rollup written to X"), not the rollup
    itself. Checking coverage against `final_text` alone (see
    `_uncovered_groups`) would then flag every entity as unmentioned
    regardless of how completely the actual written content covers them —
    a false positive on the exact well-behaved case the prompt asked for.
    Pulling the real written text back out of the tool call it came from is
    what makes the coverage check mean anything once a model does this.
    """
    for msg in reversed(message_history):
        if not isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if isinstance(part, ToolCallPart) and part.tool_name in ("write_file", "edit_file"):
                try:
                    args = part.args_as_dict()
                except Exception:
                    return ""
                return str(args.get("content", args.get("new_text", "")))
    return ""


def _uncovered_groups(files: list[Path], covered_text: str) -> list[str]:
    """Group keys whose distinguishing keywords never surface in `covered_text`.

    A cheap, generic stand-in for "was every input actually used in the
    synthesis" — see tcode_improvements.txt's Finding 3: --reduce silently
    dropped an entire entity's document while the final answer claimed
    outright that no other entity had anything new. This can't tell *why*
    a group goes unmentioned (genuinely nothing new to say vs. silently
    dropped) — it only flags the group so a human or caller can check,
    rather than letting both cases read identically. `covered_text` should
    be the model's reply plus whatever it actually wrote to disk (see
    `_last_write_content`) — checking the reply alone misses everything a
    well-behaved write_file call put only in the file.
    """
    keys = sorted({_group_key(f) for f in files})
    keywords = _group_keywords(keys)
    lowered = covered_text.lower()
    return [
        key
        for key in keys
        if (kws := [k for k in keywords[key] if len(k) >= _MIN_KEYWORD_LEN])
        and not any(re.search(rf"\b{re.escape(k)}\b", lowered) for k in kws)
    ]


async def _distill_with_retry(distill_agent, cfg: Config, prompt: str) -> str:
    """Run one distill_agent call, absorbing a single transient 429 with a
    short wait before retrying once — see _RATE_LIMIT_RETRY_DELAY. Blind
    (no parsing of the provider's own retry-after text, a message format
    Groq doesn't guarantee stays stable) but effective in practice: this
    is the one failure shape that's genuinely worth retrying, unlike a
    hallucinated tool call or a structured-output rejection the same call
    would just reproduce.

    Raises whatever the second attempt raises (or the first, for anything
    that isn't a 429) — the caller's own AgentRunError handling is
    unchanged for every other failure shape.
    """
    try:
        distilled = await distill_agent.run(prompt)
        return str(distilled.output)
    except ModelHTTPError as e:
        if e.status_code != 429:
            raise
        await asyncio.sleep(_RATE_LIMIT_RETRY_DELAY)
        # Same shared cross-process throttle as the call this is retrying —
        # this is a second Groq request the main loop doesn't know about.
        await throttle(cfg.groq_rpm_state_file, cfg.groq_max_rpm)
        distilled = await distill_agent.run(prompt)
        return str(distilled.output)


async def _map_one(
    reader, distill_agent, cfg: Config, path: Path, extraction_prompt: str, *, quiet: bool
) -> tuple[str, str]:
    raw = await reader.read_file(str(path), limit=_MAX_READ_LINES)
    # distill_agent is always Groq (see distill.py's DISTILL_MODEL) regardless
    # of which provider the primary/final-reduce model is on — pace against
    # Groq's own budget specifically, not cfg's active-provider one.
    await throttle(cfg.groq_rpm_state_file, cfg.groq_max_rpm)
    try:
        text = await _distill_with_retry(
            distill_agent, cfg, f"QUESTION: {extraction_prompt}\n\nFILE: {path}\n\n--- FILE CONTENT ---\n{raw}"
        )
        return str(path), text
    except AgentRunError as e:
        # distill_agent sits at pydantic_ai's default retry budget (1) and
        # has no tools of its own, so this is either a hallucinated tool
        # call (UnexpectedModelBehavior) or the provider's own API rejecting
        # the generation outright (ModelHTTPError — observed at high
        # concurrent group counts: Groq's structured-output parser
        # returning a 400 "output_parse_failed" for this model under load).
        # AgentRunError is the base both share, and every subclass means
        # the same thing here: this one call didn't produce a usable
        # answer. Left unguarded, either would propagate out of the
        # asyncio.gather() below and take down every other file's
        # already-completed distillation with it. Degrading just this one
        # item keeps the rest of the batch alive — the final reduce turn
        # sees one weaker input instead of the caller getting nothing.
        ui.print_notice(
            f"note: distillation failed for {path} ({e}) — degrading this "
            "file to \"no signal extracted\" instead of losing the whole batch.",
            quiet=quiet,
        )
        return str(path), _NO_SIGNAL


async def _digest_group(
    distill_agent, cfg: Config, label: str, items: list[tuple[str, str]], prompt: str, *, quiet: bool
) -> tuple[str, str]:
    combined = "\n\n".join(f"## {name}\n{text}" for name, text in items)
    # distill_agent is always Groq (see distill.py's DISTILL_MODEL) regardless
    # of which provider the primary/final-reduce model is on — pace against
    # Groq's own budget specifically, not cfg's active-provider one.
    await throttle(cfg.groq_rpm_state_file, cfg.groq_max_rpm)
    try:
        text = await _distill_with_retry(
            distill_agent, cfg, f"QUESTION: {prompt}\n\nGROUP: {label}\n\n--- COMBINED CONTENT ---\n{combined}"
        )
        return label, text
    except AgentRunError as e:
        # Same reasoning as _map_one's — see its comment for why this is
        # AgentRunError, not just UnexpectedModelBehavior.
        ui.print_notice(
            f"note: digest failed for group {label} ({e}) — degrading this "
            "group to \"no signal extracted\" instead of losing the whole batch.",
            quiet=quiet,
        )
        return label, _NO_SIGNAL


async def run_reduce(
    cfg: Config, prompt: str, pattern: str, message_history: list, usage_limits: UsageLimits, *, quiet: bool
) -> tuple[list, str]:
    """Returns (message_history, final_text), same shape as run_turn."""
    files = discover_files(pattern, cfg.cwd)
    if not files:
        return message_history, f"No files matched {pattern!r} under {cfg.cwd}."

    provider = GroqProvider(api_key=cfg.api_key)
    reader = FileSystem(root_dir=str(cfg.cwd), max_read_lines=_MAX_READ_LINES).get_toolset()
    distill_agent = make_distill_agent(provider, cfg.distill_model)
    extraction_prompt = f"Extract everything relevant to answering this: {prompt}"

    map_results = await asyncio.gather(
        *(_map_one(reader, distill_agent, cfg, f, extraction_prompt, quiet=quiet) for f in files)
    )

    if len(map_results) > GROUP_THRESHOLD:
        groups: dict[str, list[tuple[str, str]]] = {}
        for path, (name, text) in zip(files, map_results, strict=True):
            groups.setdefault(_group_key(path), []).append((name, text))

        # A single oversized group (files bunched in one directory, more of
        # them than one digest pass is reliable for) gets arithmetically
        # chunked — no meaningful sub-key left at this point, just keeping
        # every digest call under the same tested ceiling.
        chunks: list[tuple[str, list[tuple[str, str]]]] = []
        for key, items in groups.items():
            if len(items) <= GROUP_THRESHOLD:
                chunks.append((key, items))
            else:
                for i in range(0, len(items), GROUP_THRESHOLD):
                    chunks.append((f"{key}[{i}:{i + GROUP_THRESHOLD}]", items[i : i + GROUP_THRESHOLD]))

        map_results = await asyncio.gather(
            *(
                _digest_group(distill_agent, cfg, label, items, extraction_prompt, quiet=quiet)
                for label, items in chunks
            )
        )

    # The pass above caps each GROUP's raw-file count at GROUP_THRESHOLD, but
    # not the total number of groups reaching the final reduce turn — a
    # caller with many small directories (many groups, few files each) sails
    # straight through that cap with the group *count* itself still past
    # what one reduce turn is reliable for. Confirmed at production scale:
    # the primary model held through 20-40 groups but was only ~1/3 reliable
    # at 80 — the same "prints the tool call as text" failure --write's own
    # fallback exists to catch, just at a fan-in real callers weren't near
    # yet. Folding here, not raising GROUP_THRESHOLD itself, keeps every
    # individual digest call at the same tested-safe size regardless of how
    # many there end up being: each pass digests batches of up to
    # GROUP_THRESHOLD *digests* into one, same _digest_group used above,
    # applied to its own output. GROUP_THRESHOLD (12) means this halves-ish
    # the count each pass, so even a pathological fan-in converges in a
    # couple of stages, not a long or unbounded chain.
    stage = 2
    while len(map_results) > GROUP_THRESHOLD:
        stage_chunks = [
            (f"stage{stage}[{i}:{i + GROUP_THRESHOLD}]", map_results[i : i + GROUP_THRESHOLD])
            for i in range(0, len(map_results), GROUP_THRESHOLD)
        ]
        map_results = await asyncio.gather(
            *(
                _digest_group(distill_agent, cfg, label, items, extraction_prompt, quiet=quiet)
                for label, items in stage_chunks
            )
        )
        stage += 1

    combined = "\n\n".join(f"## {label}\n{text}" for label, text in map_results)
    final_prompt = (
        f"{prompt}\n\n"
        f"--- COLLECTED INPUT ({len(files)} file(s) matching {pattern!r}, "
        f"already distilled — this is your material, don't re-read the "
        f"original files) ---\n{combined}"
    )

    agent = build_agent(cfg)
    message_history, final_text = await run_turn(
        agent, final_prompt, message_history, usage_limits, cfg, quiet=quiet
    )

    covered_text = final_text + "\n" + _last_write_content(message_history)
    missing = _uncovered_groups(files, covered_text)
    if missing:
        # Observed live: the notice alone doesn't stop the false claim it's
        # warning about — a rollup can state "no other entity introduced new
        # material" in the very file the notice is flagging as omitting one.
        # One retry, same shape as _distill_with_retry's single-retry-then-
        # give-up: ask the model to explicitly address each missing group
        # before accepting the answer, reusing message_history so it has its
        # own prior turn (and the already-distilled input) as context rather
        # than re-deriving the whole prompt from scratch.
        ui.print_notice(
            f"note: the final answer doesn't mention {len(missing)} of "
            f"{len({_group_key(f) for f in files})} input group(s) at all "
            f"({', '.join(missing)}) — asking the model to address them "
            "explicitly before accepting this answer.",
            quiet=quiet,
        )
        retry_prompt = (
            "Your answer above doesn't mention " + ", ".join(missing) + " at "
            "all. Revise it now: for each of those, either incorporate what "
            "changed (using the material already given to you above) or "
            "state explicitly that nothing new was found for it — don't "
            "silently drop it. Write the corrected, complete answer the same "
            "way you wrote the first one, including calling write_file "
            "again if that's how you saved it the first time."
        )
        message_history, final_text = await run_turn(
            agent, retry_prompt, message_history, usage_limits, cfg, quiet=quiet
        )
        covered_text = final_text + "\n" + _last_write_content(message_history)
        still_missing = _uncovered_groups(files, covered_text)
        if still_missing:
            ui.print_notice(
                f"note: after one retry, the answer still doesn't mention "
                f"{len(still_missing)} group(s) ({', '.join(still_missing)}) "
                "— giving up on forcing coverage; treat this answer as "
                "potentially incomplete.",
                quiet=quiet,
            )

    return message_history, final_text
