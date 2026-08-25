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
     digest pass before the final reduce ever sees it.
  3. reduce — one ordinary tcode turn (full capabilities, respects
     `--quiet`, can write_file if the prompt asks it to) over the
     collected map/digest output plus the caller's original prompt.

A caller with a batch task looks exactly like a caller with a single-file
task: one `tcode --reduce PATTERN "..."` invocation, one answer out.
Matched-file count, grouping, and retries are tcode's problem, not the
caller's.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai import UnexpectedModelBehavior, UsageLimits
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


async def _map_one(
    reader, distill_agent, cfg: Config, path: Path, extraction_prompt: str, *, quiet: bool
) -> tuple[str, str]:
    raw = await reader.read_file(str(path), limit=_MAX_READ_LINES)
    await throttle(cfg.rpm_state_file, cfg.max_rpm)
    try:
        distilled = await distill_agent.run(
            f"QUESTION: {extraction_prompt}\n\nFILE: {path}\n\n--- FILE CONTENT ---\n{raw}"
        )
        return str(path), str(distilled.output)
    except UnexpectedModelBehavior as e:
        # distill_agent sits at pydantic_ai's default retry budget (1) and
        # has no tools of its own, so a single hallucinated tool call here
        # is an UnexpectedModelBehavior, not a normal failure. Left
        # unguarded, this exception would propagate out of the
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
    await throttle(cfg.rpm_state_file, cfg.max_rpm)
    try:
        distilled = await distill_agent.run(
            f"QUESTION: {prompt}\n\nGROUP: {label}\n\n--- COMBINED CONTENT ---\n{combined}"
        )
        return label, str(distilled.output)
    except UnexpectedModelBehavior as e:
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
    distill_agent = make_distill_agent(provider)
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

    combined = "\n\n".join(f"## {label}\n{text}" for label, text in map_results)
    final_prompt = (
        f"{prompt}\n\n"
        f"--- COLLECTED INPUT ({len(files)} file(s) matching {pattern!r}, "
        f"already distilled — this is your material, don't re-read the "
        f"original files) ---\n{combined}"
    )

    agent = build_agent(cfg)
    return await run_turn(agent, final_prompt, message_history, usage_limits, cfg, quiet=quiet)
