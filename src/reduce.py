"""Internal map-reduce over many files (`tcode --reduce`).

The problem this exists to fix isn't technical, it's architectural. TODO 3
(see improvements notes) validated that reading a whole window of files and
reducing them to one output has to be chunked — one internal call per item,
never a single turn handling a list, because that turn reliably invents a
fake "out of budget" excuse and stops early regardless of how small the
items are. The first working version of that fix lived in the *caller* — a
bash script issuing one `tcode` subprocess per file, gluing the results
back together itself. That solved the immediate problem but got the
layering backwards: the caller had to know tcode's internal reliability
limits and work around them, which means every future caller hitting the
same wall has to rediscover and reimplement the same workaround. Compare
`--verify` (verify.py): a caller invoking it doesn't know or care that it's
three model calls internally — it looks like one command that returns one
trustworthy answer. That's the shape a fix here should have had from the
start: the chunking is tcode's problem to solve once, not each caller's
problem to solve again.

So `--reduce PATTERN` takes the validated pipeline and moves all of it
in-process:

  1. map    — one internal distillation per matched file (reusing
     `distill.py`, the same mechanism `read_and_distill` already uses),
     run concurrently — `throttle()` still paces the underlying requests,
     so concurrency here is "issue them all and let the shared limiter
     serialize what needs serializing," not a way around the rate limit.
  2. group  — only when there are more files than one map/reduce turn has
     been tested reliable for (see `GROUP_THRESHOLD`): grouped by parent
     directory, a generic structural signal (files a filesystem already
     put together are presumably related) rather than anything specific
     to one caller's domain, then each group gets its own concurrent
     digest pass before the final reduce ever sees it.
  3. reduce — one ordinary tcode turn (full capabilities, respects
     `--quiet`, can write_file if the prompt asks it to) over the
     collected map/digest output plus the caller's original prompt.

A caller with a batch task now looks exactly like a caller with a single-
file task: one `tcode --reduce PATTERN "..."` invocation, one answer out.
Nothing about matched-file count, grouping, or retries is the caller's
concern — same principle `--verify` already established, applied to the
other place this session found the same layering mistake.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai import UsageLimits
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai_harness import FileSystem

from .agent import build_agent
from .config import Config
from .distill import make_distill_agent
from .ratelimit import throttle
from .runner import run_turn

# Conservative under the ~16-item ceiling TODO 3 found reliable for a single
# reduce turn — leaves headroom rather than sitting right at the edge.
GROUP_THRESHOLD = 12

_MAX_READ_LINES = 20_000


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


async def _map_one(reader, distill_agent, cfg: Config, path: Path, extraction_prompt: str) -> tuple[str, str]:
    raw = await reader.read_file(str(path), limit=_MAX_READ_LINES)
    await throttle(cfg.rpm_state_file, cfg.max_rpm)
    distilled = await distill_agent.run(
        f"QUESTION: {extraction_prompt}\n\nFILE: {path}\n\n--- FILE CONTENT ---\n{raw}"
    )
    return str(path), str(distilled.output)


async def _digest_group(distill_agent, cfg: Config, label: str, items: list[tuple[str, str]], prompt: str) -> tuple[str, str]:
    combined = "\n\n".join(f"## {name}\n{text}" for name, text in items)
    await throttle(cfg.rpm_state_file, cfg.max_rpm)
    distilled = await distill_agent.run(
        f"QUESTION: {prompt}\n\nGROUP: {label}\n\n--- COMBINED CONTENT ---\n{combined}"
    )
    return label, str(distilled.output)


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
        *(_map_one(reader, distill_agent, cfg, f, extraction_prompt) for f in files)
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
            *(_digest_group(distill_agent, cfg, label, items, extraction_prompt) for label, items in chunks)
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
