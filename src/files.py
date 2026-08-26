"""A distilling file-read tool, for files FileSystem's `read_file` mangles.

`agent.py`'s `ToolOutputLimits` truncates every tool result, `read_file`
included, to a few thousand characters — fine for ordinary coding output (a
grep hit, a source file), but it silently drops the middle of anything
longer, with no signal to the model that content is missing. Raising the
global limit or reaching for a bigger-context model both trade away
something this harness relies on elsewhere (Groq's TPM budget stays cheap
because the limit is low; a single `Agent` doesn't swap models mid-run), so
`read_and_distill` sidesteps both: read the file in full via the same
sandboxed `FileSystemToolset` the main FileSystem capability uses, then hand
it to a second, cheap model (`distill.py`, shared with `web.py`'s
`web_fetch`) along with the caller's specific question, so only a targeted
answer re-enters the primary model's context — a large document costs a few
hundred tokens of context instead of its full length this way, which is
also what lets a task reducing many such documents into one summary avoid
needing all of them in a single context window at once. Offered alongside
`read_file`, not instead of it: editing needs literal bytes, which a
distilled paraphrase can't give.
"""

from __future__ import annotations

from pydantic_ai import Tool
from pydantic_ai.capabilities import Capability
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai_harness import FileSystem

from .config import Config
from .distill import make_distill_agent
from .ratelimit import throttle

# Comfortably above anything this has been tested against (the file that
# prompted this — 34K characters — is 439 lines) while still bounding a
# pathological input before it reaches the distillation model.
_MAX_READ_LINES = 20_000

# Below this, there's nothing worth filtering — skip the extra round-trip.
_DISTILL_SKIP_CHARS = 800


def _make_read_and_distill_tool(cfg: Config, provider: GroqProvider) -> Tool:
    # Same sandboxing (traversal rejection, symlink resolution, protected
    # patterns) as the FileSystem capability itself — built the same way it
    # builds its own toolset, so this can't become a second, differently
    # configured path into the workspace.
    reader = FileSystem(root_dir=str(cfg.cwd), max_read_lines=_MAX_READ_LINES).get_toolset()
    distill_agent = make_distill_agent(provider, cfg.distill_model)

    async def read_and_distill(path: str, prompt: str) -> str:
        raw = await reader.read_file(path, limit=_MAX_READ_LINES)
        if len(raw) < _DISTILL_SKIP_CHARS:
            return raw

        # Same shared cross-process throttle the main turn loop uses (see
        # cli.py) — this is a second Groq request the main loop doesn't know
        # about, and Groq enforces RPM per account, not per call site.
        await throttle(cfg.rpm_state_file, cfg.max_rpm)
        distilled = await distill_agent.run(f"QUESTION: {prompt}\n\nFILE: {path}\n\n--- FILE CONTENT ---\n{raw}")
        return distilled.output

    return Tool(
        read_and_distill,
        name="read_and_distill",
        description=(
            "Read a file and return just the part relevant to `prompt`, "
            "instead of its raw text — which, past a few thousand "
            "characters, read_file silently truncates from the middle. "
            "State exactly what you're looking for in `prompt`, the way "
            "you'd ask someone else to read the file for you. Reach for "
            "this over read_file for a large file (a long note, a log, a "
            "data dump) where you need a specific answer, not the file's "
            "literal bytes — keep using read_file when you need to see or "
            "edit the file's actual content."
        ),
    )


def file_capabilities(cfg: Config, provider: GroqProvider) -> list[Capability]:
    return [Capability(tools=[_make_read_and_distill_tool(cfg, provider)])]
