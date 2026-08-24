"""A distilling file-read tool, for files FileSystem's `read_file` mangles.

Found by testing tcode against a real research-synthesis task (reducing many
dense daily notes into one summary — see improvements notes for the TODO
this came out of): asked it to read one real file in full and report back
specifics from throughout it. Its own answer showed the problem —

    "The tool reports that the file is 37,094 characters long, but it
    omitted the middle 35,094 characters, showing only the first ~800 and
    last ~1,200 characters."

`agent.py`'s `ToolOutputLimits` truncates every tool result, `read_file`
included, to a few thousand characters — the right default for ordinary
coding output (a grep hit, a source file), silently catastrophic for a
document running tens of thousands of characters, since whatever's in the
middle just isn't there for the model to reason about, with no signal that
anything is missing.

Two options: raise the truncation limit globally, or reach for a bigger
model with a bigger context window. Both trade away something this harness
already relies on elsewhere — a low global truncation limit is what keeps
ordinary coding turns cheap on Groq's TPM budget, and swapping models
per-task isn't something a single `Agent` does mid-run. `read_and_distill`
sidesteps both: read the file in full (bypassing the truncation entirely,
via the same sandboxed `FileSystemToolset` the main FileSystem capability
uses — not a raw open()), then hand that content to a second, cheap model
(`distill.py`, shared with `web.py`'s `web_fetch`) along with the specific
question the caller asked, so only a targeted answer re-enters the primary
model's context. A large document processed this way costs a few hundred
tokens of context instead of its full length — which also means a task that
reduces many such documents into one summary no longer has to fit all of
them, in full, in a single context window at once; each one is compressed
before it ever gets there. Offered alongside `read_file`, not instead of
it: editing a file needs its literal bytes, which a distilled paraphrase
can't give you — this tool is for when a targeted answer is what's actually
wanted, not a document too big to read at all.
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
    distill_agent = make_distill_agent(provider)

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
