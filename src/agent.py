"""Assembles the Pydantic AI agent: a Groq model plus harness capabilities.

This hand-assembles the same pieces `pydantic_ai_harness.Coder` composes
(FileSystem, Shell, RepoContext, Planning, a read-only explorer sub-agent,
plus context management) instead of using `Coder` as one opaque block.
Coder's built-in `ClearToolResults`/`ToolOutputLimits`/`WarnNearLimits` are
tuned for a generic ~200k-token context window, but Groq's lower tiers cap
throughput at a few thousand *tokens per minute*, a much tighter and
unrelated constraint. Reusing Coder as-is means those safety nets barely
ever fire before Groq's rate limiter does. Instantiating our own copies of
those three capabilities alongside Coder doesn't work either: Coder already
registers a `read_tool_result` tool via its internal ToolOutputLimits, and a
second instance collides with it.

Memory gives the model a persistent cross-project notebook. Step
persistence is best-effort, not load-bearing, so a version mismatch in the
harness package must not take down the CLI.

Shell command policy: by default (Config.allowed_commands == []) Shell runs
in denylist mode, blocking a short list of destructive commands (rm, dd,
mkfs, shutdown, ...) and allowing everything else, including things like
`gh`, `npm`, or `docker` that a small fixed allowlist would otherwise
block. Set TCODE_ALLOWED_COMMANDS to opt into a strict allowlist instead.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai_harness import (
    FileSystem,
    LLM_API_KEY_ENV_PATTERNS,
    Memory,
    Planning,
    RepoContext,
    Shell,
    SubAgent,
    SubAgents,
)
from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.guardrails import ToolGuardrail
from pydantic_ai_harness.memory import FileStore
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

from .config import Config
from .guardrails import scope_shell_exploration


def _coder_instructions(scratch_dir: Path) -> str:
    return f"""\
You are tcode, a system-wide coding agent running directly in the user's
terminal, in the current project's working directory. You have full
workspace-rooted file access and a real shell: use whatever CLI tools are
appropriate (git, gh, package managers, language toolchains, etc.), don't
assume you're restricted to a small fixed set. Be direct and efficient:
prefer taking action (reading files, running commands, making
edits) over asking the user to do it themselves. Keep prose brief; let file
diffs and command output speak for themselves. When a task is ambiguous in a
way that materially changes what you'd build, ask one focused question
before proceeding, otherwise just proceed.

You are running against a rate-limited API budget, so be economical with
tool output: prefer your own file-system tools (list_directory, read_file,
find_files) over shell commands like `ls -R`, `find`, or `cat` for exploring
or dumping file contents, since they're already scoped and bounded and raw
shell output isn't. Don't run a broad recursive listing when a targeted one
answers the question. Never re-paste large tool output back to the user in
your reply; describe or summarize it instead.

When the user asks about something that isn't the current workspace (e.g.
"the X repo", a package to inspect, a URL to fetch), resolve it the direct
way first — e.g. `gh repo view <name>` / `gh repo clone <name>` — rather
than exploring the current workspace root or asking the user to
disambiguate up front; a bare name is normally enough for a tool like `gh`
to resolve against the user's own account, and the lookup itself will tell
you if it's actually ambiguous. Only ask the user for a fuller identifier if
that direct attempt fails or turns up more than one real match. Once you
have the thing, clone or download it into {scratch_dir} — a subdirectory of
this workspace set aside for working copies and temp files, so you don't
scatter loose clones elsewhere in the user's directories — rather than
directly into cwd or the user's home directory. It's a normal path under
this workspace, so your own list_directory/read_file/find_files tools work
on it exactly like anywhere else in the project (relative paths like
`tcode-scratch/<name>/...` resolve fine); you don't need to clean it up
afterward.

When asked to review, assess, or give an opinion on a codebase, that means
reading the actual contents of its most relevant source files, not just
listing the directory tree. A file tree with generic advice ("add tests",
"add CI") is not a review; cite real files, real lines, and specific
behavior you found by reading the code.
"""


def _explorer(workspace: str | Path) -> SubAgent:
    """A read-only sub-agent for orientation, same role as Coder's built-in one."""
    agent = Agent(
        name="explorer",
        description="Explore the codebase and answer questions without modifying anything",
        instructions="Answer with concrete paths and evidence.",
        capabilities=[
            FileSystem(workspace, read_only=True),
            RepoContext(workspace_dir=Path(workspace)),
        ],
    )
    return SubAgent(agent)


def build_agent(cfg: Config) -> Agent:
    provider = GroqProvider(api_key=cfg.api_key)
    model = GroqModel(cfg.model, provider=provider)

    workspace = str(cfg.cwd)
    capabilities = [
        Capability(instructions=_coder_instructions(cfg.scratch_dir)),
        FileSystem(workspace),
        Shell(
            cwd=workspace,
            allowed_commands=cfg.allowed_commands,
            denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
        ),
        RepoContext(workspace_dir=cfg.cwd),
        Planning(),
        SubAgents(agents=[_explorer(workspace)], agent_folders=None),
        # Technical backstop for the "scope exploration" instruction above:
        # observed behavior shows the model doesn't always follow it. See
        # guardrails.py's module docstring for the incident that prompted this.
        ToolGuardrail(guard=scope_shell_exploration),
        # Tuned for Groq's tokens-per-minute budget rather than a generic
        # large context window; see module docstring.
        ClearToolResults(max_tokens=cfg.clear_after_tokens, keep_pairs=cfg.keep_tool_pairs),
        WarnNearLimits(max_total_tokens=cfg.clear_after_tokens * 2),
        ToolOutputLimits(
            bands=[
                Band(
                    over=cfg.tool_output_max_chars,
                    action=Truncate(max_chars=cfg.tool_output_max_chars),
                )
            ]
        ),
        Memory(FileStore(str(cfg.memory_dir))),
    ]

    try:
        from pydantic_ai_harness import StepPersistence
        from pydantic_ai_harness.step_persistence import FileStepStore

        capabilities.append(
            StepPersistence(
                store=FileStepStore(str(cfg.steps_dir)),
                agent_name=cfg.project_slug,
            )
        )
    except Exception:
        # Step persistence is a nice-to-have resilience feature; if the
        # installed harness version's API doesn't match, skip it rather
        # than fail the whole CLI.
        pass

    return Agent(model, capabilities=capabilities)
