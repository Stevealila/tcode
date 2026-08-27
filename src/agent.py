"""Assembles the Pydantic AI agent: the primary model (Groq by default, or
another provider — see _build_model) plus harness capabilities.

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
TCODE_SHELL=0 goes further and omits the Shell capability entirely — no
run_command tool exists at all, not even a restricted one — for a task
that processes untrusted content and has no legitimate reason to run
commands, matching a scoped tool list with no Bash in it rather than
trusting an allowlist to hold under prompt injection.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.groq import GroqModel, GroqModelSettings
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.settings import ModelSettings
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
from pydantic_ai_harness.subagents import ModelOption
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

from .config import Config
from .files import file_capabilities
from .guardrails import (
    WRITE_TOOLS,
    UrlLedger,
    citation_paths_exist,
    confidence_tags_need_citation,
    scope_shell_exploration,
    scope_writes_to,
)
from .web import current_time_instructions, web_capabilities


_WEB_INSTRUCTIONS = """
For a current/live fact you don't already know for certain (a price, a
score, today's news, a package's latest version, etc.), search the web and
then fetch a promising result — not shell commands like `curl`/`wget`,
which return a page's raw HTML/JS rather than what `web_fetch` gives you.
`web_fetch` takes the specific question you need answered, not just the
URL, and tells you plainly when the page doesn't have it — when that
happens (or a fetch fails outright), try another result or a different
query rather than answering from the search snippet alone or giving up
after one attempt. If you still don't have it after that, say so — never
state a specific current number, date, or source you're not actually sure
is real. If the user asks again (e.g. "how about today?"), that's a
request to actually retry, not to repeat the same "I couldn't get it"
answer unchanged.
"""


def _coder_instructions(
    scratch_dir: Path,
    web_search: bool,
    request_limit: int,
    shell: bool,
    readonly: bool,
    write_scope: list[str],
    expert_model_configured: bool,
) -> str:
    web_block = _WEB_INSTRUCTIONS if web_search else ""
    expert_block = (
        "\n\nFor one genuinely hard sub-problem inside a larger task — a "
        "tricky bug, a design decision with real tradeoffs — you can "
        "delegate_task(agent_name=..., model='expert') to run that specific "
        "delegation on a stronger model instead of the default one. Reach "
        "for it only when the difficulty is real, not for routine "
        "exploration or reading files, which the default model handles "
        "fine on its own."
        if expert_model_configured
        else ""
    )

    if readonly:
        files_line = (
            "You have read-only workspace file access — list_directory/read_file/"
            "find_files/search_files/read_and_distill — but no write_file, "
            "edit_file, or create_directory this session: those tools don't "
            "exist. You can look and reason, never change anything on disk."
        )
    elif write_scope:
        scope_list = " or ".join(s.rstrip("/") + "/" for s in write_scope)
        files_line = (
            "You have full workspace-rooted read access — read_file/list_directory/"
            f"find_files/search_files/read_and_distill — but write_file/edit_file/"
            f"create_directory only work under {scope_list}; a call outside that is "
            "rejected and asks you to retry. Read anywhere you need to for context, "
            "just keep every write inside that path."
        )
    else:
        files_line = (
            "You have full workspace-rooted file access — read_file/write_file/"
            "edit_file/list_directory/find_files/search_files/read_and_distill."
        )
    shell_line = (
        " You also have a real shell: use whatever CLI tools are appropriate "
        "(git, gh, package managers, language toolchains, etc.), don't assume "
        "you're restricted to a small fixed set."
        if shell
        else " There's no shell this session — no run_command tool exists, so "
        "don't reach for CLI equivalents (git, gh, etc.) of things your own "
        "file tools already cover."
    )
    shell_line = files_line + shell_line
    gh_block = (
        f"""

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
afterward."""
        if shell
        else ""
    )
    return f"""\
You are tcode, a system-wide coding agent running directly in the user's
terminal, in the current project's working directory. {shell_line} Be
direct and efficient: prefer taking action (reading files, running
commands, making edits) over asking the user to do it themselves. Keep
prose brief; let file diffs and command output speak for themselves. When a
task is ambiguous in a way that materially changes what you'd build, ask
one focused question before proceeding, otherwise just proceed.

Be economical with tool output: prefer your own file-system tools
(list_directory, read_file, find_files) over shell commands like `ls -R`,
`find`, or `cat` for exploring or dumping file contents, since they're
already scoped and bounded and raw shell output isn't. Don't run a broad
recursive listing when a targeted one answers the question. Never re-paste
large tool output back to the user in your reply; describe or summarize it
instead.

Your real budget for one turn is {request_limit} tool-call round-trips —
generous for a task that means reading or processing many files in
sequence (a batch job over a whole directory, say). That's expected and
normal, not something to avoid or apologize for: work through the entire
list. Don't stop partway and claim you're low on budget/tokens/quota unless
you've actually gotten close to the number above — and never state that you
checked, read, or found nothing in something you didn't actually call a
tool on; if you're stopping before finishing a multi-item task, say exactly
which items you covered and which you didn't, rather than implying full
coverage.

read_file silently truncates a large file from the middle past a few
thousand characters — real enough content just isn't there for you to see,
with nothing telling you that. For a large file (a long note, a log, a data
dump) where you need a specific answer rather than the file's literal
bytes, use read_and_distill(path, prompt) instead: state exactly what
you're looking for and it returns just that, read in full first. Keep using
read_file for anything you need to see or edit verbatim (source code,
configs) — a distilled paraphrase can't stand in for the actual text there.
{gh_block}

When asked to review, assess, or give an opinion on a codebase, that means
reading the actual contents of its most relevant source files, not just
listing the directory tree. A file tree with generic advice ("add tests",
"add CI") is not a review; cite real files, real lines, and specific
behavior you found by reading the code.

When the user's prompt spells out an exact output structure (numbered or
lettered sections, specific headers, a required closing section like a
sources list), follow it precisely rather than substituting your own
generic shape — a request that specific is not a suggestion, and
collapsing or skipping sections under time pressure is the most common way
a detailed prompt ends up under-followed. If more than one source states a
value for what looks like the same fact (a price, a rate, a date) and they
disagree, don't silently pick one: say so explicitly and name the
conflicting values, even if you go on to judge one more current or
reliable. Before characterizing something as having already happened,
been confirmed, or being settled, check the source's own tense and framing
carefully — "is set to," "scheduled for," and "expected to" all describe
something that has NOT happened yet, and restating it in the past tense is
a factual error, not a style choice. The same care applies to where a
number came from: if a source says a figure is its own internal estimate
or standing forecast, not external market pricing (or vice versa), keep
that distinction when you restate it — summarizing it away, or relabeling
an internal number as market data, misrepresents the source just as much
as getting the tense wrong does.
{web_block}{expert_block}"""


def _explorer(workspace: str | Path) -> SubAgent:
    """A read-only sub-agent for orientation, same role as Coder's built-in one."""
    agent = Agent(
        name="explorer",
        # This description, not `instructions` below, is what the parent
        # model actually sees when deciding how to phrase a delegate_task
        # call (SubAgents.get_instructions lists each delegate by
        # name+description only) — so the guidance on composing `task`
        # belongs here, not in explorer's own instructions, which only
        # shape how explorer behaves once it's already been invoked.
        # delegate_task's own `task: str` is a raw string, not a structured
        # object (a fixed harness tool signature tcode can't change), so
        # this prompt-level nudge is the only lever available: it packs a
        # weaker parent's task string with the fields it's most likely to
        # under- or over-specify otherwise.
        description=(
            "Explore the codebase and answer questions without modifying "
            "anything. The sub-agent has no access to this conversation, so "
            "state in the task: the specific question being asked, any "
            "file/path already known to be relevant, and the shape of "
            "answer expected (a yes/no, a list of paths, a summary)."
        ),
        instructions="Answer with concrete paths and evidence.",
        capabilities=[
            FileSystem(workspace, read_only=True),
            RepoContext(workspace_dir=Path(workspace)),
        ],
    )
    return SubAgent(agent)


def _build_model_for(provider: str, model_id: str, provider_api_key: str):
    """A Model instance for `model_id` on `provider`, authenticated with
    `provider_api_key` — the per-provider branch both the primary turn
    (`_build_model`) and an expert-model menu entry (see
    `_expert_models_menu`) need, since each provider takes a different API
    key and Model/Provider class pair.

    Groq is the only provider imported unconditionally (see module
    docstring): it's tcode's zero-setup default, so its client library is a
    hard dependency. Every other provider is imported lazily, right here,
    so installing tcode with just the Groq extra still works for every
    caller who never asks for one — see config.py's parse_model_spec for
    how a --model value picks a provider in the first place.
    """
    if provider == "groq":
        return GroqModel(model_id, provider=GroqProvider(api_key=provider_api_key))
    if provider == "google":
        try:
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider
        except ImportError as e:
            raise RuntimeError(
                "--model google:... needs the google-genai package, which isn't "
                'installed. Add the `google` extra (pydantic-ai-slim[groq,google]) '
                "and reinstall to use it."
            ) from e
        return GoogleModel(model_id, provider=GoogleProvider(api_key=provider_api_key))
    if provider == "zai":
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.zai import ZaiProvider
        except ImportError as e:
            raise RuntimeError(
                "--model zai:... needs the openai package, which isn't installed. "
                'Add the `zai` extra (pydantic-ai-slim[groq,zai]) and reinstall to use it.'
            ) from e
        # Z.AI is OpenAI-Chat-Completions-compatible (see ZaiProvider), not
        # a bespoke model class the way Groq/Google are.
        return OpenAIChatModel(model_id, provider=ZaiProvider(api_key=provider_api_key))
    # Unreachable: config.py's parse_model_spec only ever returns a
    # provider name from the same registry this function's branches cover.
    raise AssertionError(f"no model builder wired up for provider {provider!r}")


def _build_model(cfg: Config):
    """The primary turn's model — on whichever provider cfg.provider names."""
    return _build_model_for(cfg.provider, cfg.model, cfg.provider_api_key)


def _expert_models_menu(cfg: Config) -> dict[str, ModelOption] | None:
    """SubAgents' model menu, populated only when TCODE_EXPERT_MODEL is set.

    Reuses `SubAgents.models`/`ModelOption` rather than a hand-rolled
    second model client: a menu key can mean "different model" or "same
    model, more thinking," so this single mechanism covers both a
    stronger-model escalation and a reasoning-effort dial at once. Only one
    key today ('expert') — offered to every configured sub-agent (none set
    `models=[...]` themselves), so the parent can route any delegation to
    it, not just explorer's.
    """
    if cfg.expert_model is None:
        return None
    assert cfg.expert_provider is not None and cfg.expert_provider_api_key is not None
    model = _build_model_for(cfg.expert_provider, cfg.expert_model, cfg.expert_provider_api_key)
    return {
        "expert": ModelOption(
            model,
            description="a genuinely hard sub-problem, not routine exploration",
            settings=ModelSettings(thinking="high"),
        )
    }


def _primary_model_settings(cfg: Config) -> ModelSettings | None:
    """Reasoning-effort for the primary turn (--effort / --think / TCODE_EFFORT).

    Groq's gpt-oss models take `groq_reasoning_effort` directly. Other
    providers expose the knob differently (google_thinking_config,
    openai_reasoning_effort); until those are wired, `--effort` is a
    documented no-op there rather than an error — the CLI still accepts it
    so a multi-provider script doesn't have to special-case.
    """
    if cfg.effort is None:
        return None
    if cfg.provider == "groq":
        return GroqModelSettings(groq_reasoning_effort=cfg.effort)
    return None


def build_agent(cfg: Config) -> Agent:
    model = _build_model(cfg)
    # file_capabilities/web_capabilities' own internal distillation agents
    # are always Groq (see distill.py's DISTILL_MODEL) regardless of what
    # provider the primary model above is on — a second, always-Groq
    # provider object, deliberately separate from `model`'s own.
    groq_provider = GroqProvider(api_key=cfg.api_key)

    workspace = str(cfg.cwd)
    url_ledger = UrlLedger()
    capabilities = [
        Capability(
            instructions=[
                _coder_instructions(
                    cfg.scratch_dir,
                    cfg.web_search,
                    cfg.request_limit,
                    cfg.shell,
                    cfg.readonly,
                    cfg.write_scope,
                    cfg.expert_model is not None,
                ),
                # A weak model's sense of "now" defaults to its training
                # cutoff, not the wall clock — evaluated per run (not baked
                # in once here) so a long-lived session stays correct across
                # a day boundary. See web.py's module docstring.
                current_time_instructions,
            ]
        ),
        FileSystem(workspace, read_only=cfg.readonly),
        # home_dir enables walk-up so a monorepo-root CLAUDE.md above cfg.cwd
        # is loaded too. Precedence is ancestor-first, workspace-last, so a
        # workspace-level CLAUDE.md still wins on conflict with a home-level
        # one — see README's "How state is laid out" section.
        RepoContext(workspace_dir=cfg.cwd, home_dir=Path.home()),
        Planning(),
        # Always on, unlike web_capabilities: this is a FileSystem gap
        # (large local files), not something that needs an external
        # capability or API key. See files.py's module docstring.
        *file_capabilities(cfg, groq_provider),
    ]
    if cfg.shell:
        capabilities.append(
            Shell(
                cwd=workspace,
                allowed_commands=cfg.allowed_commands,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            )
        )
    if cfg.web_search:
        capabilities += web_capabilities(cfg, groq_provider)
    capabilities += [
        # agent_folders='agents' (the harness default, named explicitly here
        # rather than left implicit) loads a project's own
        # .claude/agents/*.md / .agents/agents/*.md as real, invocable
        # delegates alongside the hardcoded _explorer below. Disk delegates
        # get no tools unless a definition sets inherit_tools=True, which
        # caps the blast radius of a markdown file tcode didn't write
        # steering a tool-bearing agent — the trust boundary is accepted
        # deliberately, not overlooked.
        SubAgents(
            agents=[_explorer(workspace)],
            agent_folders="agents",
            # An empty menu (the default, TCODE_EXPERT_MODEL unset) means
            # delegate_task's schema drops the `model` argument entirely and
            # behavior is unchanged — see _expert_models_menu.
            models=_expert_models_menu(cfg) or {},
        ),
        # Technical backstop for the "scope exploration" instruction above:
        # observed behavior shows the model doesn't always follow it. See
        # guardrails.py's module docstring for the incident that prompted this.
        ToolGuardrail(guard=scope_shell_exploration),
        *(
            [
                # A caller-side post-hoc check (diff git status before/after,
                # revert anything outside an allowed path) can't see a write
                # into a gitignored directory at all — this fires on the
                # tool call itself, before the write happens, so it isn't.
                # See guardrails.py's module docstring.
                ToolGuardrail(
                    guard=scope_writes_to(workspace, cfg.write_scope),
                    tools=WRITE_TOOLS,
                )
            ]
            if cfg.write_scope
            else []
        ),
        # Always on: catches a write whose text mangles a URL a tool result
        # actually returned intact this run. See guardrails.py's UrlLedger
        # and its module docstring for the incident that prompted this
        # (tcode_improvements.txt's Finding 5).
        ToolGuardrail(result_guard=url_ledger.record),
        ToolGuardrail(guard=url_ledger.check_write, tools=WRITE_TOOLS),
        *(
            [ToolGuardrail(guard=citation_paths_exist(workspace), tools=WRITE_TOOLS)]
            if cfg.check_citations
            else []
        ),
        *(
            [ToolGuardrail(guard=confidence_tags_need_citation(cfg.require_citation_for), tools=WRITE_TOOLS)]
            if cfg.require_citation_for
            else []
        ),
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

    return Agent(
        model,
        model_settings=_primary_model_settings(cfg),
        capabilities=capabilities,
        # Pydantic AI's default tool-retry budget is 1, counted per tool name
        # for the whole run (not per call): a second ModelRetry from the same
        # tool anywhere in the run — e.g. web_fetch hitting a 403 on one URL,
        # then a timeout on a different one — raises UnexpectedModelBehavior
        # and kills the turn outright instead of letting the model try
        # another source. Real web research hits exactly this (flaky sites,
        # anti-bot blocks). `request_limit` (see cli.py) already caps total
        # round-trips per turn, so a higher per-tool budget doesn't risk an
        # unbounded loop — it just survives the kind of transient failures a
        # human would also just retry past.
        retries={"tools": 3},
    )
