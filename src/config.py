"""Configuration and on-disk layout for tcode.

Global state lives under ``~/.tcode``, with per-project subdirectories
keyed by the absolute, slugified path of the directory the tool was
launched from. Memory is the one exception: it stays global, so the agent
carries what it knows about you into every project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_REQUEST_LIMIT = 50
# A single tool return over this many characters is head/tail-truncated as it
# enters history (a runaway `ls -R`, a whole-file `cat`). Generous by design:
# the aggregate is the compaction pipeline's job (see context.py), this only
# stops one call from dominating a single request.
DEFAULT_TOOL_OUTPUT_MAX_CHARS = 8000
# How many recent tool-call/result pairs ClearToolResults keeps in full when
# it fires; older ones are blanked (re-fetchable on demand).
DEFAULT_KEEP_TOOL_PAIRS = 6
# Compaction runs once the estimated history reaches this fraction of the
# model's real context window; summarization only if clearing tool results
# first isn't enough to get back under it. A fraction, not an absolute token
# count, so it stays correct across a /model switch to any size of model.
DEFAULT_CONTEXT_COMPACT_FRACTION = 0.80
# The model is told (an injected note) to wrap up as it nears this fraction.
DEFAULT_CONTEXT_WARN_FRACTION = 0.92
# Recent messages SummarizingCompaction keeps verbatim past the summary.
DEFAULT_CONTEXT_KEEP_MESSAGES = 12
# A single message part (response text, tool-call args) larger than this many
# estimated tokens is clamped in place — the "model printed a giant tool
# call as text" failure that nothing else in the pipeline can reach.
DEFAULT_MAX_PART_TOKENS = 24000
DEFAULT_MAX_RPM = 30
DEFAULT_WEB_SEARCH = True
DEFAULT_SHELL = True
DEFAULT_READONLY = False
DEFAULT_MEMORY = True

GLOBAL_DIR = Path.home() / ".tcode"

# Groq is the zero-setup default this whole tool is built around (one free
# key, see README), but the model that does best on a given task isn't
# always hosted there — Groq's own catalog is small and changes over time.
# `--model provider:model_id` addresses a model on a different backend
# instead; a bare model id with no recognized `provider:` prefix (every
# existing config, script, and habit predating this) is still read as a
# Groq model id, unchanged. Extend this dict (plus the matching branch in
# agent.py's build_agent) to wire up another backend — the api-key env var
# and per-provider RPM default are the only two pieces of config a new one
# needs here.
_PROVIDER_API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    # Z.AI's own env var name (ZaiProvider reads it directly) — kept as-is
    # rather than a tcode-specific name so a key already set for another
    # pydantic_ai-based tool just works here too.
    "zai": "ZAI_API_KEY",
}

# Each provider enforces its own account-wide rate limit, so a shared
# default tuned for Groq's tier would either throttle a roomier provider
# for no reason or, worse, undershoot a tighter one and draw real 429s.
# TCODE_MAX_RPM still overrides this outright when set, same as before.
_DEFAULT_MAX_RPM_BY_PROVIDER = {
    "groq": 30,
    # Google AI Studio's free tier is a flat 15 RPM per project as of
    # 2026-08; conservative on purpose since Google no longer publishes one
    # authoritative number and per-project limits vary.
    "google": 15,
    # Z.AI's free GLM-*-Flash models publish a daily request cap, not an
    # official RPM figure — 15 is a conservative guess pending real usage,
    # same posture as google's above.
    "zai": 15,
}

# Models tcode refuses to run — anywhere: primary, verifier, distiller, or
# expert sub-agent. Matched case-insensitively as a substring of the bare
# model id. Qwen is on the list after proving unreliable at structured
# output in real use: it emits raw tool-call envelopes as prose (see cli.py's
# leaked-tool-call recovery), rejects documented reasoning-effort values (see
# distill.py), and has produced fabricated figures in otherwise-clean output.
# A bad verifier or distiller is worse than none — it silently vetoes or
# corrupts good primary answers — so this is enforced at config load, not
# left to habit.
_BANNED_MODEL_SUBSTRINGS = ("qwen", "qwq", "tongyi")


def _reject_banned_model(model_id: str, source: str) -> None:
    """Raise ConfigError if `model_id` is on the permanent block list."""
    low = model_id.lower()
    for bad in _BANNED_MODEL_SUBSTRINGS:
        if bad in low:
            raise ConfigError(
                f"{source} requests model {model_id!r}, which tcode will not run: "
                f"{bad!r} is on the permanent block list (unreliable structured "
                f"output in practice). Choose another model — on Groq, "
                f"openai/gpt-oss-120b or openai/gpt-oss-20b."
            )


def parse_model_spec(raw: str, *, source: str = "--model") -> tuple[str, str]:
    """Split a --model value into (provider, model_id).

    Groq model ids never contain `:` (they're bare names or `org/name`), so
    a bare string is unambiguous as "no prefix, assume groq" — the only
    behavior that existed before this and the only one most callers will
    ever hit. `provider:model_id` opts into a different backend; an
    unrecognized prefix is a config error rather than being silently folded
    into the model id (better than a confusing "model not found" from
    Groq's API for a caller who just mistyped the provider name).
    """
    if ":" in raw:
        prefix, _, rest = raw.partition(":")
        if rest:
            if prefix not in _PROVIDER_API_KEY_ENV:
                known = ", ".join(sorted(_PROVIDER_API_KEY_ENV))
                raise ConfigError(f"Unknown provider {prefix!r} in --model {raw!r}. Known providers: {known}.")
            _reject_banned_model(rest, source)
            return prefix, rest
    _reject_banned_model(raw, source)
    return "groq", raw


def _rpm_for(provider: str) -> tuple[int, Path]:
    """(max_rpm, state_file) for `provider`'s own shared rate-limit budget.

    TCODE_MAX_RPM, if set, overrides Groq's specifically — that variable
    predates multi-provider support, and every existing use of it means "my
    Groq account". A non-Groq provider gets its own conservative default
    from _DEFAULT_MAX_RPM_BY_PROVIDER instead, since a different account has
    a different limit entirely and no existing config was ever tuning it.
    Each provider gets its own state file so two providers' shared
    cross-process throttles never interleave; Groq keeps the original
    unsuffixed path so an existing ~/.tcode/rpm_state isn't orphaned.
    """
    if provider == "groq":
        rpm = int(os.environ.get("TCODE_MAX_RPM", DEFAULT_MAX_RPM))
        state_file = GLOBAL_DIR / "rpm_state"
    else:
        rpm = _DEFAULT_MAX_RPM_BY_PROVIDER.get(provider, DEFAULT_MAX_RPM)
        state_file = GLOBAL_DIR / f"rpm_state_{provider}"
    return rpm, state_file


def _slugify(path: Path) -> str:
    """Turn an absolute path into a filesystem-safe slug, e.g.

    /home/alice/some-project -> home-alice-some-project

    The leading separator is stripped so the slug never starts with `-`: a
    `~/.tcode/projects/-home-alice-x` directory can't be `cd`'d into or passed
    to most commands without a `--` guard, which made the state dir a chore to
    inspect. `_migrate_legacy_project_dirs` renames the old `-`-prefixed dirs.
    """
    resolved = str(path.resolve()).strip(os.sep)
    slug = resolved.replace(os.sep, "-")
    return slug or "root"


def _migrate_legacy_project_dirs(projects_dir: Path) -> None:
    """One-shot rename of every `-`-prefixed project dir to the current scheme.

    Runs on each startup (a single `iterdir`), not just for the current
    project, so the whole `~/.tcode/projects/` tree stops being a pile of
    `cd`-hostile `-name` directories after one more launch. Best-effort: a dir
    whose de-dashed name is already taken, or that won't rename, is left as-is.
    """
    try:
        entries = list(projects_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        name = entry.name
        if not entry.is_dir() or not name.startswith("-") or len(name) < 2:
            continue
        target = entry.with_name(name[1:])
        if not target.exists():
            try:
                entry.rename(target)
            except OSError:
                pass


def _load_env_files(cwd: Path) -> None:
    """Load .env files with precedence: real shell env > project .env > global .env.

    load_dotenv(..., override=False) never overwrites a key already set in
    os.environ, so loading project-local first and global second gives that
    precedence order without touching anything already exported.
    """
    project_env = cwd / ".env"
    if project_env.is_file():
        load_dotenv(project_env, override=False)
    global_env = GLOBAL_DIR / ".env"
    if global_env.is_file():
        load_dotenv(global_env, override=False)


@dataclass
class Config:
    cwd: Path
    api_key: str
    model: str
    # "groq" unless --model used a provider:model_id prefix — see
    # parse_model_spec. build_agent (agent.py) branches on this to pick the
    # right Model/Provider class; everything Groq-only by design (the
    # map/digest distillation model, --verify's compare/default-verifier
    # model) ignores it and keeps using `api_key` regardless, since those
    # call sites hardcode Groq model ids independent of the primary model.
    provider: str
    # The API key for `provider` specifically — same value as `api_key`
    # when provider == "groq" (the common case), a different env var's
    # value otherwise. `api_key` itself always stays the Groq key: several
    # call sites need it even when the primary model isn't on Groq.
    provider_api_key: str
    request_limit: int
    allowed_commands: list[str]
    shell: bool
    readonly: bool
    # True when this process is running inside the bubblewrap/firejail
    # sandbox it re-exec'd itself into (--sandbox / TCODE_SANDBOX=1) — the
    # honest signal for the banner, set by sandbox.py before re-exec. Not a
    # request flag: it means "the boundary is actually in place".
    sandboxed: bool
    # Reasoning-effort for the *primary* turn's model: "low" | "medium" |
    # "high", or None to leave the model at its own default. Set by
    # --effort / --think (cli.py) or TCODE_EFFORT. Distinct from the
    # `expert` sub-agent menu (agent.py), which only affects delegations.
    # Applied as GroqModelSettings(groq_reasoning_effort=...) in build_agent;
    # a no-op on providers that don't expose the knob (google/zai) for now.
    effort: str | None
    memory_enabled: bool
    tool_output_max_chars: int
    keep_tool_pairs: int
    # Context-window management — see context.py. All fractions are of the
    # model's real window, resolved per request, so they survive a /model
    # switch. context_window_override forces the window for a model
    # genai-prices doesn't know (some google/zai ids); None = resolve it.
    context_compact_fraction: float
    context_warn_fraction: float
    context_keep_messages: int
    context_max_part_tokens: int
    context_window_override: int | None
    max_rpm: int
    groq_max_rpm: int
    web_search: bool
    tavily_api_key: str | None
    write_scope: list[str]
    check_citations: bool
    require_citation_for: list[str]
    distill_model: str | None
    expert_model: str | None
    expert_provider: str | None
    expert_provider_api_key: str | None
    rpm_state_file: Path
    groq_rpm_state_file: Path
    project_slug: str
    project_dir: Path
    memory_dir: Path
    skills_dir: Path
    sessions_dir: Path
    steps_dir: Path
    scratch_dir: Path
    history_file: Path
    latest_session_file: Path

    @property
    def project_label(self) -> str:
        return str(self.cwd)


class ConfigError(RuntimeError):
    """Raised when required configuration (like the API key) is missing."""


_EFFORT_LEVELS = ("low", "medium", "high")


def load_config(
    cwd: Path | None = None,
    model_override: str | None = None,
    effort_override: str | None = None,
) -> Config:
    cwd = (cwd or Path.cwd()).resolve()

    _load_env_files(cwd)

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "GROQ_API_KEY is not set.\n\n"
            f"Add it to a .env file in this directory ({cwd / '.env'}) or to "
            f"your global config ({GLOBAL_DIR / '.env'}):\n\n"
            "  GROQ_API_KEY=gsk_...\n\n"
            "Get a key at https://console.groq.com/keys"
        )

    model_spec = model_override or os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_MODEL
    provider, model = parse_model_spec(model_spec)
    if provider == "groq":
        provider_api_key = api_key
    else:
        key_env = _PROVIDER_API_KEY_ENV[provider]
        provider_api_key = os.environ.get(key_env, "").strip()
        if not provider_api_key:
            raise ConfigError(
                f"--model {model_spec!r} needs {key_env}, which is not set.\n\n"
                f"Add it to a .env file in this directory ({cwd / '.env'}) or to "
                f"your global config ({GLOBAL_DIR / '.env'}):\n\n"
                f"  {key_env}=...\n\n"
                "(GROQ_API_KEY stays required regardless — tcode's own "
                "distillation and --verify passes are Groq-only by design.)"
            )
    request_limit = int(os.environ.get("GROQ_REQUEST_LIMIT", DEFAULT_REQUEST_LIMIT))

    # --effort / --think (via effort_override) beats TCODE_EFFORT; empty ->
    # None (model's own default). Validate here so a typo fails fast rather
    # than being silently forwarded to the provider and rejected mid-turn.
    raw_effort = (effort_override or os.environ.get("TCODE_EFFORT", "")).strip().lower()
    if raw_effort and raw_effort not in _EFFORT_LEVELS:
        raise ConfigError(
            f"invalid effort {raw_effort!r} — use one of {', '.join(_EFFORT_LEVELS)} "
            "(--effort low|medium|high, or --think for high)."
        )
    effort = raw_effort or None

    # An empty list here means "denylist mode": Shell blocks a short list of
    # destructive commands (rm, dd, mkfs, shutdown, ...) and allows
    # everything else. Set TCODE_ALLOWED_COMMANDS to flip to strict
    # allowlist mode instead, e.g. "git,python,pytest,ruff".
    raw_allowed = os.environ.get("TCODE_ALLOWED_COMMANDS", "").strip()
    allowed_commands = [c.strip() for c in raw_allowed.split(",") if c.strip()]

    # Denylist/allowlist mode above still leaves *some* shell — for a task
    # that processes untrusted content (a web page, an arbitrary file) and
    # has no legitimate reason to run commands at all, that's still a
    # prompt-injection target. TCODE_SHELL=0 removes the Shell capability
    # entirely: no run_command/start_command tool exists for the model to
    # be tricked into calling, matching a scoped `--allowedTools` list with
    # no Bash in it.
    raw_shell = os.environ.get("TCODE_SHELL", "").strip().lower()
    shell = raw_shell not in ("0", "false", "no") if raw_shell else DEFAULT_SHELL

    # Off by default (most sessions want to edit files), opt-in for a task
    # that should look but never touch: a diagnosis, a decision, anything
    # where the model reading the workspace is the point but write_file/
    # edit_file/create_directory existing at all is not a risk worth taking.
    # Shell (if also enabled) is unaffected — this scopes the FileSystem
    # tools only, matching FileSystem's own read_only=True.
    raw_readonly = os.environ.get("TCODE_READONLY", "").strip().lower()
    readonly = raw_readonly not in ("", "0", "false", "no") if raw_readonly else DEFAULT_READONLY

    # The global cross-project memory notebook (~/.tcode/memory). On by
    # default. TCODE_MEMORY=0 omits the capability entirely — no
    # write_memory/read_memory tools, nothing injected — for a short,
    # repeated, headless run that has no durable facts worth carrying
    # between sessions and, with a weak model, tends to dump its own
    # injected context back into the notebook instead.
    raw_memory = os.environ.get("TCODE_MEMORY", "").strip().lower()
    memory_enabled = raw_memory not in ("0", "false", "no") if raw_memory else DEFAULT_MEMORY

    # Set by sandbox.py on the process it re-execs into a bubblewrap/firejail
    # jail; absent in a normal run. Purely informational here (the banner).
    sandboxed = os.environ.get("TCODE_SANDBOX_ACTIVE") == "1"

    # Context-window management — see context.py's module docstring for the
    # two-limits distinction (window size vs. Groq's tokens-per-minute rate
    # limit, which is ratelimit.throttle()'s job, not this). `ls -R` on a
    # real repo still gets capped as it enters history; the aggregate is
    # handled by fraction-of-window triggers now, not a flat token count.
    tool_output_max_chars = int(
        os.environ.get("TCODE_TOOL_OUTPUT_LIMIT", DEFAULT_TOOL_OUTPUT_MAX_CHARS)
    )
    keep_tool_pairs = int(os.environ.get("TCODE_KEEP_TOOL_PAIRS", DEFAULT_KEEP_TOOL_PAIRS))

    def _fraction(env: str, default: float) -> float:
        raw = os.environ.get(env, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError as e:
            raise ConfigError(f"{env}={raw!r} is not a number.") from e
        # Floored at 0.25: below that, the fixed overhead (system prompt,
        # tool schemas, the minimum keep window) already exceeds the target,
        # so compaction can never get under it and just re-summarizes every
        # turn — burning the request budget and wiping the model's memory of
        # what it just did.
        if not 0.25 <= value <= 1.0:
            raise ConfigError(
                f"{env}={raw!r} must be a fraction of the context window between 0.25 and 1.0."
            )
        return value

    context_compact_fraction = _fraction("TCODE_CONTEXT_COMPACT_FRACTION", DEFAULT_CONTEXT_COMPACT_FRACTION)
    context_warn_fraction = _fraction("TCODE_CONTEXT_WARN_FRACTION", DEFAULT_CONTEXT_WARN_FRACTION)
    context_keep_messages = int(
        os.environ.get("TCODE_CONTEXT_KEEP_MESSAGES", DEFAULT_CONTEXT_KEEP_MESSAGES)
    )
    context_max_part_tokens = int(os.environ.get("TCODE_MAX_PART_TOKENS", DEFAULT_MAX_PART_TOKENS))
    raw_window = os.environ.get("TCODE_CONTEXT_WINDOW", "").strip()
    context_window_override: int | None = None
    if raw_window:
        try:
            context_window_override = int(raw_window)
        except ValueError as e:
            raise ConfigError(f"TCODE_CONTEXT_WINDOW={raw_window!r} is not an integer.") from e
        if context_window_override < 1:
            raise ConfigError("TCODE_CONTEXT_WINDOW must be a positive token count.")

    # Requests-per-minute is a separate, harder limit from tokens-per-minute,
    # and each provider enforces it per account, not per process — the
    # throttle state lives in one shared global file per provider, not per
    # project, so it applies across every tcode invocation hitting that
    # account. 0 disables it (TCODE_MAX_RPM only, and only for Groq — see
    # _rpm_for).
    #
    # Two separate budgets, not one: distill.py's map/digest passes and
    # verify.py's compare/default-verifier passes are Groq-only by design
    # regardless of what the primary model is (see the `provider` field
    # above), so they must keep pacing against Groq's own limit even when
    # the primary model has moved to a different provider — otherwise a
    # primary model on a roomier (or tighter) provider would silently pace
    # Groq's shared account wrong. `max_rpm`/`rpm_state_file` track
    # whichever provider is actually active (the primary model's own
    # throttle, used by runner.py); `groq_max_rpm`/`groq_rpm_state_file`
    # always track Groq specifically. The two are identical whenever
    # provider == "groq" — the common case — so this changes nothing for
    # any existing single-provider setup.
    max_rpm, rpm_state_file = _rpm_for(provider)
    groq_max_rpm, groq_rpm_state_file = _rpm_for("groq")

    # Web search/fetch: on by default, zero setup (DuckDuckGo search + a
    # distilling fetch, see web.py). Set TCODE_WEB_SEARCH=0 to turn it off,
    # e.g. if DuckDuckGo is rate-limiting or a project shouldn't touch the
    # live web at all.
    raw_web_search = os.environ.get("TCODE_WEB_SEARCH", "").strip().lower()
    web_search = raw_web_search not in ("0", "false", "no") if raw_web_search else DEFAULT_WEB_SEARCH

    # Optional upgrade over DuckDuckGo's free-text scrape: Tavily is a
    # search API built for LLM/agent consumption (clean extracted content,
    # a finance/news topic mode) rather than raw SERP snippets. Free tier,
    # no card required — get a key at https://app.tavily.com. Auto-detected
    # by presence, same as GROQ_API_KEY; DuckDuckGo is the fallback when unset.
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "").strip() or None

    # Restricts write_file/edit_file/create_directory to one or more path
    # prefixes (comma-separated, relative to cwd) without touching read
    # access — for a caller whose model decides WHERE to write, not just
    # whether (story_intake.sh-shaped), where trusting the prompt's "write
    # only under X" rule alone leaves an opening for a prompt-injected page
    # to steer a write anywhere else the FileSystem tools reach. Unset (the
    # default) leaves writes unrestricted, same as today. See
    # guardrails.py's scope_writes_to for why this has to live at the tool
    # layer rather than as a caller-side post-hoc git-diff check.
    raw_write_scope = os.environ.get("TCODE_WRITE_SCOPE", "").strip()
    write_scope = [s.strip() for s in raw_write_scope.split(",") if s.strip()]

    # Off by default: rejecting a write that cites a backtick-quoted,
    # nonexistent source path is exactly right for a caller whose whole
    # output is citations of files it was handed (a research rollup), but a
    # generic coding session legitimately proposes paths that don't exist
    # *yet* — a file to create next — which this can't tell apart from a
    # fabricated citation. See guardrails.py's citation_paths_exist.
    raw_check_citations = os.environ.get("TCODE_CHECK_CITATIONS", "").strip().lower()
    check_citations = raw_check_citations not in ("", "0", "false", "no")

    # Unset (the default) means no confidence-tag vocabulary at all, so the
    # guard it wires up is a no-op — most tcode tasks don't tag confidence
    # levels in their output. A caller whose own prompt defines tags like
    # [CONFIRMED]/[MEASURED] and asks the model to use them sets this to
    # require each one carry a real citation on the same line, or the write
    # is rejected. See guardrails.py's confidence_tags_need_citation.
    raw_require_citation_for = os.environ.get("TCODE_REQUIRE_CITATION_FOR", "").strip()
    require_citation_for = [t.strip() for t in raw_require_citation_for.split(",") if t.strip()]

    # Unset (the default) keeps distill.py's own small/cheap DISTILL_MODEL —
    # right for most callers. A caller whose whole task is precise
    # multi-document extraction (--reduce over many files, say, where a
    # misread here poisons every downstream turn) can point this at
    # something stronger; see distill.py's module docstring for why this
    # matters more than it looks like a "just a helper model" setting would
    # suggest. Always a bare Groq model id, never a provider:model_id
    # spec — make_distill_agent is Groq-only by design regardless of what
    # `provider` above is (same constraint TCODE_VERIFY_MODEL already has).
    distill_model = os.environ.get("TCODE_DISTILL_MODEL", "").strip() or None
    if distill_model:
        _reject_banned_model(distill_model, "TCODE_DISTILL_MODEL")

    # Unset (the default) means SubAgents' model menu stays empty, so
    # delegate_task's schema never offers a `model` argument at all and
    # behavior is unchanged — same conditional-capability pattern
    # web_capabilities/Shell already use. Set to reach for a genuinely
    # stronger model (or the same model with more reasoning effort) for one
    # hard delegated sub-problem, provider:model_id format matching
    # parse_model_spec above. See agent.py's SubAgents(models=...) wiring.
    raw_expert_model = os.environ.get("TCODE_EXPERT_MODEL", "").strip()
    expert_model: str | None = None
    expert_provider: str | None = None
    expert_provider_api_key: str | None = None
    if raw_expert_model:
        expert_provider, expert_model = parse_model_spec(raw_expert_model, source="TCODE_EXPERT_MODEL")
        if expert_provider == "groq":
            expert_provider_api_key = api_key
        else:
            key_env = _PROVIDER_API_KEY_ENV[expert_provider]
            expert_provider_api_key = os.environ.get(key_env, "").strip()
            if not expert_provider_api_key:
                raise ConfigError(
                    f"TCODE_EXPERT_MODEL={raw_expert_model!r} needs {key_env}, which is not set.\n\n"
                    f"Add it to a .env file in this directory ({cwd / '.env'}) or to "
                    f"your global config ({GLOBAL_DIR / '.env'}):\n\n"
                    f"  {key_env}=..."
                )

    slug = _slugify(cwd)
    _migrate_legacy_project_dirs(GLOBAL_DIR / "projects")
    project_dir = GLOBAL_DIR / "projects" / slug
    memory_dir = GLOBAL_DIR / "memory"
    # Global, not per-project, like memory_dir above: a skill is a reusable
    # prompt snippet the user wrote once, not conversation state tied to a
    # single project. /skill <name> in the interactive REPL loads one on
    # demand (see skills.py) — a deliberately human-invoked, deterministic
    # alternative to the harness's own model-initiated Skills capability.
    skills_dir = GLOBAL_DIR / "skills"
    sessions_dir = project_dir / "sessions"
    steps_dir = project_dir / "steps"
    # Inside the workspace, not under GLOBAL_DIR like everything else here:
    # FileSystem sandboxes read_file/list_directory/find_files to a single
    # root_dir (cwd) with no multi-root support, so a clone placed outside
    # it is reachable only by raw shell commands, never by those safer,
    # scoped tools. It also can't be a dot-prefixed directory: list_directory
    # /search_files/find_files skip any path with a dot-prefixed component,
    # so e.g. `.tcode-scratch/repo/README.md` would resolve but never show
    # up in a listing, making it undiscoverable in practice.
    scratch_dir = cwd / "tcode-scratch"

    for d in (memory_dir, skills_dir, sessions_dir, steps_dir, scratch_dir):
        d.mkdir(parents=True, exist_ok=True)

    return Config(
        cwd=cwd,
        api_key=api_key,
        model=model,
        provider=provider,
        provider_api_key=provider_api_key,
        request_limit=request_limit,
        allowed_commands=allowed_commands,
        shell=shell,
        readonly=readonly,
        sandboxed=sandboxed,
        effort=effort,
        memory_enabled=memory_enabled,
        tool_output_max_chars=tool_output_max_chars,
        keep_tool_pairs=keep_tool_pairs,
        context_compact_fraction=context_compact_fraction,
        context_warn_fraction=context_warn_fraction,
        context_keep_messages=context_keep_messages,
        context_max_part_tokens=context_max_part_tokens,
        context_window_override=context_window_override,
        max_rpm=max_rpm,
        groq_max_rpm=groq_max_rpm,
        web_search=web_search,
        tavily_api_key=tavily_api_key,
        write_scope=write_scope,
        check_citations=check_citations,
        require_citation_for=require_citation_for,
        distill_model=distill_model,
        expert_model=expert_model,
        expert_provider=expert_provider,
        expert_provider_api_key=expert_provider_api_key,
        rpm_state_file=rpm_state_file,
        groq_rpm_state_file=groq_rpm_state_file,
        project_slug=slug,
        project_dir=project_dir,
        memory_dir=memory_dir,
        skills_dir=skills_dir,
        sessions_dir=sessions_dir,
        steps_dir=steps_dir,
        scratch_dir=scratch_dir,
        # Per-project like everything except memory (see module docstring):
        # a global history file would mix REPL input across every project
        # the tool has ever been run in.
        history_file=project_dir / "input_history",
        latest_session_file=sessions_dir / "latest.json",
    )
