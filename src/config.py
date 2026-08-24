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
DEFAULT_TOOL_OUTPUT_MAX_CHARS = 2000
DEFAULT_KEEP_TOOL_PAIRS = 3
DEFAULT_CLEAR_AFTER_TOKENS = 3000
DEFAULT_MAX_RPM = 30
DEFAULT_WEB_SEARCH = True

GLOBAL_DIR = Path.home() / ".tcode"


def _slugify(path: Path) -> str:
    """Turn an absolute path into a filesystem-safe slug, e.g.

    /home/alice/some-project -> -home-alice-some-project
    """
    resolved = str(path.resolve())
    slug = resolved.replace(os.sep, "-")
    return slug or "-root"


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
    request_limit: int
    allowed_commands: list[str]
    tool_output_max_chars: int
    keep_tool_pairs: int
    clear_after_tokens: int
    max_rpm: int
    web_search: bool
    tavily_api_key: str | None
    rpm_state_file: Path
    project_slug: str
    project_dir: Path
    memory_dir: Path
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


def load_config(cwd: Path | None = None, model_override: str | None = None) -> Config:
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

    model = model_override or os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_MODEL
    request_limit = int(os.environ.get("GROQ_REQUEST_LIMIT", DEFAULT_REQUEST_LIMIT))

    # An empty list here means "denylist mode": Shell blocks a short list of
    # destructive commands (rm, dd, mkfs, shutdown, ...) and allows
    # everything else. Set TCODE_ALLOWED_COMMANDS to flip to strict
    # allowlist mode instead, e.g. "git,python,pytest,ruff".
    raw_allowed = os.environ.get("TCODE_ALLOWED_COMMANDS", "").strip()
    allowed_commands = [c.strip() for c in raw_allowed.split(",") if c.strip()]

    # Groq's lower tiers cap throughput at a few thousand tokens per minute,
    # so a single unbounded tool result (e.g. `ls -R` on a real repo) can
    # blow the whole request budget on its own, and a growing conversation
    # compounds it turn over turn. These two keep both in check: cap any one
    # tool's output as it enters history, then prune old tool results once
    # the conversation has grown past a few turns.
    tool_output_max_chars = int(
        os.environ.get("TCODE_TOOL_OUTPUT_LIMIT", DEFAULT_TOOL_OUTPUT_MAX_CHARS)
    )
    keep_tool_pairs = int(os.environ.get("TCODE_KEEP_TOOL_PAIRS", DEFAULT_KEEP_TOOL_PAIRS))
    clear_after_tokens = int(
        os.environ.get("TCODE_CLEAR_AFTER_TOKENS", DEFAULT_CLEAR_AFTER_TOKENS)
    )

    # Requests-per-minute is a separate, harder limit from tokens-per-minute,
    # and Groq enforces it per account, not per process. The throttle state
    # lives in one shared global file, not per-project, so it applies across
    # every tcode invocation hitting this account. 0 disables it.
    max_rpm = int(os.environ.get("TCODE_MAX_RPM", DEFAULT_MAX_RPM))

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

    slug = _slugify(cwd)
    project_dir = GLOBAL_DIR / "projects" / slug
    memory_dir = GLOBAL_DIR / "memory"
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

    for d in (memory_dir, sessions_dir, steps_dir, scratch_dir):
        d.mkdir(parents=True, exist_ok=True)

    return Config(
        cwd=cwd,
        api_key=api_key,
        model=model,
        request_limit=request_limit,
        allowed_commands=allowed_commands,
        tool_output_max_chars=tool_output_max_chars,
        keep_tool_pairs=keep_tool_pairs,
        clear_after_tokens=clear_after_tokens,
        max_rpm=max_rpm,
        web_search=web_search,
        tavily_api_key=tavily_api_key,
        rpm_state_file=GLOBAL_DIR / "rpm_state",
        project_slug=slug,
        project_dir=project_dir,
        memory_dir=memory_dir,
        sessions_dir=sessions_dir,
        steps_dir=steps_dir,
        scratch_dir=scratch_dir,
        # Per-project like everything except memory (see module docstring):
        # a global history file would mix REPL input across every project
        # the tool has ever been run in.
        history_file=project_dir / "input_history",
        latest_session_file=sessions_dir / "latest.json",
    )
