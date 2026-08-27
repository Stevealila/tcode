"""Per-project conversation persistence, independent of the harness's
StepPersistence capability (which durably tracks execution steps within a
single run, not chat history across CLI invocations).

We serialize the full pydantic-ai message list with the library's own
TypeAdapter, so round-tripping is exact regardless of message shape.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from .config import Config

MAX_SESSION_ARCHIVES = 200


def save_session(cfg: Config, messages: list[ModelMessage]) -> None:
    if not messages:
        return
    data = ModelMessagesTypeAdapter.dump_json(messages)
    cfg.latest_session_file.write_bytes(data)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = cfg.sessions_dir / f"{stamp}.json"
    archive.write_bytes(data)
    _prune_archives(cfg.sessions_dir)


def clear_session(cfg: Config) -> None:
    """Drop the 'continue from' pointer so the next `-c` starts fresh.

    Per-turn archives in `sessions_dir` are untouched — `/sessions` still
    lists them, and `save_session` never overwrites `latest.json` with an
    empty history (its `if not messages` guard), so without this an
    in-memory `/clear` would be silently undone by the next `-c`.
    """
    cfg.latest_session_file.unlink(missing_ok=True)


def _prune_archives(sessions_dir: Path) -> None:
    """Keep only the most recent MAX_SESSION_ARCHIVES turn archives.

    save_session runs on every turn, so without a cap this directory grows
    by one file per turn forever.
    """
    archives = sorted(
        (p for p in sessions_dir.glob("*.json") if p.name != "latest.json")
    )
    for stale in archives[:-MAX_SESSION_ARCHIVES]:
        stale.unlink(missing_ok=True)


def load_latest_session(cfg: Config) -> list[ModelMessage]:
    """Load the last saved conversation, or start fresh if it can't be read.

    latest.json is best-effort, not load-bearing (see agent.py's module
    docstring on StepPersistence for the same reasoning): a corrupted write
    (e.g. a Ctrl-C mid-save) or a ModelMessage schema change after a
    pydantic-ai upgrade must not permanently break `-c`/`--continue` by
    raising out of every future invocation.
    """
    if not cfg.latest_session_file.exists():
        return []
    data = cfg.latest_session_file.read_bytes()
    if not data:
        return []
    try:
        return ModelMessagesTypeAdapter.validate_json(data)
    except ValidationError:
        return []


def load_archive(path: Path) -> list[ModelMessage]:
    """Load one archived session file, same best-effort posture as
    load_latest_session (a corrupted or schema-mismatched archive returns
    empty rather than raising)."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if not data:
        return []
    try:
        return ModelMessagesTypeAdapter.validate_json(data)
    except ValidationError:
        return []


def list_sessions(cfg: Config) -> list[Path]:
    archives = sorted(
        (p for p in cfg.sessions_dir.glob("*.json") if p.name != "latest.json"),
        reverse=True,
    )
    return archives
