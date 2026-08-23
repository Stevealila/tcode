"""Per-project conversation persistence, independent of the harness's
StepPersistence capability (which durably tracks execution steps within a
single run, not chat history across CLI invocations).

We serialize the full pydantic-ai message list with the library's own
TypeAdapter, so round-tripping is exact regardless of message shape.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from .config import Config


def save_session(cfg: Config, messages: list[ModelMessage]) -> None:
    if not messages:
        return
    data = ModelMessagesTypeAdapter.dump_json(messages)
    cfg.latest_session_file.write_bytes(data)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = cfg.sessions_dir / f"{stamp}.json"
    archive.write_bytes(data)


def load_latest_session(cfg: Config) -> list[ModelMessage]:
    if not cfg.latest_session_file.exists():
        return []
    data = cfg.latest_session_file.read_bytes()
    if not data:
        return []
    return ModelMessagesTypeAdapter.validate_json(data)


def list_sessions(cfg: Config) -> list[Path]:
    archives = sorted(
        (p for p in cfg.sessions_dir.glob("*.json") if p.name != "latest.json"),
        reverse=True,
    )
    return archives
