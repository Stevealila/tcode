"""Shared fixtures."""

from __future__ import annotations

import pytest

from src import config as config_mod


@pytest.fixture
def tcode_cfg(tmp_path, monkeypatch):
    """A real `Config` built over a throwaway ~/.tcode, no network.

    `load_config` reads module-global `GLOBAL_DIR` at call time, so
    monkeypatching it here redirects every derived path (sessions, memory,
    rpm state) into `tmp_path`. GroqProvider construction doesn't touch the
    network, so a placeholder key is fine.
    """
    monkeypatch.setattr(config_mod, "GLOBAL_DIR", tmp_path / ".tcode")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("TCODE_MAX_RPM", raising=False)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return config_mod.load_config(workspace)
