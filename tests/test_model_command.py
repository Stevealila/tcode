"""The REPL `/model` command — src.cli._model_command.

Robustness contract: whatever the arg, a failed or rejected switch must
leave `cfg` and the agent exactly as they were, and never raise.
"""

from __future__ import annotations

import pytest

from src import cli


class _Agent:
    def __init__(self, tag):
        self.tag = tag


@pytest.fixture
def ui_spy(monkeypatch):
    """Collect every ui notice/error string the command emits."""
    msgs: list[str] = []
    monkeypatch.setattr(cli.ui, "print_notice", lambda m, **k: msgs.append(m))
    monkeypatch.setattr(cli.ui, "print_error", lambda m, **k: msgs.append(m))
    monkeypatch.setattr(cli.ui, "show_models", lambda cfg, cat, err=None: msgs.append(f"LIST:{cat}:{err}"))
    return msgs


@pytest.fixture
def patched(monkeypatch):
    """Stub build_agent and the catalogue; primary stays on Groq gpt-oss-120b."""
    monkeypatch.setattr(cli, "build_agent", lambda cfg: _Agent(cfg.model))
    import src.models as models_mod

    catalogue = [
        "groq/compound",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
    ]
    monkeypatch.setattr(models_mod, "list_groq_models", lambda *a, **k: (catalogue, None))
    return catalogue


def test_no_arg_lists_without_changing_anything(patched, ui_spy, tcode_cfg):
    before = (tcode_cfg.model, tcode_cfg.provider)
    agent = _Agent("orig")
    out = cli._model_command(tcode_cfg, agent, "")
    assert out is agent
    assert (tcode_cfg.model, tcode_cfg.provider) == before
    assert any(m.startswith("LIST:") for m in ui_spy)


def test_valid_switch_rebuilds_agent_and_updates_cfg(patched, ui_spy, tcode_cfg):
    agent = _Agent("orig")
    # "20b" is ambiguous (gpt-oss-20b + safeguard-20b) -> no change
    assert cli._model_command(tcode_cfg, agent, "20b") is agent
    assert tcode_cfg.model == "openai/gpt-oss-120b"
    # a precise arg switches
    out = cli._model_command(tcode_cfg, agent, "safeguard")
    assert out is not agent
    assert tcode_cfg.model == "openai/gpt-oss-safeguard-20b"
    assert tcode_cfg.provider == "groq"
    assert any("model" in m and "safeguard" in m for m in ui_spy)


def test_banned_model_is_rejected(patched, ui_spy, tcode_cfg):
    agent = _Agent("orig")
    out = cli._model_command(tcode_cfg, agent, "qwen/qwen3.8-27b")
    assert out is agent
    assert tcode_cfg.model == "openai/gpt-oss-120b"
    assert any("block list" in m for m in ui_spy)


def test_no_match_leaves_session_alone(patched, ui_spy, tcode_cfg):
    agent = _Agent("orig")
    out = cli._model_command(tcode_cfg, agent, "made-up-model")
    assert out is agent
    assert tcode_cfg.model == "openai/gpt-oss-120b"
    assert any("no model matching" in m for m in ui_spy)


def test_ambiguous_shows_candidates(patched, ui_spy, tcode_cfg):
    agent = _Agent("orig")
    out = cli._model_command(tcode_cfg, agent, "oss")
    assert out is agent
    assert any("ambiguous" in m for m in ui_spy)


def test_already_on_model_is_a_noop(patched, ui_spy, tcode_cfg):
    agent = _Agent("orig")
    out = cli._model_command(tcode_cfg, agent, "openai/gpt-oss-120b")
    assert out is agent
    assert any("already using" in m for m in ui_spy)


def test_build_agent_failure_rolls_back(patched, ui_spy, tcode_cfg, monkeypatch):
    def boom(cfg):
        raise RuntimeError("provider extra not installed")

    monkeypatch.setattr(cli, "build_agent", boom)
    agent = _Agent("orig")
    out = cli._model_command(tcode_cfg, agent, "openai/gpt-oss-20b")
    assert out is agent
    assert tcode_cfg.model == "openai/gpt-oss-120b"
    assert tcode_cfg.provider == "groq"
    assert any("couldn't switch" in m for m in ui_spy)


def test_non_groq_without_key_is_rejected(patched, ui_spy, tcode_cfg, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    agent = _Agent("orig")
    out = cli._model_command(tcode_cfg, agent, "google:gemini-3-pro")
    assert out is agent
    assert tcode_cfg.model == "openai/gpt-oss-120b"
    assert any("GOOGLE_API_KEY" in m for m in ui_spy)


def test_offline_catalogue_still_allows_switch(ui_spy, tcode_cfg, monkeypatch):
    monkeypatch.setattr(cli, "build_agent", lambda cfg: _Agent(cfg.model))
    import src.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "list_groq_models",
        lambda *a, **k: (list(models_mod.GROQ_STATIC_MODELS), "ConnectError: offline"),
    )
    agent = _Agent("orig")
    out = cli._model_command(tcode_cfg, agent, "safeguard")
    assert out is not agent
    assert tcode_cfg.model == "openai/gpt-oss-safeguard-20b"


def test_non_groq_switch_warns_effort_is_ignored(patched, ui_spy, tcode_cfg, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    monkeypatch.setattr(cli, "build_agent", lambda cfg: _Agent(cfg.model))
    object.__setattr__(tcode_cfg, "effort", "high")
    out = cli._model_command(tcode_cfg, _Agent("orig"), "google:gemini-3-pro")
    assert tcode_cfg.provider == "google"
    assert tcode_cfg.model == "gemini-3-pro"
    assert out.tag == "gemini-3-pro"
    assert any("effort" in m and "no effect" in m for m in ui_spy)
