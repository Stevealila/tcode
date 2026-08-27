"""Tests for the primary-turn effort dial (--effort / --think / TCODE_EFFORT).
See betterment/plan.txt O5 / 3.4.c.
"""

from __future__ import annotations

import pytest

from src.agent import _primary_model_settings
from src.config import ConfigError


def _cfg(tcode_cfg, **over):
    for k, v in over.items():
        object.__setattr__(tcode_cfg, k, v)
    return tcode_cfg


def test_no_effort_is_a_no_op(tcode_cfg):
    assert tcode_cfg.effort is None
    assert _primary_model_settings(tcode_cfg) is None


def test_groq_effort_becomes_reasoning_effort_setting(tcode_cfg):
    cfg = _cfg(tcode_cfg, effort="high", provider="groq")
    assert _primary_model_settings(cfg) == {"groq_reasoning_effort": "high"}


def test_non_groq_provider_is_a_documented_no_op(tcode_cfg):
    cfg = _cfg(tcode_cfg, effort="high", provider="google")
    assert _primary_model_settings(cfg) is None


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_effort_override_accepts_each_level(tcode_cfg, monkeypatch, tmp_path, level):
    from src import config as config_mod

    monkeypatch.setattr(config_mod, "GLOBAL_DIR", tmp_path / ".tcode")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    ws = tmp_path / "ws2"
    ws.mkdir()
    cfg = config_mod.load_config(ws, effort_override=level)
    assert cfg.effort == level


def test_invalid_effort_fails_fast(tcode_cfg, monkeypatch, tmp_path):
    from src import config as config_mod

    monkeypatch.setattr(config_mod, "GLOBAL_DIR", tmp_path / ".tcode")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    ws = tmp_path / "ws3"
    ws.mkdir()
    with pytest.raises(ConfigError, match="invalid effort"):
        config_mod.load_config(ws, effort_override="ultra")


def test_flag_beats_env(tcode_cfg, monkeypatch, tmp_path):
    from src import config as config_mod

    monkeypatch.setattr(config_mod, "GLOBAL_DIR", tmp_path / ".tcode")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("TCODE_EFFORT", "low")
    ws = tmp_path / "ws4"
    ws.mkdir()
    assert config_mod.load_config(ws, effort_override="high").effort == "high"
    assert config_mod.load_config(ws).effort == "low"
