"""The permanent model block list and the default verifier-model pick."""

from __future__ import annotations

import pytest

from src.config import ConfigError, _reject_banned_model, parse_model_spec
from src.verify import _default_verify_model


@pytest.mark.parametrize(
    "model_id",
    [
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b",
        "Qwen/Qwen3.6-27b",
        "qwq-32b",
        "tongyi-something",
    ],
)
def test_reject_banned_model_raises(model_id):
    with pytest.raises(ConfigError, match="permanent block list"):
        _reject_banned_model(model_id, "--model")


@pytest.mark.parametrize("model_id", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound"])
def test_reject_banned_model_allows_others(model_id):
    _reject_banned_model(model_id, "--model")  # must not raise


def test_parse_model_spec_blocks_banned_bare_id():
    with pytest.raises(ConfigError, match="block list"):
        parse_model_spec("qwen/qwen3.6-27b")


def test_parse_model_spec_blocks_banned_with_provider_prefix():
    with pytest.raises(ConfigError, match="block list"):
        parse_model_spec("groq:qwen/qwen3.6-27b")


def test_default_verify_model_is_never_banned():
    for primary in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound"):
        picked = _default_verify_model(primary)
        _reject_banned_model(picked, "verifier")  # must not raise
        assert picked != primary


def _load(tmp_path, monkeypatch, **env):
    from src import config as config_mod

    monkeypatch.setattr(config_mod, "GLOBAL_DIR", tmp_path / ".tcode")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    for k in ("GROQ_MODEL", "TCODE_DISTILL_MODEL", "TCODE_EXPERT_MODEL", "TCODE_MAX_RPM"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ws = tmp_path / "ws"
    ws.mkdir()
    return config_mod.load_config(ws)


def test_load_config_blocks_banned_primary(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match="block list"):
        _load(tmp_path, monkeypatch, GROQ_MODEL="qwen/qwen3.6-27b")


def test_load_config_blocks_banned_distill_model(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match="block list"):
        _load(tmp_path, monkeypatch, TCODE_DISTILL_MODEL="qwen/qwen3.8-27b")


def test_load_config_blocks_banned_expert_model(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match="block list"):
        _load(tmp_path, monkeypatch, TCODE_EXPERT_MODEL="groq:qwen/qwen3.6-27b")


def test_load_config_clean_by_default(tmp_path, monkeypatch):
    cfg = _load(tmp_path, monkeypatch)
    assert "qwen" not in cfg.model
