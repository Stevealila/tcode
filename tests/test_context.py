"""The context-management pipeline (src/context.py) and its config knobs."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from src import context as ctx
from src.config import ConfigError

# --- _read_file_key ----------------------------------------------------


def _call(tool_name, **args):
    return ToolCallPart(tool_name=tool_name, args=args)


def test_read_file_key_full_read():
    assert ctx._read_file_key(_call("read_file", path="a.py")) == "a.py@0:None"


def test_read_file_key_partial_reads_differ():
    k1 = ctx._read_file_key(_call("read_file", path="a.py", offset=0, limit=100))
    k2 = ctx._read_file_key(_call("read_file", path="a.py", offset=100, limit=100))
    assert k1 != k2  # a later slice must not dedupe an earlier one


def test_read_file_key_ignores_non_reads():
    assert ctx._read_file_key(_call("write_file", path="a.py", content="x")) is None
    assert ctx._read_file_key(_call("run_command", command="ls")) is None


def test_read_file_key_missing_path():
    assert ctx._read_file_key(_call("read_file")) is None


# --- window kwargs + capabilities ------------------------------------


def test_window_kwargs_override(tcode_cfg):
    cfg = dataclasses.replace(tcode_cfg, context_window_override=64_000)
    assert ctx._window_kwargs(cfg) == {"context_window": 64_000}


def test_window_kwargs_fallback(tcode_cfg):
    cfg = dataclasses.replace(tcode_cfg, context_window_override=None)
    assert "fallback_context_window" in ctx._window_kwargs(cfg)
    assert "context_window" not in ctx._window_kwargs(cfg)


def test_compaction_capabilities_shape(tcode_cfg):
    caps = ctx.compaction_capabilities(tcode_cfg)
    assert [type(c).__name__ for c in caps] == [
        "ClampOversizedMessages",
        "DeduplicateFileReads",
        "TieredCompaction",
        "ReportContextUsage",
        "WarnNearLimits",
    ]


def test_compaction_capabilities_builds_for_non_groq(tcode_cfg):
    cfg = dataclasses.replace(tcode_cfg, provider="google", model="gemini-3-pro")
    caps = ctx.compaction_capabilities(cfg)  # must not raise
    assert len(caps) == 5


def test_tiered_uses_the_configured_fraction(tcode_cfg):
    cfg = dataclasses.replace(tcode_cfg, context_compact_fraction=0.5)
    tiered = ctx.compaction_capabilities(cfg)[2]
    assert tiered.target_fraction == 0.5


# --- gauge + describe_context ----------------------------------------


def test_gauge_update_and_reset():
    g = ctx.ContextGauge()
    assert g.latest is None
    g.update(ctx.ContextUsage(used_tokens=100, window_tokens=1000, resolved=True))
    assert g.latest.fraction == pytest.approx(0.1)
    g.reset()
    assert g.latest is None


def test_describe_context_empty_history(tcode_cfg, monkeypatch):
    monkeypatch.setattr(ctx.GAUGE, "latest", None)
    r = ctx.describe_context(tcode_cfg, [])
    assert r.message_count == 0
    assert r.tool_result_count == 0
    assert r.live is False
    assert r.window_tokens > 0


def test_describe_context_prefers_live_gauge(tcode_cfg, monkeypatch):
    monkeypatch.setattr(
        ctx.GAUGE,
        "latest",
        ctx.ContextUsage(used_tokens=42_000, window_tokens=131_072, resolved=True),
    )
    r = ctx.describe_context(tcode_cfg, [ModelRequest(parts=[UserPromptPart(content="hi")])])
    assert r.live is True
    assert r.used_tokens == 42_000
    assert r.fraction == pytest.approx(42_000 / 131_072)


def test_describe_context_counts_tool_results(tcode_cfg, monkeypatch):
    from pydantic_ai.messages import ToolReturnPart

    monkeypatch.setattr(ctx.GAUGE, "latest", None)
    history = [
        ModelRequest(parts=[UserPromptPart(content="do it")]),
        ModelResponse(parts=[TextPart(content="ok"), ToolCallPart(tool_name="read_file", args={}, tool_call_id="1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="read_file", content="data", tool_call_id="1")]),
    ]
    r = ctx.describe_context(tcode_cfg, history)
    assert r.message_count == 3
    assert r.tool_result_count == 1


# --- config validation ----------------------------------------------


def _load(tmp_path, monkeypatch, **env):
    from src import config as config_mod

    monkeypatch.setattr(config_mod, "GLOBAL_DIR", tmp_path / ".tcode")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    return config_mod.load_config(ws)


@pytest.mark.parametrize("value", ["0", "1.5", "-0.2", "abc", "80"])
def test_bad_compact_fraction_rejected(tmp_path, monkeypatch, value):
    with pytest.raises(ConfigError):
        _load(tmp_path, monkeypatch, TCODE_CONTEXT_COMPACT_FRACTION=value)


def test_valid_fractions_and_window_override(tmp_path, monkeypatch):
    cfg = _load(
        tmp_path,
        monkeypatch,
        TCODE_CONTEXT_COMPACT_FRACTION="0.6",
        TCODE_CONTEXT_WARN_FRACTION="0.9",
        TCODE_CONTEXT_WINDOW="64000",
    )
    assert cfg.context_compact_fraction == 0.6
    assert cfg.context_warn_fraction == 0.9
    assert cfg.context_window_override == 64000


def test_bad_window_override_rejected(tmp_path, monkeypatch):
    with pytest.raises(ConfigError):
        _load(tmp_path, monkeypatch, TCODE_CONTEXT_WINDOW="huge")


# --- TCODE_MEMORY ---------------------------------------------------


def test_memory_on_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TCODE_MEMORY", raising=False)
    assert _load(tmp_path, monkeypatch).memory_enabled is True


@pytest.mark.parametrize("value,expected", [("0", False), ("false", False), ("no", False), ("1", True)])
def test_memory_env_toggle(tmp_path, monkeypatch, value, expected):
    assert _load(tmp_path, monkeypatch, TCODE_MEMORY=value).memory_enabled is expected


def test_build_agent_builds_with_memory_on_and_off(tmp_path, monkeypatch):
    from src.agent import build_agent

    build_agent(_load(tmp_path / "a", monkeypatch))  # memory on
    monkeypatch.setenv("TCODE_MEMORY", "0")
    build_agent(_load(tmp_path / "b", monkeypatch))  # memory off — must not raise
