"""The /clear and /compact REPL handlers (src.cli._clear_command / _compact_command)."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from src import cli
from src import context as ctx


@pytest.fixture
def ui_spy(monkeypatch):
    msgs: list[str] = []
    monkeypatch.setattr(cli.ui, "print_notice", lambda m, **k: msgs.append(m))
    monkeypatch.setattr(cli.ui, "print_error", lambda m, **k: msgs.append(m))
    return msgs


def _history(n=4):
    h = []
    for i in range(n):
        h.append(ModelRequest(parts=[UserPromptPart(content=f"q{i} " * 50)]))
        h.append(ModelResponse(parts=[TextPart(content=f"a{i} " * 50)]))
    return h


# --- /clear ---------------------------------------------------------


def test_clear_wipes_history_session_and_gauge(tcode_cfg, ui_spy, monkeypatch):
    tcode_cfg.latest_session_file.write_bytes(b"[]")
    cleared = {"n": 0}
    monkeypatch.setattr(cli, "clear_session", lambda cfg: cleared.__setitem__("n", cleared["n"] + 1))
    ctx.GAUGE.update(ctx.ContextUsage(used_tokens=5, window_tokens=100, resolved=True))

    out = cli._clear_command(tcode_cfg, _history(3))

    assert out == []
    assert cleared["n"] == 1
    assert ctx.GAUGE.latest is None
    assert any("cleared 6 message(s)" in m for m in ui_spy)


def test_clear_on_empty_history(tcode_cfg, ui_spy, monkeypatch):
    monkeypatch.setattr(cli, "clear_session", lambda cfg: None)
    out = cli._clear_command(tcode_cfg, [])
    assert out == []
    assert any("already empty" in m for m in ui_spy)


# --- /compact ------------------------------------------------------


def test_compact_empty_history_is_a_noop(tcode_cfg, ui_spy):
    out = asyncio.run(cli._compact_command(tcode_cfg, [], None))
    assert out == []
    assert any("nothing to compact" in m for m in ui_spy)


def test_compact_changed_saves_and_reports(tcode_cfg, ui_spy, monkeypatch):
    hist = _history(4)
    new = _history(1)
    saved = {"n": 0}

    async def fake_compact(cfg, messages, *, focus=None):
        assert focus == "the parser work"
        return new, ctx.CompactionResult(
            changed=True, messages_before=8, messages_after=2, tokens_before=9000, tokens_after=1200
        )

    monkeypatch.setattr(ctx, "compact_history", fake_compact)
    monkeypatch.setattr(cli, "save_session", lambda cfg, m: saved.__setitem__("n", saved["n"] + 1))

    out = asyncio.run(cli._compact_command(tcode_cfg, hist, "the parser work"))

    assert out is new
    assert saved["n"] == 1
    assert any("8 → 2 messages" in m for m in ui_spy)


def test_compact_unchanged_does_not_save(tcode_cfg, ui_spy, monkeypatch):
    hist = _history(1)

    async def fake_compact(cfg, messages, *, focus=None):
        return messages, ctx.CompactionResult(
            changed=False, messages_before=2, messages_after=2, tokens_before=500, tokens_after=500
        )

    monkeypatch.setattr(ctx, "compact_history", fake_compact)
    monkeypatch.setattr(cli, "save_session", lambda cfg, m: pytest.fail("must not save an unchanged history"))

    out = asyncio.run(cli._compact_command(tcode_cfg, hist, None))
    assert out is hist
    assert any("already compact" in m for m in ui_spy)


def test_compact_failure_keeps_history(tcode_cfg, ui_spy, monkeypatch):
    hist = _history(3)

    async def boom(cfg, messages, *, focus=None):
        raise RuntimeError("summary model 500")

    monkeypatch.setattr(ctx, "compact_history", boom)
    out = asyncio.run(cli._compact_command(tcode_cfg, hist, None))
    assert out is hist
    assert any("compaction failed" in m for m in ui_spy)
