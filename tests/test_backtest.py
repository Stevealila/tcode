"""Tests for cli.backtest_mode's own control flow — the archive skip
paths and the synthesized "replay crashed" row — with build_agent /
run_turn stubbed so no model is called. See betterment/plan.txt O4.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

from src import cli


def _write_archive(sessions_dir, stem: str, messages) -> None:
    (sessions_dir / f"{stem}.json").write_bytes(ModelMessagesTypeAdapter.dump_json(messages))


@pytest.fixture
def patched_backtest(monkeypatch):
    """Stub the model-touching parts of backtest_mode and capture output."""
    rendered: list = []
    notices: list[str] = []
    monkeypatch.setattr(cli, "build_agent", lambda cfg: object())
    monkeypatch.setattr(cli.ui, "render_backtest_table", lambda rows: rendered.append(rows))
    monkeypatch.setattr(cli.ui, "print_notice", lambda msg, **kw: notices.append(msg))
    return rendered, notices


def test_skip_paths_and_synthesized_crash_row(tcode_cfg, patched_backtest, monkeypatch):
    rendered, notices = patched_backtest
    sd = tcode_cfg.sessions_dir

    # 1. unparseable filename stem -> skipped_unparseable
    (sd / "not-a-timestamp.json").write_bytes(b"[]")
    # 2. valid stem but only a harness-injected synthetic user turn -> skipped_no_prompt
    _write_archive(sd, "20260827-120000", [
        ModelRequest(parts=[UserPromptPart(content="[WarnNearLimits]\nCRITICAL: limits approaching")]),
    ])
    # 3. valid stem + a real prompt -> replayed, and we make the replay crash
    _write_archive(sd, "20260827-130000", [
        ModelRequest(parts=[UserPromptPart(content="the real task")]),
    ])

    async def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "run_turn", boom)

    asyncio.run(cli.backtest_mode(tcode_cfg, 10))

    assert len(rendered) == 1
    rows = rendered[0]
    assert len(rows) == 1
    prompt, old, new = rows[0]
    assert prompt == "the real task"
    assert old is None
    assert new["outcome"] == "replay_crashed"
    assert "RuntimeError: kaboom" in new["error"]

    joined = " ".join(notices)
    assert "1 unparseable name" in joined
    assert "1 no prompt" in joined
    assert "failed to replay" in joined


def test_reports_when_no_archive_has_a_usable_prompt(tcode_cfg, patched_backtest, monkeypatch):
    rendered, notices = patched_backtest
    sd = tcode_cfg.sessions_dir
    (sd / "not-a-timestamp.json").write_bytes(b"[]")
    _write_archive(sd, "20260827-120000", [
        ModelRequest(parts=[UserPromptPart(content="[LimitWarner] wrap up now")]),
    ])

    ran = []
    monkeypatch.setattr(cli, "run_turn", lambda *a, **k: ran.append(1))

    asyncio.run(cli.backtest_mode(tcode_cfg, 10))

    assert ran == []            # never got as far as a replay
    assert rendered == []       # returned before rendering an empty table
    assert any("0 of 2 archive(s) had a usable prompt" in n for n in notices)


def test_matches_before_record_for_a_clean_replay(tcode_cfg, patched_backtest, monkeypatch):
    """The happy path the O1 fix restores: a replayed prompt's row carries
    its original ("before") smell record, matched by timestamp."""
    rendered, _ = patched_backtest
    sd = tcode_cfg.sessions_dir
    _write_archive(sd, "20260827-140506", [
        ModelRequest(parts=[UserPromptPart(content="summarize the module")]),
    ])
    # A realistic smell line: written a fraction of a second before the
    # archive's whole-second stem (see _closest_smell_record / O1).
    (sd / "smell.jsonl").write_text(
        '{"timestamp": "2026-08-27T14:05:06.421000", "tool_counts": {"read_file": 2}, '
        '"retry_count": 0, "elapsed_s": 9.0, "total_tokens": 1200, "outcome": "ok"}\n'
    )

    async def noop(*a, **k):
        return ([], "")

    monkeypatch.setattr(cli, "run_turn", noop)

    asyncio.run(cli.backtest_mode(tcode_cfg, 10))

    rows = rendered[0]
    assert len(rows) == 1
    _, old, _new = rows[0]
    assert old is not None
    assert old["total_tokens"] == 1200
