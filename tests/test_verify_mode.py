"""verify_mode's stdout/exit contract — strict (default) vs advisory."""

from __future__ import annotations

import asyncio

import pytest

from src import cli
from src.cli import verify_mode


@pytest.fixture
def _patched(monkeypatch):
    """Stub the three verify passes; caller sets agreement per test."""
    state = {"agreed": True, "primary": '{"stance":"flat","conviction":0.0}'}

    async def fake_run_turn(agent, prompt, history, limits, cfg, capture=False):
        return [], state["primary"]

    async def fake_get_verifier_answer(cfg, prompt, limits, timeout):
        return "verifier text"

    async def fake_compare(cfg, primary_text, verifier_text):
        return state["agreed"], ("AGREE x" if state["agreed"] else "DISAGREE x")

    monkeypatch.setattr(cli, "build_agent", lambda cfg: object())
    monkeypatch.setattr(cli, "run_turn", fake_run_turn)
    import src.verify as verify_mod

    monkeypatch.setattr(verify_mod, "get_verifier_answer", fake_get_verifier_answer)
    monkeypatch.setattr(verify_mod, "compare", fake_compare)
    return state


def _run(cfg):
    asyncio.run(verify_mode(cfg, "q"))


def test_strict_agreement_prints_primary(_patched, tcode_cfg, capsys, monkeypatch):
    monkeypatch.delenv("TCODE_VERIFY_ADVISORY", raising=False)
    _patched["agreed"] = True
    _run(tcode_cfg)
    assert capsys.readouterr().out.strip() == _patched["primary"]


def test_strict_disagreement_exits_2_with_empty_stdout(_patched, tcode_cfg, capsys, monkeypatch):
    monkeypatch.delenv("TCODE_VERIFY_ADVISORY", raising=False)
    _patched["agreed"] = False
    with pytest.raises(SystemExit) as ei:
        _run(tcode_cfg)
    assert ei.value.code == 2
    assert capsys.readouterr().out == ""


def test_advisory_disagreement_still_prints_primary(_patched, tcode_cfg, capsys, monkeypatch):
    monkeypatch.setenv("TCODE_VERIFY_ADVISORY", "1")
    _patched["agreed"] = False
    _run(tcode_cfg)  # must NOT raise SystemExit
    captured = capsys.readouterr()
    assert captured.out.strip() == _patched["primary"]
    assert "tcode-verify: DISAGREE" in captured.err


def test_advisory_agreement_marks_agree_on_stderr(_patched, tcode_cfg, capsys, monkeypatch):
    monkeypatch.setenv("TCODE_VERIFY_ADVISORY", "1")
    _patched["agreed"] = True
    _run(tcode_cfg)
    captured = capsys.readouterr()
    assert captured.out.strip() == _patched["primary"]
    assert "tcode-verify: AGREE" in captured.err
