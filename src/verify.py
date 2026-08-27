"""Independent-verifier decision mode (`tcode --verify`).

A single model's tool-call/read discipline can be made reliable (see
files.py/web.py), but a confidently wrong, cleanly-formed answer is a
different problem nothing about correct tool use prevents — the failure
mode that matters most for a caller acting on the answer without a human
reading it first (a script parsing `--quiet` output for a decision, say).
Self-consistency (majority-voting the same model) raises accuracy on random
errors but not systematic ones, since a model's own blind spot tends to
reproduce across samples, and self-reported confidence scores are poorly
calibrated — so instead of either, `--verify` re-derives the answer
independently and checks agreement, closer to how multi-agent verification
outperforms same-model voting in practice.

Three passes, never two:
  1. primary   — the configured model, full capabilities, the real answer.
  2. verifier  — an independent re-derivation from the same prompt: a
     different model by default (never given the primary's own answer,
     which would just invite anchoring), or `TCODE_VERIFY_CMD` for an
     external/cross-provider command when one's configured. Neither is a
     hard dependency on the other — the default path needs nothing beyond
     the same Groq account already in use; TCODE_VERIFY_CMD is there for a
     caller wanting a stronger verifier when one's reachable, same
     opt-in-when-available shape as TAVILY_API_KEY in web.py.
  3. compare   — a third, cheap pass judging substantive agreement, not
     wording.

On agreement, the primary's answer is what the caller gets. On
disagreement, `run_verified` returns nothing usable rather than a hedge or
a pick-one — see cli.py's `verify_mode` for the output contract.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shlex
import subprocess

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models.groq import GroqModel, GroqModelSettings

from .agent import build_agent
from .config import Config, _reject_banned_model
from .ratelimit import throttle

# Cheap, fast, low-reasoning — this is a substance-match judgment over two
# short texts, not a task that benefits from a bigger model or deliberation.
# Same choice and same reasoning-effort fix as distill.py's DISTILL_MODEL.
_COMPARE_MODEL = "openai/gpt-oss-20b"

_COMPARE_INSTRUCTIONS = """\
You are given two independently-produced answers to the same question,
from two different attempts that could not see each other's work. Judge
only whether they agree on the actual substance of the answer — the core
decision, conclusion, or recommendation — not whether the wording,
reasoning detail, or secondary observations match.

Two answers that reach the same bottom line by different reasoning still
AGREE. Two answers that reach a different bottom line — even if similarly
worded, even if both hedge or express low confidence — DISAGREE. If either
answer is empty, an error, or doesn't actually answer the question, that
counts as DISAGREE: no verification happened.

Reply with exactly one word, AGREE or DISAGREE, on the first line. On the
second line, one sentence saying what the core answer was (or the
mismatch, if they disagree). Nothing else.
"""


def _default_verify_model(primary_model: str) -> str:
    """A second, independent model for the verifier pass.

    Constrained to Groq's small catalog and to models tcode will actually
    run: qwen — the previous default — is on config._BANNED_MODEL_SUBSTRINGS
    after proving unreliable at structured output in real use (raw tool-call
    envelopes emitted as prose, documented reasoning-effort values rejected,
    fabricated figures in otherwise-clean output). That leaves the gpt-oss
    pair. gpt-oss-20b is a genuinely separate checkpoint from the 120b
    primary — smaller, independently trained — and reliable at the short
    structured answers this pass compares; when the primary already is 20b,
    step up to 120b so the two passes are never the identical model.
    TCODE_VERIFY_MODEL overrides this; TCODE_VERIFY_CMD swaps in a
    cross-provider verifier entirely.
    """
    return "openai/gpt-oss-120b" if primary_model == "openai/gpt-oss-20b" else "openai/gpt-oss-20b"


async def _run_external_verifier(cmd: str, prompt: str, timeout: int) -> str:
    """Run an arbitrary external command as the verifier, prompt via stdin.

    Stdin (not argv), since most external LLM CLIs accept a prompt that way
    and it sidesteps shell-quoting a long prompt into the command string.
    Set `TCODE_VERIFY_CMD` to any read-only external LLM CLI invocation and
    tcode feeds it the prompt on stdin.
    """
    def _run() -> str:
        try:
            proc = subprocess.run(
                shlex.split(cmd),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return (proc.stdout or "").strip()

    return await asyncio.to_thread(_run)


async def get_verifier_answer(cfg: Config, prompt: str, usage_limits: UsageLimits, timeout: int) -> str:
    """The verifier's independently-derived answer — external command if
    configured, otherwise a second tcode agent on a different Groq model.
    Either way, this never sees the primary's answer.
    """
    cmd = os.environ.get("TCODE_VERIFY_CMD", "").strip()
    if cmd:
        return await _run_external_verifier(cmd, prompt, timeout)

    model = os.environ.get("TCODE_VERIFY_MODEL", "").strip() or _default_verify_model(cfg.model)
    _reject_banned_model(model, "TCODE_VERIFY_MODEL")
    # provider/provider_api_key pinned back to Groq explicitly, not just
    # left as `cfg`'s — the default verifier is always a second Groq model
    # (see this function's docstring) even when the primary model has moved
    # to a different provider via --model provider:model_id, and a plain
    # `dataclasses.replace(cfg, model=model)` would otherwise carry the
    # primary's own (possibly non-Groq) provider/key over unchanged, pairing
    # a Groq model id with the wrong backend entirely.
    verify_cfg = dataclasses.replace(cfg, model=model, provider="groq", provider_api_key=cfg.api_key)
    agent = build_agent(verify_cfg)
    # Same shared cross-process throttle the main turn loop uses when the
    # primary is also on Groq (see cli.py/runner.py) — this is always a
    # second Groq request the main loop doesn't know about regardless of
    # what the primary model's own provider is, and Groq enforces RPM per
    # account, not per call site or process.
    await throttle(cfg.groq_rpm_state_file, cfg.groq_max_rpm)
    result = await agent.run(prompt, usage_limits=usage_limits)
    return str(result.output)


def _compare_agent(cfg: Config) -> Agent:
    from pydantic_ai.providers.groq import GroqProvider

    provider = GroqProvider(api_key=cfg.api_key)
    model = GroqModel(_COMPARE_MODEL, provider=provider)
    settings = GroqModelSettings(groq_reasoning_effort="low")
    return Agent(model, instructions=_COMPARE_INSTRUCTIONS, model_settings=settings)


async def compare(cfg: Config, primary_text: str, verifier_text: str) -> tuple[bool, str]:
    """Returns (agreed, verdict_text)."""
    if not primary_text.strip() or not verifier_text.strip():
        return False, "DISAGREE\nOne side produced no answer to compare."
    agent = _compare_agent(cfg)
    # _compare_agent is always Groq (hardcoded GroqProvider/_COMPARE_MODEL
    # above) regardless of the primary model's own provider — pace against
    # Groq's own budget specifically, same reasoning as get_verifier_answer.
    await throttle(cfg.groq_rpm_state_file, cfg.groq_max_rpm)
    result = await agent.run(f"ANSWER A (primary):\n{primary_text}\n\nANSWER B (verifier):\n{verifier_text}")
    text = str(result.output).strip()
    agreed = text.upper().startswith("AGREE")
    return agreed, text
