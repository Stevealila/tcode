"""Independent-verifier decision mode (`tcode --verify`).

Built for a specific, well-evidenced gap: a single model's tool-call/read
discipline can be made reliable (see files.py/web.py's module docstrings),
but a *confidently wrong, cleanly-formed* answer is a different problem —
nothing about calling the right tools correctly stops a model from
misreading its own inputs. That matters most exactly where a caller acts
on the answer without a human reading it critically first (a script parsing
`--quiet` output for a decision, say) — the "look correct" property write
tools give you says nothing about factual correctness.

Researched before building rather than guessed at: self-consistency
(majority-voting several samples from the *same* model) is documented to
raise accuracy on random errors but *not* on systematic ones — if a model
has a consistent blind spot, it tends to reproduce that same blind spot
across samples, and voting just returns it with false confidence. Eliciting
a confidence/conviction score from the model itself doesn't reliably fix
this either — self-reported LLM confidence is well-documented as poorly
calibrated. What the same research does support: an *independent* verifier
— ideally a different model, so it isn't sharing the same blind spot —
re-deriving an answer from the same inputs and checked for agreement,
which is closer to how multi-agent verification setups outperform
single-pass and same-model self-consistency in practice.

So `--verify` runs three passes, never two:
  1. primary   — the configured model, full capabilities, the real answer.
  2. verifier  — a genuinely independent re-derivation from the same
     prompt: a different model by default (never the primary's own
     answer as context — that would just invite anchoring, not
     independent judgment), or an arbitrary external command
     (TCODE_VERIFY_CMD) for cross-provider verification when one is
     configured. Neither is a hard dependency on the other: the default
     path needs nothing beyond the same Groq account already in use,
     keeping "runs without a subscription" intact for the verifier too;
     TCODE_VERIFY_CMD is there for a caller wanting a stronger (paid,
     different-architecture) verifier when one happens to be reachable,
     the same shape as TAVILY_API_KEY opting web.py into a better search
     backend when available and falling back cleanly when it isn't.
  3. compare   — a third, cheap pass judging whether primary and verifier
     actually agree on the substance, not the wording.

On agreement, the primary's answer is what the caller gets. On
disagreement, `run_verified` deliberately returns nothing usable rather
than a hedge or a pick-one — see cli.py's `verify_mode` for why that
specific shape (empty stdout, non-zero exit) was chosen over inventing a
new output contract.
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
from .config import Config

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
    """A model genuinely different from the primary, not just a name.

    qwen3.6-27b is a different architecture/lineage from the gpt-oss
    family, not just a smaller sibling — picked specifically because a
    same-family model is more likely to share the same blind spot the
    verifier exists to catch. Falls back to gpt-oss-120b in the one case
    where the primary already is qwen.
    """
    return "openai/gpt-oss-120b" if primary_model == "qwen/qwen3.6-27b" else "qwen/qwen3.6-27b"


async def _run_external_verifier(cmd: str, prompt: str, timeout: int) -> str:
    """Run an arbitrary external command as the verifier, prompt via stdin.

    Stdin (not argv) specifically to match how a Claude Code-shaped
    verifier is actually invoked elsewhere in this codebase's own
    ecosystem (brain_call.py's `_run`), so `TCODE_VERIFY_CMD="claude -p
    --model haiku --allowedTools Read Grep Glob Bash --disallowedTools
    Edit Write NotebookEdit"` works without the caller needing to know
    tcode's own argv-based convention.
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
    verify_cfg = dataclasses.replace(cfg, model=model)
    agent = build_agent(verify_cfg)
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
    result = await agent.run(f"ANSWER A (primary):\n{primary_text}\n\nANSWER B (verifier):\n{verifier_text}")
    text = str(result.output).strip()
    agreed = text.upper().startswith("AGREE")
    return agreed, text
