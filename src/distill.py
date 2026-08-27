"""Shared "read big content, answer one question about it" helper.

`web.py`'s `web_fetch` and `files.py`'s `read_and_distill` both hand a large,
noisy blob (a fetched page, a big file) to a second, cheap model along with
the caller's specific question, so only the distilled answer reaches the
primary model's context. The distillation agent doesn't care whether the
content came from a URL or a path, so it's built once here.

`openai/gpt-oss-20b` is a reasoning model, so `groq_reasoning_effort` is
forced to `low`: left at its default, it can spend its entire per-request
token budget on hidden chain-of-thought before emitting any visible answer,
which single-document extraction doesn't need — cheaper and faster on every
call, and removes a failure mode this task never needed to risk. (`'none'`
is rejected outright by this model despite the wider type hint; checked
against the live API.)

`TCODE_DISTILL_MODEL` overrides this default (see config.py) for a caller
whose whole task is precise multi-document extraction and citation — a
research pipeline synthesizing many files via `--reduce`, say — where this
narrow-looking extraction pass is not actually low-stakes: everything the
primary model ever sees about a large file, a `--reduce` batch, or a
fetched web page goes through it first, so a misread here poisons every
downstream turn regardless of how good the primary model is. The
`groq_reasoning_effort="low"` forcing above is specific to
`openai/gpt-oss-20b` needing it (see the model's own case above) — a
caller opting into a different, presumably-better distillation model wants
that model's own default reasoning depth, not this one's speed shortcut
force-fed to it (and it isn't guaranteed to accept `'low'` at all — some
Groq models reject it outright with `` `must be one of 'none' or
'default'` ``), so overriding away from `DISTILL_MODEL` also drops the
forced setting.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings
from pydantic_ai.providers.groq import GroqProvider

# Smallest/fastest model in Groq's current catalog: this is a narrow
# extraction task (one document, one question), not agentic reasoning, so it
# doesn't need the main model's capacity — and running it on a separate,
# smaller model keeps this pass cheap relative to the budget the primary
# model's turn is already spending. TCODE_DISTILL_MODEL overrides it — see
# module docstring for why a caller would want to, and config.py for the
# env var itself.
DISTILL_MODEL = "openai/gpt-oss-20b"

DISTILL_INSTRUCTIONS = """\
You read one piece of content (a fetched web page, a local file) and answer
one question about it for another agent, which will act on your answer
without seeing the content itself.

Quote the specific fact requested (a number, a date, a status, a tag like
[REPORTED]/[RUMOUR]/[CONFIRMED] if the content uses that convention) plus
enough context to judge how current or reliable it is. A number given as a
historical or hypothetical example, or one clearly superseded elsewhere in
the same content, is not the answer, even if it's the only one present —
say the content doesn't have a current answer rather than reporting it as
one, and never drop a [RUMOUR]/uncertainty tag when relaying a claim that
carried one.

If the content is boilerplate, empty, an error page, or otherwise doesn't
answer the question, say so plainly in one line. Never state a fact, a
number, a quote, or a tag that isn't actually present in the content below —
the agent reading your answer cannot check your work against the original.
"""


def make_distill_agent(provider: GroqProvider, model_id: str | None = None) -> Agent:
    is_default = model_id is None or model_id == DISTILL_MODEL
    model = GroqModel(model_id or DISTILL_MODEL, provider=provider)
    # The forced-low reasoning effort is a speed shortcut specific to
    # DISTILL_MODEL — see module docstring. An override gets its own default
    # reasoning depth instead, both because that's presumably the point of
    # overriding at all, and because it isn't guaranteed to accept "low"
    # (some Groq models reject that value outright).
    settings = GroqModelSettings(groq_reasoning_effort="low") if is_default else None
    return Agent(model, instructions=DISTILL_INSTRUCTIONS, model_settings=settings)
