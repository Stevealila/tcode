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
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings
from pydantic_ai.providers.groq import GroqProvider

# Smallest/fastest model in Groq's current catalog: this is a narrow
# extraction task (one document, one question), not agentic reasoning, so it
# doesn't need the main model's capacity — and running it on a separate,
# smaller model keeps this pass cheap relative to the budget the primary
# model's turn is already spending.
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


def make_distill_agent(provider: GroqProvider) -> Agent:
    model = GroqModel(DISTILL_MODEL, provider=provider)
    settings = GroqModelSettings(groq_reasoning_effort="low")
    return Agent(model, instructions=DISTILL_INSTRUCTIONS, model_settings=settings)
