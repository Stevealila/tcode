"""Web search and fetch tools.

Two failure modes drove this design, both found by testing against real
queries, not guessed at:

1. DuckDuckGo's local search tool (the zero-setup default) returns index
   snippets for "live price"-style pages that are almost always stale: static
   SEO copy with a historical example number and an old date baked into the
   page text, not the number the page actually displays right now (that's
   client-rendered). A model answering straight from the snippet states last
   year's number as current.

2. The markdownify-based local fetch tool has the matching failure on the
   other side: a page whose real content is client-rendered JS returns mostly
   JS/CSS boilerplate as "content". Faced with that noise (or a fetch that
   flat-out failed), the model's path of least resistance was inventing a
   specific-sounding number and attributing it to the page it just failed to
   read — worse than saying nothing.

Neither is a fetching problem, so a fancier fetch (a headless browser, say)
doesn't fix it — the fix is putting a checkpoint between "content we
retrieved" and "claim the model makes." `_make_web_fetch_tool` builds a
two-stage pipeline: fetch and convert to markdown, then hand that content to
a second, cheap Groq model (`openai/gpt-oss-20b`) along with the *specific
question* the caller asked, so only a targeted, distilled answer reaches the
orchestrator — never the raw page. That model is instructed to say plainly
when the page doesn't contain the answer, which is what actually stops the
hallucination: a blank or noisy page produces "not found" instead of
material for the (weaker, more confident) primary model to run with. Content
under `_DISTILL_SKIP_CHARS` skips the extra round-trip; there's nothing to
filter. (Checked afterward against Claude Code's own `WebFetch` tool
description: it does the same fetch-then-distill split, for the same reason
— corroboration that the design holds, not the source of it.)

Search gets the equivalent fix from the other direction: DuckDuckGo's
snippets are the input that made problem 1 possible in the first place, so
the zero-setup default stays available (no query should hard-fail because a
key is missing) but isn't the only option. `TAVILY_API_KEY` opts into
Tavily — a search API returning cleaner, already-extracted content instead
of raw SERP snippets, with a `finance` topic mode that's a direct match for
"what's the current price of X" — auto-detected by the key's presence,
DuckDuckGo used otherwise. (Same provider-precedence shape other agent
harnesses converge on independently, e.g. OpenClaw's search-provider chain —
which is a sign this is the generic right answer, not a borrowed one.)
Nothing about any of this is trading-specific; "what's the current
price/version/score of X" is a generic agent capability gap, and the fix is
the same for all of it.
"""

from __future__ import annotations

from datetime import datetime

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import Capability, WebFetch, WebSearch
from pydantic_ai.common_tools.web_fetch import web_fetch_tool
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai_harness.guardrails import ToolGuardrail

from .config import Config
from .guardrails import prefer_web_fetch_tool
from .ratelimit import throttle

# Smallest/fastest model in Groq's current catalog: this is a narrow
# extraction task (one page, one question), not agentic reasoning, so it
# doesn't need the main model's capacity — and running it on a separate,
# smaller model keeps this distillation pass cheap relative to the budget
# the primary model's turn is already spending.
_DISTILL_MODEL = "openai/gpt-oss-20b"

# Below this, there's nothing worth filtering (e.g. a short JSON API
# response) — skip the extra round-trip and return the raw content.
_DISTILL_SKIP_CHARS = 800

_DISTILL_INSTRUCTIONS = """\
You read one fetched web page and answer one question about it for another
agent, which will act on your answer without seeing the page itself.

Quote the specific fact requested (a number, a date, a status) plus enough
context to judge how current it is — the date it's attributed to, the
source name. A number embedded in evergreen copy as a historical example
("if the price were $X...") or clearly dated in the past is not the answer,
even if it's the only number on the page — say the page doesn't have a
current figure rather than reporting it as one.

If the page is boilerplate, a JS/CSS shell, an error page, a paywall, or
otherwise doesn't contain an answer to the question, say so plainly in one
line. Never state a fact, a number, or a quote that isn't actually present
in the page content below — the agent reading your answer cannot check your
work against the original page.
"""


def current_time_instructions() -> str:
    now = datetime.now().astimezone()
    return (
        f"Current date/time: {now.strftime('%A, %Y-%m-%d %H:%M %Z')}. Treat "
        "this as ground truth for what's current — your training data has a "
        'cutoff well before this, and most of what you\'ll read on the web '
        "(cached snippets, evergreen pages) describes the past, not now."
    )


def _search_capability(cfg: Config) -> WebSearch:
    if cfg.tavily_api_key:
        from pydantic_ai.common_tools.tavily import tavily_search_tool

        # topic/time_range/etc. are left unset (not _UNSET) so the model can
        # set them per call — 'finance'/'news' topics and a time_range are
        # exactly what a "what's the current price of X" query wants, and
        # Tavily's own defaults are sane when it doesn't.
        return WebSearch(native=False, local=tavily_search_tool(api_key=cfg.tavily_api_key, max_results=5))
    return WebSearch(native=False, local="duckduckgo")


def _make_web_fetch_tool(cfg: Config, provider: GroqProvider) -> Tool:
    distill_model = GroqModel(_DISTILL_MODEL, provider=provider)
    distill_agent = Agent(distill_model, instructions=_DISTILL_INSTRUCTIONS)
    base_fetch = web_fetch_tool()

    async def web_fetch(url: str, prompt: str) -> str | BinaryContent:
        result = await base_fetch.function(url)
        if isinstance(result, BinaryContent):
            return result

        content = result["content"]
        if len(content) < _DISTILL_SKIP_CHARS:
            return content or "(empty page)"

        # Same shared cross-process throttle the main turn loop uses (see
        # cli.py) — this is a second Groq request the main loop doesn't know
        # about, and Groq enforces RPM per account, not per call site.
        await throttle(cfg.rpm_state_file, cfg.max_rpm)
        distilled = await distill_agent.run(
            f"QUESTION: {prompt}\n\n"
            f"PAGE URL: {result['url']}\n"
            f"PAGE TITLE: {result['title']}\n\n"
            f"--- PAGE CONTENT ---\n{content}"
        )
        return distilled.output

    return Tool(
        web_fetch,
        name="web_fetch",
        description=(
            "Fetch a URL and return just the part relevant to `prompt` — not "
            "the raw page. State exactly what you're looking for in `prompt` "
            '(e.g. "the current spot price and when it was quoted"), the way '
            "you'd ask someone else to read the page for you. Says plainly "
            "when the page doesn't have the answer, rather than returning "
            "unrelated content for you to guess from."
        ),
    )


def web_capabilities(cfg: Config, provider: GroqProvider) -> list[Capability]:
    """WebSearch + a distilling WebFetch. Only called when `cfg.web_search` is set."""
    return [
        _search_capability(cfg),
        WebFetch(native=False, local=_make_web_fetch_tool(cfg, provider)),
        # Technical backstop: observed behavior shows the model reaching for
        # shell curl/wget instead of retrying web_fetch/the search tool when
        # a fetch fails. See guardrails.py's module docstring for the
        # incident that prompted this.
        ToolGuardrail(guard=prefer_web_fetch_tool),
    ]
