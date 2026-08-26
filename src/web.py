"""Web search and fetch tools.

DuckDuckGo's snippets and a raw markdownify fetch both push a model toward
guessing: stale SEO copy reads as a current number, and a client-rendered
page's JS/CSS boilerplate reads as "no real content," which a model tends to
paper over by inventing a plausible-sounding answer. `_make_web_fetch_tool`
fixes the fetch side with a two-stage pipeline: fetch and convert to
markdown, then hand that content plus the caller's *specific question* to a
second, cheap Groq model (`distill.py`) that says plainly when the page
doesn't have the answer — so a blank or noisy page produces "not found"
instead of material to guess from. (Same fetch-then-distill split Claude
Code's own `WebFetch` uses.) Content under `_DISTILL_SKIP_CHARS` skips the
extra round-trip.

Search gets the same fix from the other side: DuckDuckGo stays the
zero-setup default, but `TAVILY_API_KEY` opts into Tavily — cleaner,
already-extracted content instead of raw snippets, with a `finance` topic
mode — auto-detected by the key's presence. Same provider-precedence shape
as OpenClaw's search-provider chain.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic_ai import ModelRetry, Tool
from pydantic_ai.capabilities import Capability, WebFetch, WebSearch
from pydantic_ai.common_tools.web_fetch import web_fetch_tool
from pydantic_ai.messages import BinaryContent
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai_harness.guardrails import ToolGuardrail

from .config import Config
from .distill import make_distill_agent
from .guardrails import prefer_web_fetch_tool
from .ratelimit import throttle

# Below this, there's nothing worth filtering (e.g. a short JSON API
# response) — skip the extra round-trip and return the raw content.
_DISTILL_SKIP_CHARS = 800


def current_time_instructions() -> str:
    now = datetime.now().astimezone()
    return (
        f"Current date/time: {now.strftime('%A, %Y-%m-%d %H:%M %Z')}. Treat "
        "this as ground truth for what's current — your training data has a "
        'cutoff well before this, and most of what you\'ll read on the web '
        "(cached snippets, evergreen pages) describes the past, not now."
    )


def _make_tavily_search_tool(cfg: Config) -> Tool:
    from tavily.errors import BadRequestError

    from pydantic_ai.common_tools.tavily import tavily_search_tool

    # topic/time_range/etc. are left unset (not _UNSET) so the model can set
    # them per call — 'finance'/'news' topics and a time_range are exactly
    # what a "what's the current price of X" query wants, and Tavily's own
    # defaults are sane when it doesn't.
    base_search = tavily_search_tool(api_key=cfg.tavily_api_key, max_results=5)

    # tavily-python raises its own exceptions straight out of client.search()
    # (confirmed live: 'fast'/'ultra-fast' search_depth combined with the
    # 'finance' topic above — a combination the model reaches for on its
    # own — gets rejected as a BadRequestError, HTTP 400) with no try/except
    # anywhere in pydantic_ai's TavilySearchTool.__call__, and pydantic_ai's
    # own on_tool_execute_error default is `raise error`, not recover.
    # Uncaught, that propagates straight out of run_turn and kills the whole
    # process instead of giving the model a correctable retry — the same
    # shape FileSystemToolset's own wrapper already handles for its
    # exceptions (PermissionError, FileNotFoundError, ..., converted to
    # ModelRetry). Deliberately NOT catching ForbiddenError/InvalidAPIKeyError
    # here too, despite being from the same errors.py: those map to HTTP
    # 401/403/432/433 (checked against Tavily's own API reference, not
    # assumed) — an invalid key or a plan/usage-limit ceiling, none of which
    # a retry with different search_depth/topic/etc. can ever fix. Catching
    # those would burn the tool's whole retry budget on an error no retry
    # solves, then fail anyway via UnexpectedModelBehavior, just slower and
    # under a "please retry" framing that hides what's actually a config or
    # billing problem — worse than letting it propagate immediately to
    # one_shot()'s catch-all, which at least surfaces it as what it is.
    async def tavily_search(
        query: str,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        topic: Literal["general", "news", "finance"] = "general",
        time_range: Literal["day", "week", "month", "year"] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list:
        """Searches Tavily for the given query and returns the results.

        Args:
            query: The search query to execute with Tavily.
            search_depth: The depth of the search.
            topic: The category of the search.
            time_range: The time range back from the current date to filter results.
            include_domains: List of domains to specifically include in the search results.
            exclude_domains: List of domains to specifically exclude from the search results.
        """
        try:
            return await base_search.function(
                query,
                search_depth=search_depth,
                topic=topic,
                time_range=time_range,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
        except BadRequestError as e:
            raise ModelRetry(str(e)) from e

    return Tool(tavily_search, name="tavily_search", description=base_search.description)


def _search_capability(cfg: Config) -> WebSearch:
    if cfg.tavily_api_key:
        return WebSearch(native=False, local=_make_tavily_search_tool(cfg))
    return WebSearch(native=False, local="duckduckgo")


def _make_web_fetch_tool(cfg: Config, provider: GroqProvider) -> Tool:
    distill_agent = make_distill_agent(provider, cfg.distill_model)
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
