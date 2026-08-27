"""Web search and fetch tools — both through Tavily.

SERP snippets and raw markdownified HTML both push a weak model toward
guessing: stale SEO copy reads as a current number, and a client-rendered
page's JS/CSS boilerplate reads as "no real content," which the model papers
over by inventing a plausible answer. Tavily is a search/extract API built
for agent consumption — already-extracted content, a finance/news topic
mode — so both tools start from clean material instead of noise.

`web_fetch` adds a second stage: hand Tavily's extracted markdown plus the
caller's *specific question* to a cheap Groq model (`distill.py`) that says
plainly when the page doesn't have the answer, so a blank or thin page
produces "not found" rather than material to guess from. Content under
`_DISTILL_SKIP_CHARS` skips that round-trip.

Both tools require `TAVILY_API_KEY`; without it `cfg.web_search` is False
and `web_capabilities` is never called (see config.py). Free tier, no card:
https://app.tavily.com.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic_ai import ModelRetry, Tool
from pydantic_ai.capabilities import Capability, WebFetch, WebSearch
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
    from pydantic_ai.common_tools.tavily import tavily_search_tool
    from tavily.errors import BadRequestError

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


def _make_web_fetch_tool(cfg: Config, provider: GroqProvider) -> Tool:
    from tavily import AsyncTavilyClient
    from tavily.errors import BadRequestError

    distill_agent = make_distill_agent(provider, cfg.distill_model)
    client = AsyncTavilyClient(api_key=cfg.tavily_api_key)

    async def web_fetch(url: str, prompt: str) -> str:
        # Tavily /extract: clean markdown instead of the raw HTML/JS a
        # markdownify pass leaves a weak model to hallucinate from. Same
        # BadRequestError-only catch as tavily_search — a malformed URL is
        # retryable, an auth/quota failure isn't (see that tool's comment).
        try:
            resp = await client.extract(url, format="markdown", query=prompt)
        except BadRequestError as e:
            raise ModelRetry(str(e)) from e

        results = resp.get("results") or []
        if not results:
            failed = resp.get("failed_results") or []
            reason = (failed[0].get("error") if failed and isinstance(failed[0], dict) else None) or "no content extracted"
            return f"(could not fetch {url}: {reason})"

        top = results[0]
        content = top.get("raw_content") or ""
        if len(content) < _DISTILL_SKIP_CHARS:
            return content or "(empty page)"

        # Same shared cross-process throttle the main turn loop uses (see
        # cli.py) — this is a second Groq request the main loop doesn't know
        # about, and Groq enforces RPM per account, not per call site.
        await throttle(cfg.rpm_state_file, cfg.max_rpm)
        distilled = await distill_agent.run(
            f"QUESTION: {prompt}\n\n"
            f"PAGE URL: {top.get('url', url)}\n\n"
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
    """Tavily search + a distilling Tavily-extract fetch. Only called when
    `cfg.web_search` is set, which already requires `cfg.tavily_api_key`."""
    return [
        WebSearch(native=False, local=_make_tavily_search_tool(cfg)),
        WebFetch(native=False, local=_make_web_fetch_tool(cfg, provider)),
        # Technical backstop: observed behavior shows the model reaching for
        # shell curl/wget instead of retrying web_fetch/the search tool when
        # a fetch fails. See guardrails.py's module docstring for the
        # incident that prompted this.
        ToolGuardrail(guard=prefer_web_fetch_tool),
    ]
