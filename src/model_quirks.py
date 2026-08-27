"""Work around provider quirks that pydantic-ai otherwise surfaces as hard errors.

Currently one quirk, one workaround: gpt-oss models on Groq intermittently
address a tool by its OpenAI-harmony namespace — `functions/read_file`, and
sometimes `functions.read_file` — instead of the bare `read_file` the tool is
actually registered under. pydantic-ai rejects the call as an unknown tool,
and the weaker models this happens on most often can't recover from the retry
(observed: a model burning a whole turn alternating `functions/read_file`
failures with bare `list_directory` successes, then giving up).

`normalize_tool_namespaces` wraps a model so the namespace prefix is stripped
back off every tool call — on both the streamed and non-streamed paths —
before the agent graph tries to resolve it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel

_SEPARATORS = ("/", ".", ":")
# Namespace tokens gpt-oss/harmony puts in front of a tool name. Used only as a
# fallback when the bare tail isn't a currently-known tool (e.g. a deferred
# tool that tool-search hasn't revealed yet).
_NAMESPACE_TOKENS = frozenset({"functions", "function", "tools", "tool", "namespace"})


def _known_tool_names(params: ModelRequestParameters) -> frozenset[str]:
    names = {td.name for td in params.function_tools}
    names.update(td.name for td in params.output_tools)
    names.update(params.revealed_tool_names or ())
    return frozenset(names)


def _strip_namespace(name: str, known: frozenset[str]) -> str:
    """`functions/read_file` -> `read_file`; a name already valid is left alone."""
    if not name or name in known:
        return name
    for sep in _SEPARATORS:
        head, found, tail = name.partition(sep)
        if not found or not tail:
            continue
        if tail in known or head in _NAMESPACE_TOKENS:
            return _strip_namespace(tail, known)
    return name


def _normalize_response(
    response: ModelResponse, known: frozenset[str]
) -> ModelResponse:
    changed = False
    parts = []
    for part in response.parts:
        if isinstance(part, ToolCallPart):
            fixed = _strip_namespace(part.tool_name, known)
            if fixed != part.tool_name:
                part = replace(part, tool_name=fixed)
                changed = True
        parts.append(part)
    return replace(response, parts=parts) if changed else response


def _patch_stream_get(stream: StreamedResponse, known: frozenset[str]) -> None:
    """Normalize the assembled `ModelResponse` the agent resolves tools from.

    The agent graph dispatches tools from `stream.get()` (the continuation
    stitcher folds each segment's `sub.get()` into the response it returns),
    not from the raw stream events — and it re-derives the tool-call events
    from those same parts — so shadowing `get` on the segment stream fixes
    dispatch, the visible tool name, and the agent-smell telemetry at once.
    `StreamedResponse` is a plain dataclass (it needs `__dict__` for its own
    `cached_property`), so the instance attribute reliably shadows the method.
    """
    original_get = stream.get

    def get() -> ModelResponse:
        return _normalize_response(original_get(), known)

    stream.get = get  # type: ignore[method-assign]


class _NamespaceStrippingModel(WrapperModel):
    async def request(self, messages, model_settings, model_request_parameters):
        response = await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )
        return _normalize_response(
            response, _known_tool_names(model_request_parameters)
        )

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ):
        known = _known_tool_names(model_request_parameters)
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            _patch_stream_get(stream, known)
            yield stream


def normalize_tool_namespaces(model: Model) -> Model:
    """Wrap `model` so namespaced tool calls (`functions/read_file`) are
    stripped to the bare tool name before the agent resolves them."""
    return _NamespaceStrippingModel(model)
