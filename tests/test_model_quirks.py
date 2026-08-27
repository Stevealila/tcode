"""`normalize_tool_namespaces` strips the harmony `functions/` prefix that
gpt-oss on Groq intermittently puts in front of a tool name.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from src.model_quirks import _strip_namespace, normalize_tool_namespaces

# (input, expected). `known` for the test is {read_file, write_file}.
_STRIP_CASES = [
    ("functions/read_file", "read_file"),
    ("functions.read_file", "read_file"),
    ("functions:read_file", "read_file"),
    ("read_file", "read_file"),  # already valid — untouched
    ("functions/find_files", "find_files"),  # tail unknown, head is a namespace token
    ("some.legit_tool", "some.legit_tool"),  # unknown, non-namespace head — left alone
]


@pytest.mark.parametrize("name,expected", _STRIP_CASES)
def test_strip_namespace(name, expected):
    assert _strip_namespace(name, frozenset({"read_file", "write_file"})) == expected


def _echo_tool(value: str) -> str:
    return value


def _make_agent(response_tool_name: str) -> Agent:
    calls: list[int] = []

    def respond(messages, info: AgentInfo) -> ModelResponse:
        # First call: emit the (possibly namespaced) tool call. Second: finish.
        if not calls:
            calls.append(1)
            return ModelResponse(
                parts=[ToolCallPart(response_tool_name, {"value": "hi"})]
            )
        return ModelResponse(parts=[TextPart("done")])

    async def respond_stream(messages, info: AgentInfo):
        if not calls:
            calls.append(1)
            yield {
                0: DeltaToolCall(name=response_tool_name, json_args='{"value": "hi"}')
            }
        else:
            yield "done"

    agent = Agent(
        normalize_tool_namespaces(
            FunctionModel(respond, stream_function=respond_stream)
        )
    )
    agent.tool_plain(_echo_tool)
    return agent


def test_namespaced_tool_call_is_dispatched():
    agent = _make_agent("functions/_echo_tool")
    result = asyncio.run(agent.run("go"))
    # If the prefix leaked through, the tool would never run and the model
    # would get an "Unknown tool name" retry instead.
    tool_returns = [
        m
        for msg in result.all_messages()
        for m in getattr(msg, "parts", [])
        if type(m).__name__ == "ToolReturnPart"
    ]
    assert any(p.tool_name == "_echo_tool" and p.content == "hi" for p in tool_returns)


def test_bare_tool_call_still_works():
    agent = _make_agent("_echo_tool")
    result = asyncio.run(agent.run("go"))
    assert result.output == "done"


def test_streaming_path_normalizes():
    agent = _make_agent("functions/_echo_tool")

    async def go() -> str:
        async with agent.run_stream("go") as stream:
            return await stream.get_output()

    # Driving the full stream + tool dispatch without an UnexpectedModelBehavior
    # means the streamed path normalized the tool name too.
    assert asyncio.run(go()) == "done"
