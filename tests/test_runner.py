"""Tests for runner.py: the faked-tool-call / skipped-reading heuristics
(pure functions written after specific observed failure modes — see their
docstrings), plus one end-to-end run_turn against a stubbed model so the
telemetry write path itself is exercised. See betterment/plan.txt 3.9.b.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from src.runner import _looks_like_faked_tool_call, _skipped_reading_files


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"name": "write_file", "arguments": {"path": "x.py"}}', True),
        ('  {  "name" : "edit_file" , "arguments" :  {}}', True),
        ('{"name": "write_file"}', False),  # no "arguments" key
        ("Here is the answer you wanted.", False),
        ("", False),
        ('The tool call looked like {"name": "x", "arguments": ...}', False),  # not at start
    ],
)
def test_looks_like_faked_tool_call(text, expected):
    assert _looks_like_faked_tool_call(text) is expected


class TestSkippedReadingFiles:
    def test_no_tools_at_all_is_not_flagged(self):
        assert _skipped_reading_files(Counter(), "x" * 2000) is False

    def test_read_file_present_is_not_flagged(self):
        assert _skipped_reading_files(Counter({"read_file": 1}), "x" * 2000) is False

    def test_long_answer_without_read_file_is_flagged(self):
        assert _skipped_reading_files(Counter({"list_directory": 3}), "x" * 900) is True

    def test_short_answer_without_read_file_is_not_flagged(self):
        assert _skipped_reading_files(Counter({"list_directory": 3}), "short") is False

    def test_markdown_table_without_read_file_is_flagged_even_if_short(self):
        text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        assert _skipped_reading_files(Counter({"list_directory": 1}), text) is True


def test_run_turn_writes_a_smell_record(tcode_cfg):
    from pydantic_ai import Agent, UsageLimits
    from pydantic_ai.models.test import TestModel

    from src.runner import run_turn
    from src.telemetry import load_smell_log

    cfg = tcode_cfg
    agent = Agent(TestModel())

    assert load_smell_log(cfg) == []
    _, final_text = asyncio.run(
        run_turn(agent, "hello there", [], UsageLimits(request_limit=5), cfg, capture=True)
    )

    records = load_smell_log(cfg)
    assert len(records) == 1
    rec = records[0]
    assert rec["provider"] == "groq"
    assert rec["prompt_length"] == len("hello there")
    assert rec["retry_count"] == 0
    assert rec["total_tokens"] is not None
    assert rec["outcome"] == "ok"
    assert isinstance(final_text, str)


def test_run_turn_records_a_smell_line_when_the_turn_fails(tcode_cfg):
    """A turn that loops until it hits the request limit is exactly the
    regression --backtest exists to catch — it must leave a smell record
    even though it raises. See betterment/plan.txt 6.2 D1.
    """
    from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits
    from pydantic_ai.models.test import TestModel

    from src.runner import run_turn
    from src.telemetry import load_smell_log

    cfg = tcode_cfg
    agent = Agent(TestModel())

    @agent.tool_plain
    def ping() -> str:  # TestModel calls every tool once -> a 2nd request
        return "pong"

    with pytest.raises(UsageLimitExceeded):
        asyncio.run(
            run_turn(agent, "loop forever", [], UsageLimits(request_limit=1), cfg, capture=True)
        )

    records = load_smell_log(cfg)
    assert len(records) == 1
    assert records[0]["outcome"] == "usage_limit"
    assert "UsageLimitExceeded" in (records[0]["error"] or "")


def test_run_turn_salvages_text_and_records_when_a_tool_call_finally_fails(tcode_cfg):
    """The garbled-tool-call salvage branch: text was already produced, the
    model's own tool call then blew the retry budget — return the text but
    record the turn with outcome 'salvaged_after_tool_failure'.
    """
    from pydantic_ai import Agent, ModelRetry, UsageLimits
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    from src.runner import run_turn
    from src.telemetry import load_smell_log

    def model_fn(messages, info) -> ModelResponse:
        return ModelResponse(parts=[TextPart("Here is my analysis."), ToolCallPart("boom", {})])

    async def stream_fn(messages, info):
        yield "Here is my analysis."
        yield {0: DeltaToolCall(name="boom", json_args="{}")}

    cfg = tcode_cfg
    agent = Agent(FunctionModel(model_fn, stream_function=stream_fn), retries={"tools": 1})

    @agent.tool_plain
    def boom() -> str:
        raise ModelRetry("nope")

    _, final_text = asyncio.run(
        run_turn(agent, "analyze the repo", [], UsageLimits(request_limit=10), cfg, capture=True)
    )
    assert "analysis" in final_text

    records = load_smell_log(cfg)
    assert len(records) == 1
    assert records[0]["outcome"] == "salvaged_after_tool_failure"
    assert records[0]["error"]


def test_run_turn_retries_then_gives_up_on_a_persistent_faked_tool_call(tcode_cfg):
    """The _MAX_EMPTY_TURN_RETRIES loop: a model that keeps printing its
    tool call as JSON text instead of invoking it is retried the whole way,
    then returned as-is with outcome 'faked_tool_call' and retry_count at
    the ceiling. Distinct from the salvage branch (that one raises
    UnexpectedModelBehavior mid-turn; this one completes each attempt with
    a full but malformed answer). See betterment/plan.txt O4.
    """
    from pydantic_ai import Agent, UsageLimits
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse, TextPart

    from src.runner import _MAX_EMPTY_TURN_RETRIES, run_turn
    from src.telemetry import load_smell_log

    leak = '{"name": "write_file", "arguments": {"path": "out.md", "content": "hi"}}'
    attempts = []

    def model_fn(messages, info) -> ModelResponse:
        return ModelResponse(parts=[TextPart(leak)])

    async def stream_fn(messages, info):
        attempts.append(1)
        yield leak

    cfg = tcode_cfg
    agent = Agent(FunctionModel(model_fn, stream_function=stream_fn))

    _, final_text = asyncio.run(
        run_turn(agent, "write the file", [], UsageLimits(request_limit=20), cfg, capture=True)
    )

    assert final_text.strip() == leak  # returned as-is, not discarded
    assert len(attempts) == _MAX_EMPTY_TURN_RETRIES + 1  # initial + every retry

    records = load_smell_log(cfg)
    assert len(records) == 1  # one record for the whole turn, not one per retry
    assert records[0]["outcome"] == "faked_tool_call"
    assert records[0]["retry_count"] == _MAX_EMPTY_TURN_RETRIES
