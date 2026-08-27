"""Table-driven tests for cli.py's pure functions — the leaked-tool-call
recovery and write-confirmation-detection helpers, each written after a
specific observed regression (see their own docstrings in cli.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import (
    _closest_smell_record,
    _extract_leaked_write_payload,
    _last_user_prompt,
    _looks_like_leaked_tool_call,
    _looks_like_write_confirmation_only,
    _unescape_json_string_prefix,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"name": "write_file", "arguments": {"path": "x.py"}}', True),
        ("{'name': 'write_file', 'arguments': {}}", True),
        ("<tool_call><function=write_file>", True),
        ("Here's the summary you asked for.", False),
        ("", False),
        ("  <tool_call> mid-sentence mention", True),
    ],
)
def test_looks_like_leaked_tool_call(text, expected):
    assert _looks_like_leaked_tool_call(text) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello", "hello"),
        ("line1\\nline2", "line1\nline2"),
        ("tab\\there", "tab\there"),
        ('quote\\"here', 'quote"here'),
        ("unicode\\u0041end", "unicodeAend"),
        ("dangling backslash at end\\", "dangling backslash at end"),
        ("incomplete escape\\u12", "incomplete escape"),
    ],
)
def test_unescape_json_string_prefix(raw, expected):
    assert _unescape_json_string_prefix(raw) == expected


class TestExtractLeakedWritePayload:
    def test_complete_json_call_recovers_content(self):
        text = '{"name": "write_file", "arguments": {"path": "x.md", "content": "hello world"}}'
        value, truncated = _extract_leaked_write_payload(text)
        assert value == "hello world"
        assert truncated is False

    def test_complete_json_call_recovers_new_text_for_edit(self):
        text = '{"name": "edit_file", "arguments": {"path": "x.md", "new_text": "updated"}}'
        value, truncated = _extract_leaked_write_payload(text)
        assert value == "updated"
        assert truncated is False

    def test_truncated_json_recovers_partial_content(self):
        text = '{"name": "write_file", "arguments": {"path": "x.md", "content": "partial answer that got cut off'
        value, truncated = _extract_leaked_write_payload(text)
        assert value == "partial answer that got cut off"
        assert truncated is True

    def test_truncated_mid_escape_drops_incomplete_unicode(self):
        text = '{"name": "write_file", "arguments": {"path": "x.md", "content": "text then\\u12'
        value, truncated = _extract_leaked_write_payload(text)
        assert value == "text then"
        assert truncated is True

    def test_no_recoverable_payload_returns_none(self):
        value, truncated = _extract_leaked_write_payload("just plain prose, nothing leaked")
        assert value is None
        assert truncated is False


class TestLooksLikeWriteConfirmationOnly:
    def test_short_confirmation_mentioning_path_name(self, tmp_path):
        path = tmp_path / "rollup.md"
        assert _looks_like_write_confirmation_only(
            "The rollup has been written to rollup.md.", path
        )

    def test_short_confirmation_mentioning_path_stem(self, tmp_path):
        path = tmp_path / "rollup.md"
        assert _looks_like_write_confirmation_only("Saved rollup to disk.", path)

    def test_real_terse_answer_is_not_flagged(self, tmp_path):
        path = tmp_path / "rollup.md"
        assert not _looks_like_write_confirmation_only("Nothing new this window.", path)

    def test_long_answer_is_not_flagged_even_if_it_mentions_the_path(self, tmp_path):
        path = tmp_path / "rollup.md"
        long_text = "written to rollup.md. " + ("Detailed findings follow. " * 50)
        assert not _looks_like_write_confirmation_only(long_text, path)

    def test_unrelated_short_text_is_not_flagged(self, tmp_path):
        path = tmp_path / "rollup.md"
        assert not _looks_like_write_confirmation_only("All good here.", path)


class TestLastUserPrompt:
    def test_returns_none_for_empty_history(self):
        assert _last_user_prompt([]) is None

    def test_returns_most_recent_user_prompt(self):
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        messages = [
            ModelRequest(parts=[UserPromptPart(content="first")]),
            ModelResponse(parts=[TextPart(content="answer 1")]),
            ModelRequest(parts=[UserPromptPart(content="second")]),
            ModelResponse(parts=[TextPart(content="answer 2")]),
        ]
        assert _last_user_prompt(messages) == "second"

    def test_ignores_non_string_content(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        messages = [ModelRequest(parts=[UserPromptPart(content="text prompt")])]
        assert _last_user_prompt(messages) == "text prompt"


class TestClosestSmellRecord:
    def test_picks_nearest_record_before_archive_time(self):
        import datetime as dt

        now = dt.datetime(2026, 8, 26, 12, 0, 0)
        records = [
            {"timestamp": (now - dt.timedelta(seconds=200)).isoformat(), "tag": "too_old"},
            {"timestamp": (now - dt.timedelta(seconds=5)).isoformat(), "tag": "closest"},
            {"timestamp": (now + dt.timedelta(seconds=5)).isoformat(), "tag": "after_archive"},
        ]
        best = _closest_smell_record(records, now)
        assert best["tag"] == "closest"

    def test_returns_none_when_nothing_within_window(self):
        import datetime as dt

        now = dt.datetime(2026, 8, 26, 12, 0, 0)
        records = [{"timestamp": (now - dt.timedelta(seconds=999)).isoformat()}]
        assert _closest_smell_record(records, now, window_s=120.0) is None

    def test_skips_malformed_records(self):
        import datetime as dt

        now = dt.datetime(2026, 8, 26, 12, 0, 0)
        records = [{"no_timestamp": True}, {"timestamp": "not-a-date"}]
        assert _closest_smell_record(records, now) is None
