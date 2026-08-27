"""Table-driven tests for guardrails.py's pure functions — see that
module's docstring for the production incidents each guard closes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.guardrails import (
    _is_curl_or_wget_fetch,
    _is_unscoped_recursive_ls,
    _looks_truncated,
    _write_text,
    citation_paths_exist,
    confidence_tags_need_citation,
    prefer_web_fetch_tool,
    scope_shell_exploration,
    scope_writes_to,
)
from pydantic_ai_harness.guardrails import ToolCallInfo, ToolResultInfo


def _call(name: str, **args) -> ToolCallInfo:
    return ToolCallInfo(name=name, args=args, tool_call_id="tc-1")


@pytest.mark.parametrize(
    "command,expected",
    [
        ("ls -R", True),
        ("ls -R .", True),
        ("ls -R ~", True),
        ("ls -R $HOME", True),
        (f"ls -R {Path.home()}", True),
        ("ls -R src/", False),
        ("ls -la", False),
        ("find . -name '*.py'", False),
        ("echo hi && ls -R", True),
        ("git status; ls -R /", False),  # "/" isn't in _BROAD_TARGETS
        ("not-ls -R", False),
    ],
)
def test_is_unscoped_recursive_ls(command, expected):
    assert _is_unscoped_recursive_ls(command) is expected


def test_scope_shell_exploration_blocks_unscoped_ls():
    result = scope_shell_exploration(_call("run_command", command="ls -R ."))
    assert result.action == "retry"


def test_scope_shell_exploration_allows_scoped_ls():
    result = scope_shell_exploration(_call("run_command", command="ls -R src/"))
    assert result.action == "allow"


def test_scope_shell_exploration_ignores_other_tools():
    result = scope_shell_exploration(_call("read_file", path="src/x.py"))
    assert result.action == "allow"


@pytest.mark.parametrize(
    "command,expected",
    [
        ("curl https://example.com", True),
        ("wget https://example.com/file.txt", True),
        ("curl -o out.html https://example.com", True),
        ("curl http://example.com", True),
        ("curl ftp://example.com", False),
        ("echo curl https://example.com", False),  # 'curl' isn't tokens[0]
        ("git log", False),
        ("curl --help", False),
    ],
)
def test_is_curl_or_wget_fetch(command, expected):
    assert _is_curl_or_wget_fetch(command) is expected


def test_prefer_web_fetch_tool_blocks_curl():
    result = prefer_web_fetch_tool(_call("run_command", command="curl https://x.com"))
    assert result.action == "retry"


def test_prefer_web_fetch_tool_allows_other_commands():
    result = prefer_web_fetch_tool(_call("run_command", command="git status"))
    assert result.action == "allow"


def test_write_text_prefers_content_then_new_text_then_empty():
    assert _write_text(_call("write_file", content="hello")) == "hello"
    assert _write_text(_call("edit_file", new_text="world")) == "world"
    assert _write_text(_call("create_directory", path="x")) == ""


class TestScopeWritesTo:
    def test_allows_write_inside_scope(self, tmp_path):
        guard = scope_writes_to(tmp_path, ["src/research"])
        result = guard(_call("write_file", path="src/research/notes.md", content="x"))
        assert result.action == "allow"

    def test_allows_write_at_scope_root(self, tmp_path):
        guard = scope_writes_to(tmp_path, ["src/research"])
        result = guard(_call("create_directory", path="src/research"))
        assert result.action == "allow"

    def test_blocks_write_outside_scope(self, tmp_path):
        guard = scope_writes_to(tmp_path, ["src/research"])
        result = guard(_call("write_file", path="src/other/notes.md", content="x"))
        assert result.action == "retry"

    def test_blocks_path_traversal_escape(self, tmp_path):
        guard = scope_writes_to(tmp_path, ["src/research"])
        result = guard(
            _call("write_file", path="src/research/../../outside.md", content="x")
        )
        assert result.action == "retry"


@pytest.mark.parametrize(
    "url,seen,expected",
    [
        ("https://example.com/page...", set(), True),
        ("https://example.com/page…", set(), True),
        ("https://example.com/pag", {"https://example.com/page"}, True),
        ("https://example.com/page", {"https://example.com/page"}, False),
        ("https://example.com/page", set(), False),
        ("https://example.com/page?a=1...", set(), True),
    ],
)
def test_looks_truncated(url, seen, expected):
    assert _looks_truncated(url, seen) is expected


class TestUrlLedgerViaGuardrails:
    def test_record_then_check_write_flags_truncated_copy(self):
        from src.guardrails import UrlLedger

        ledger = UrlLedger()
        ledger.record(
            ToolResultInfo(
                name="web_fetch",
                args={},
                tool_call_id="tc-1",
                result="see https://example.com/full-page-path for details",
            )
        )
        result = ledger.check_write(
            _call("write_file", content="source: https://example.com/full-page...")
        )
        assert result.action == "retry"

    def test_check_write_allows_exact_copy(self):
        from src.guardrails import UrlLedger

        ledger = UrlLedger()
        ledger.record(
            ToolResultInfo(
                name="web_fetch",
                args={},
                tool_call_id="tc-1",
                result="see https://example.com/full-page-path for details",
            )
        )
        result = ledger.check_write(
            _call("write_file", content="source: https://example.com/full-page-path")
        )
        assert result.action == "allow"


class TestCitationPathsExist:
    def test_allows_write_citing_existing_file(self, tmp_path):
        (tmp_path / "profiles").mkdir()
        (tmp_path / "profiles" / "notes.md").write_text("hi")
        guard = citation_paths_exist(tmp_path)
        result = guard(_call("write_file", content="see `profiles/notes.md`"))
        assert result.action == "allow"

    def test_rejects_write_citing_missing_file(self, tmp_path):
        guard = citation_paths_exist(tmp_path)
        result = guard(_call("write_file", content="see `profiles/missing.md`"))
        assert result.action == "retry"

    def test_ignores_bare_filenames_without_slash(self, tmp_path):
        guard = citation_paths_exist(tmp_path)
        result = guard(_call("write_file", content="see `README.md` for details"))
        assert result.action == "allow"

    def test_rejects_path_traversal_citation(self, tmp_path):
        (tmp_path.parent / "outside.md").write_text("x")
        guard = citation_paths_exist(tmp_path)
        result = guard(_call("write_file", content="see `profiles/../../outside.md`"))
        assert result.action == "retry"


class TestConfidenceTagsNeedCitation:
    def test_noop_when_no_tags_configured(self):
        guard = confidence_tags_need_citation([])
        result = guard(_call("write_file", content="[CONFIRMED] no source at all"))
        assert result.action == "allow"

    def test_rejects_tag_with_no_citation_on_line(self):
        guard = confidence_tags_need_citation(["[CONFIRMED]"])
        result = guard(_call("write_file", content="[CONFIRMED] revenue grew 10%"))
        assert result.action == "retry"

    def test_allows_tag_with_url_on_same_line(self):
        guard = confidence_tags_need_citation(["[CONFIRMED]"])
        result = guard(
            _call("write_file", content="[CONFIRMED] revenue grew 10% https://example.com")
        )
        assert result.action == "allow"

    def test_allows_tag_with_cited_path_on_same_line(self):
        guard = confidence_tags_need_citation(["[CONFIRMED]"])
        result = guard(
            _call("write_file", content="[CONFIRMED] revenue grew 10% `data/q3.csv`")
        )
        assert result.action == "allow"

    def test_citation_on_other_line_does_not_count(self):
        guard = confidence_tags_need_citation(["[CONFIRMED]"])
        result = guard(
            _call(
                "write_file",
                content="[CONFIRMED] revenue grew 10%\nsource: https://example.com",
            )
        )
        assert result.action == "retry"
