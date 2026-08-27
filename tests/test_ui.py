"""Tests for ui.py's backtest_regression: the four-signal
regression-smell heuristic --backtest flags rows with.
"""

from __future__ import annotations

from src.ui import backtest_regression


def _rec(tools=0, retries=0, elapsed=None, tokens=None, outcome="ok"):
    return {
        "tool_counts": {"read_file": tools} if tools else {},
        "retry_count": retries,
        "elapsed_s": elapsed,
        "total_tokens": tokens,
        "outcome": outcome,
    }


class TestBacktestRegression:
    def test_none_when_either_side_missing(self):
        assert backtest_regression(None, _rec()) is None
        assert backtest_regression(_rec(), None) is None

    def test_flat_run_is_not_a_regression(self):
        old = _rec(tools=3, retries=0, elapsed=10.0, tokens=1000)
        new = _rec(tools=4, retries=0, elapsed=12.0, tokens=1200)
        assert backtest_regression(old, new) is None

    def test_tool_count_jump_over_two(self):
        reason = backtest_regression(_rec(tools=2), _rec(tools=6))
        assert reason == "+4 tool calls"

    def test_retry_increase(self):
        reason = backtest_regression(_rec(retries=0), _rec(retries=1))
        assert reason == "+1 retries"

    def test_much_slower_same_tools(self):
        old = _rec(tools=3, elapsed=10.0)
        new = _rec(tools=3, elapsed=40.0)
        assert backtest_regression(old, new) == "4.0x slower"

    def test_slightly_slower_is_not_flagged(self):
        old = _rec(elapsed=10.0)
        new = _rec(elapsed=20.0)
        assert backtest_regression(old, new) is None

    def test_token_blowup(self):
        old = _rec(tokens=1000)
        new = _rec(tokens=2000)
        assert backtest_regression(old, new) == "2.0x tokens"

    def test_multiple_reasons_joined(self):
        old = _rec(tools=1, retries=0, elapsed=10.0, tokens=1000)
        new = _rec(tools=5, retries=2, elapsed=50.0, tokens=3000)
        reason = backtest_regression(old, new)
        assert reason == "+4 tool calls, +2 retries, 5.0x slower, 3.0x tokens"

    def test_missing_timing_data_skips_timing_signal(self):
        old = _rec(tools=1)
        new = _rec(tools=1)
        assert backtest_regression(old, new) is None

    def test_flags_replay_failure_outcome_even_without_baseline(self):
        assert backtest_regression(None, {"outcome": "usage_limit"}) == "replay: usage_limit"

    def test_flags_synthesized_replay_crash_row(self):
        new = {"outcome": "replay_crashed", "tool_counts": {}, "retry_count": 0}
        assert backtest_regression(_rec(tokens=1000), new) == "replay: replay_crashed"

    def test_salvaged_outcome_is_not_treated_as_a_failure(self):
        old = _rec(tools=1, elapsed=10.0, tokens=1000)
        new = _rec(tools=1, elapsed=11.0, tokens=1050, outcome="salvaged_after_tool_failure")
        assert backtest_regression(old, new) is None

    def test_ok_outcome_with_flat_metrics_is_clean(self):
        assert backtest_regression(_rec(tools=2, elapsed=10.0, tokens=1000, outcome="ok"),
                                   _rec(tools=2, elapsed=11.0, tokens=1050, outcome="ok")) is None
