"""Table-driven tests for reduce.py's pure functions: file discovery (the
@listfile convention), grouping-keyword extraction, and the coverage
heuristic — see reduce.py's module docstring for how these fit together.
"""

from __future__ import annotations

from pathlib import Path

from src.reduce import _group_keywords, _uncovered_groups, discover_files


class TestDiscoverFiles:
    def test_bare_glob_matches_files_only(self, tmp_path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "sub").mkdir()
        result = discover_files("*.md", tmp_path)
        assert result == [Path("a.md"), Path("b.md")]

    def test_recursive_glob(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.md").write_text("c")
        (tmp_path / "a.md").write_text("a")
        result = discover_files("**/*.md", tmp_path)
        assert set(result) == {Path("a.md"), Path("sub/c.md")}

    def test_plain_path_with_no_wildcard(self, tmp_path):
        (tmp_path / "single.md").write_text("x")
        result = discover_files("single.md", tmp_path)
        assert result == [Path("single.md")]

    def test_nonexistent_plain_path_returns_empty(self, tmp_path):
        result = discover_files("missing.md", tmp_path)
        assert result == []

    def test_listfile_convention(self, tmp_path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "c.md").write_text("c")
        listfile = tmp_path / "files.txt"
        listfile.write_text("a.md\n# a comment\n\nb.md\n")
        result = discover_files(f"@{listfile}", tmp_path)
        assert result == [Path("a.md"), Path("b.md")]

    def test_listfile_dedupes_and_sorts(self, tmp_path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")
        listfile = tmp_path / "files.txt"
        listfile.write_text("b.md\na.md\na.md\n")
        result = discover_files(f"@{listfile}", tmp_path)
        assert result == [Path("a.md"), Path("b.md")]

    def test_listfile_entries_can_be_globs(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x.md").write_text("x")
        (tmp_path / "sub" / "y.md").write_text("y")
        listfile = tmp_path / "files.txt"
        listfile.write_text("sub/*.md\n")
        result = discover_files(f"@{listfile}", tmp_path)
        assert result == [Path("sub/x.md"), Path("sub/y.md")]


class TestGroupKeywords:
    def test_subtracts_common_segments(self):
        keys = ["profiles/bessent", "profiles/powell"]
        result = _group_keywords(keys)
        assert result["profiles/bessent"] == {"bessent"}
        assert result["profiles/powell"] == {"powell"}

    def test_falls_back_to_full_segments_when_nothing_distinguishes(self):
        # Two distinct keys whose segment sets are identical once split
        # (order doesn't matter to a set) -- every segment is "common", so
        # subtracting it out would leave nothing to match on; the function
        # falls back to the full segment set instead of an empty one.
        keys = ["profiles/bessent", "bessent/profiles"]
        result = _group_keywords(keys)
        assert result["profiles/bessent"] == {"profiles", "bessent"}
        assert result["bessent/profiles"] == {"profiles", "bessent"}

    def test_single_group_keeps_all_segments(self):
        keys = ["profiles/bessent"]
        result = _group_keywords(keys)
        assert result["profiles/bessent"] == {"profiles", "bessent"}

    def test_splits_on_slash_underscore_and_dash(self):
        keys = ["a/b_c-d", "a/x"]
        result = _group_keywords(keys)
        assert result["a/b_c-d"] == {"b", "c", "d"}


class TestUncoveredGroups:
    def test_mentioned_group_is_not_flagged(self):
        files = [Path("profiles/bessent/daily.md"), Path("profiles/powell/daily.md")]
        covered = "Bessent's daily note discussed rates. Powell's note discussed inflation."
        assert _uncovered_groups(files, covered) == []

    def test_unmentioned_group_is_flagged(self):
        files = [Path("profiles/bessent/daily.md"), Path("profiles/powell/daily.md")]
        covered = "Bessent's daily note discussed rates. Nothing else new."
        assert _uncovered_groups(files, covered) == ["profiles/powell"]

    def test_case_insensitive_match(self):
        files = [Path("profiles/bessent/daily.md")]
        covered = "BESSENT made new remarks today."
        assert _uncovered_groups(files, covered) == []

    def test_short_keywords_below_min_length_are_ignored(self):
        # "ab"/"cd" are both 2 chars, under _MIN_KEYWORD_LEN=3 -- with no
        # keyword long enough to check, the group can't be flagged either way.
        files = [Path("ab/x.md"), Path("cd/x.md")]
        covered = "totally unrelated text"
        assert _uncovered_groups(files, covered) == []

    def test_word_boundary_prevents_substring_false_negative(self):
        # "east" appearing inside "northeastern" should NOT count as a match
        # for a group keyed on "east" -- \b enforces a real word boundary.
        # "west" is mentioned as a real word, so only the "east" group
        # should come back flagged.
        files = [Path("regions/east/x.md"), Path("regions/west/x.md")]
        covered = "The northeastern division and the west division both reported strong results."
        assert _uncovered_groups(files, covered) == ["regions/east"]
