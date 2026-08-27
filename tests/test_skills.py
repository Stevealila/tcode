from __future__ import annotations

from src.skills import list_skills, load_skill


def test_list_skills_empty_dir(tmp_path):
    assert list_skills(tmp_path) == []


def test_list_skills_sorted_by_name(tmp_path):
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "not-a-skill.txt").write_text("ignored")
    assert list_skills(tmp_path) == ["a", "b"]


def test_load_skill_returns_content(tmp_path):
    (tmp_path / "review.md").write_text("focus on correctness first")
    assert load_skill(tmp_path, "review") == "focus on correctness first"


def test_load_skill_missing_returns_none(tmp_path):
    assert load_skill(tmp_path, "missing") is None
