"""Wiring checks for instruction-file autoload (TCODE.md and friends).

These pin the two things that are easy to regress silently: the
within-directory precedence order encoded in `_INSTRUCTION_FILENAMES`, and
the split between the global `~/.tcode/TCODE.md` RepoContext and the
workspace walk-up one.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai_harness import RepoContext
from pydantic_ai_harness.repo_context._loader import discover_instruction_files

from src.agent import _INSTRUCTION_FILENAMES, _repo_context


def test_instruction_filenames_precedence_order():
    # RepoContext renders a directory's files in tuple order and the model
    # weights the last block most, so tcode's own files must come last and
    # the personal override last of all.
    assert _INSTRUCTION_FILENAMES == (
        "CLAUDE.md",
        "AGENTS.md",
        "TCODE.md",
        "TCODE.local.md",
    )


def test_repo_context_splits_global_and_workspace():
    global_rc, workspace_rc = _repo_context(Path("/ws"), home_dir=Path("/"))

    assert isinstance(global_rc, RepoContext) and isinstance(workspace_rc, RepoContext)

    # Global: rooted at ~/.tcode, only TCODE.md, no walk-up, and it must not
    # register the asset-inventory tool (the workspace instance owns that
    # name).
    assert global_rc.workspace_dir == Path.home() / ".tcode"
    assert global_rc.filenames == ("TCODE.md",)
    assert global_rc.home_dir is None
    assert global_rc.expose_inventory_tool is False

    # Workspace: full filename tuple, walk-up to home, nested-on-traversal on.
    assert workspace_rc.workspace_dir == Path("/ws")
    assert workspace_rc.home_dir == Path("/")
    assert workspace_rc.filenames == _INSTRUCTION_FILENAMES
    assert workspace_rc.nested_traversal is True


def test_explorer_repo_context_has_no_walkup():
    _, workspace_rc = _repo_context(Path("/ws"), home_dir=None)
    assert workspace_rc.home_dir is None


def test_tcode_local_wins_over_borrowed_files_in_same_dir(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("from claude")
    (tmp_path / "TCODE.md").write_text("from tcode")
    (tmp_path / "TCODE.local.md").write_text("from tcode local")

    found = discover_instruction_files(
        workspace_dir=tmp_path,
        home_dir=None,
        filenames=_INSTRUCTION_FILENAMES,
    )
    # Rendered order is discovery order; the last one is closest to the
    # model's recency window and therefore wins on conflict.
    assert [f.path.name for f in found] == [
        "CLAUDE.md",
        "TCODE.md",
        "TCODE.local.md",
    ]
