"""Project-slug shape and the one-shot migration off the old `-`-prefixed dirs."""

from __future__ import annotations

from pathlib import Path

from src.config import _migrate_legacy_project_dirs, _slugify


def test_slug_has_no_leading_dash():
    slug = _slugify(Path("/home/alice/some-project"))
    assert slug == "home-alice-some-project"
    assert not slug.startswith("-")


def test_slug_root():
    assert _slugify(Path("/")) == "root"


def test_migrate_renames_every_legacy_dir(tmp_path):
    projects = tmp_path / "projects"
    (projects / "-home-alice-x" / "sessions").mkdir(parents=True)
    (projects / "-home-alice-x" / "sessions" / "s.json").write_text("{}")
    (projects / "-tmp-claude--home-alice-y").mkdir()
    (projects / "already-fine").mkdir()

    _migrate_legacy_project_dirs(projects)

    assert not (projects / "-home-alice-x").exists()
    assert (projects / "home-alice-x" / "sessions" / "s.json").read_text() == "{}"
    # only the leading dash is stripped — internal `--` is preserved
    assert (projects / "tmp-claude--home-alice-y").is_dir()
    assert (projects / "already-fine").is_dir()


def test_migrate_keeps_legacy_when_target_exists(tmp_path):
    projects = tmp_path / "projects"
    (projects / "-home-alice-x").mkdir(parents=True)
    (projects / "home-alice-x").mkdir()

    _migrate_legacy_project_dirs(projects)

    # de-dashed name already taken — don't clobber, leave the old one alone
    assert (projects / "-home-alice-x").is_dir()
    assert (projects / "home-alice-x").is_dir()


def test_migrate_noop_when_projects_dir_missing(tmp_path):
    _migrate_legacy_project_dirs(tmp_path / "nope")  # must not raise


def test_load_config_uses_dashless_slug(tcode_cfg):
    # `tcode_cfg` builds a Config over a throwaway ~/.tcode; the slug is derived
    # from a real tmp workspace path and must not start with `-`.
    assert not tcode_cfg.project_slug.startswith("-")
    assert tcode_cfg.project_dir.is_dir()
