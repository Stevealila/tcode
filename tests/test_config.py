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


def _load_web(tmp_path, monkeypatch, **env):
    from src import config as config_mod

    monkeypatch.setattr(config_mod, "GLOBAL_DIR", tmp_path / ".tcode")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TCODE_WEB_SEARCH", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ws = tmp_path / "ws"
    ws.mkdir()
    return config_mod.load_config(ws)


def test_web_off_without_tavily_key(tmp_path, monkeypatch):
    cfg = _load_web(tmp_path, monkeypatch)  # web wanted by default, no key
    assert cfg.web_search is False
    assert cfg.web_disabled_no_key is True


def test_web_on_with_tavily_key(tmp_path, monkeypatch):
    cfg = _load_web(tmp_path, monkeypatch, TAVILY_API_KEY="tvly-x")
    assert cfg.web_search is True
    assert cfg.web_disabled_no_key is False


def test_web_explicitly_off_is_not_flagged_as_missing_key(tmp_path, monkeypatch):
    cfg = _load_web(tmp_path, monkeypatch, TCODE_WEB_SEARCH="0")
    assert cfg.web_search is False
    assert cfg.web_disabled_no_key is False  # user turned it off, not a missing-key gap


def test_web_off_even_with_key_when_disabled(tmp_path, monkeypatch):
    cfg = _load_web(tmp_path, monkeypatch, TAVILY_API_KEY="tvly-x", TCODE_WEB_SEARCH="0")
    assert cfg.web_search is False
    assert cfg.web_disabled_no_key is False
