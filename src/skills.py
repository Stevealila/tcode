"""Deterministic, human-invoked skills: `~/.tcode/skills/*.md` loaded on
demand via `/skill <name>` in the interactive REPL.

Deliberately not the harness's own model-initiated `pydantic_ai_harness.
skills.Skills` capability (deferred-loading, the model decides when to load
one): a human deciding carries no invocation-reliability risk at all, and
this is cheap enough to ship first. The model-initiated capability stays a
possible follow-up once a low-stakes skill has enough real usage to show
whether a Groq-class model reaches for one reliably without being told to.
"""

from __future__ import annotations

from pathlib import Path


def list_skills(skills_dir: Path) -> list[str]:
    return sorted(p.stem for p in skills_dir.glob("*.md"))


def load_skill(skills_dir: Path, name: str) -> str | None:
    path = skills_dir / f"{name}.md"
    if not path.is_file():
        return None
    return path.read_text()
