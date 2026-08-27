"""Per-turn "agent smell" log: a cheap regression signal for model swaps.

tcode runs the same prompts against swappable providers/models (--model
groq:..., google:..., zai:...), and Groq's own catalog changes over time —
the only way to know a swap behaved differently used to be a human noticing.
This appends one JSON line per turn to smell.jsonl (same directory
sessions.py already owns) so a later run (see --backtest) can diff a new
model's behavior against the old.

Best-effort, not load-bearing: same posture sessions.py's save_session
takes, for the same reason (a logging write must never fail a turn).
"""

from __future__ import annotations

import datetime as _dt
import json

from .config import Config


def record_smell(
    cfg: Config,
    prompt: str,
    tool_counts: dict[str, int],
    retries: int,
    elapsed: float,
    usage: object,
) -> None:
    try:
        cost = getattr(usage, "cost", None)
        record = {
            "timestamp": _dt.datetime.now().isoformat(),
            "provider": cfg.provider,
            "model": cfg.model,
            # Length only, not the prompt text itself — avoid duplicating
            # potentially sensitive content into a second log.
            "prompt_length": len(prompt),
            "tool_counts": tool_counts,
            "retry_count": retries,
            "elapsed_s": elapsed,
            "total_tokens": getattr(usage, "total_tokens", None),
            # Decimal isn't JSON-serializable directly.
            "cost": float(cost) if cost is not None else None,
        }
        with (cfg.sessions_dir / "smell.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def load_smell_log(cfg: Config) -> list[dict]:
    """All records in smell.jsonl, oldest first. A corrupted line is
    skipped rather than failing the whole read — same best-effort posture
    record_smell itself takes."""
    path = cfg.sessions_dir / "smell.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
