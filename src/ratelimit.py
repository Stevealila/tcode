"""Client-side requests-per-minute throttle, shared across tcode processes.

Groq enforces RPM per account, separately from tokens-per-minute (you hit
whichever limit arrives first), so even small, well-under-budget requests
can still get a raw 429 if fired too rapidly. A purely in-process limiter
would only protect a single long-running interactive session, so the state
lives in one shared file under ~/.tcode instead: a one-shot invocation run
right after (or alongside) another one, or an interactive session running
concurrently with a script, all see the same rolling budget.
"""

from __future__ import annotations

import asyncio
import fcntl
import time
from collections.abc import Callable
from pathlib import Path

WINDOW_SECONDS = 60.0
_RECHECK_INTERVAL = 2.0


def _try_reserve_slot(state_file: Path, max_rpm: int) -> float:
    """Return 0.0 and record a request if there's room in the trailing
    window, otherwise return the number of seconds until there will be.
    """
    with open(state_file, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            now = time.time()
            cutoff = now - WINDOW_SECONDS
            timestamps = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts = float(line)
                except ValueError:
                    continue
                if ts > cutoff:
                    timestamps.append(ts)

            if len(timestamps) < max_rpm:
                timestamps.append(now)
                f.seek(0)
                f.truncate()
                f.writelines(f"{ts}\n" for ts in timestamps)
                return 0.0

            return (min(timestamps) + WINDOW_SECONDS) - now
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


async def throttle(
    state_file: Path,
    max_rpm: int,
    *,
    on_wait: Callable[[float], None] | None = None,
) -> None:
    """Block until issuing another request won't exceed max_rpm in the
    trailing 60s. Reserves the slot itself, so callers don't need to
    separately record the request.
    """
    if max_rpm <= 0:
        return

    state_file.parent.mkdir(parents=True, exist_ok=True)
    announced = False

    while True:
        wait_for = _try_reserve_slot(state_file, max_rpm)
        if wait_for <= 0:
            return
        if not announced:
            if on_wait:
                on_wait(wait_for)
            announced = True
        await asyncio.sleep(min(wait_for, _RECHECK_INTERVAL))
