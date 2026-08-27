"""Model-catalogue lookup for the interactive `/model` command.

Deliberately small and network-optional: the live `GET /openai/v1/models`
call is best-effort, and every failure path falls back to a static list so
`/model` keeps working offline or when Groq is unreachable.
"""

from __future__ import annotations

from .config import _BANNED_MODEL_SUBSTRINGS

# Groq's chat catalogue as last verified (2026-08-27). Used only when the
# live models call fails — offline, key rejected, Groq down. Most useful
# general-purpose model first.
GROQ_STATIC_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "groq/compound",
    "groq/compound-mini",
)

_GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# Substrings marking a listed model as unusable as a primary chat model
# (speech, moderation classifiers). Matched case-insensitively. Banned
# models (config._BANNED_MODEL_SUBSTRINGS) are filtered here too so they
# never show up as a pickable option or a fuzzy-match target.
_NON_CHAT = ("whisper", "tts", "orpheus", "prompt-guard")

# A coding agent needs real context headroom; anything this small (Groq's
# tiny speech/embedding-adjacent and legacy models) can't hold tcode's own
# conversation budget. Only applied when the catalogue entry reports a
# context window at all.
_MIN_CONTEXT = 16384


def _is_pickable(model_id: str, context_window: object = None) -> bool:
    low = model_id.lower()
    if not low:
        return False
    if any(h in low for h in _NON_CHAT):
        return False
    if any(b in low for b in _BANNED_MODEL_SUBSTRINGS):
        return False
    return not (isinstance(context_window, int) and context_window < _MIN_CONTEXT)


def list_groq_models(api_key: str, *, timeout: float = 4.0) -> tuple[list[str], str | None]:
    """`(sorted pickable model ids, error_or_None)`.

    Never raises. On any failure — no httpx, network error, non-200, weird
    body — returns the static fallback list and a one-line error string the
    caller can show.
    """
    try:
        import httpx

        resp = httpx.get(_GROQ_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:  # noqa: BLE001 - best-effort by design, see docstring
        return list(GROQ_STATIC_MODELS), f"{type(e).__name__}: {e}".strip()

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return list(GROQ_STATIC_MODELS), "unexpected response shape from Groq /models"

    ids = sorted(
        {
            m["id"]
            for m in data
            if isinstance(m, dict) and _is_pickable(str(m.get("id", "")), m.get("context_window"))
        }
    )
    return (ids or list(GROQ_STATIC_MODELS)), None


def resolve_model_choice(arg: str, catalogue: list[str]) -> tuple[str | None, list[str]]:
    """Map a user's `/model <arg>` to a concrete model id.

    Returns `(chosen, candidates)`. `chosen` is set only when the choice is
    unambiguous, tried in order of decreasing confidence:
      1. an exact (case-insensitive) catalogue hit;
      2. an explicit `provider:id` / `org/name` string — taken literally
         even if it isn't in the (Groq-only) catalogue, so a brand-new model
         or a `google:` / `zai:` one can still be selected;
      3. exactly one catalogue id whose bare name (the part after the last
         `/`) or whose last `-`-delimited segment equals the arg — so
         `compound` picks `groq/compound` (not `groq/compound-mini`) and
         `mini` picks `groq/compound-mini`;
      4. exactly one case-insensitive substring match.
    Otherwise `chosen` is None and `candidates` is what to show the user:
    empty means nothing matched, more than one means it was ambiguous (the
    narrowest non-empty tier's hits). A bare token that matches nothing
    returns `(None, [])` on purpose — better "no match, here's the list"
    than a silent typo'd model id that only fails on the next turn.
    """
    arg = arg.strip()
    if not arg:
        return None, []

    low = arg.lower()

    for m in catalogue:
        if m.lower() == low:
            return m, [m]

    if ":" in arg or "/" in arg:
        return arg, [arg]

    def _name_forms(model_id: str) -> set[str]:
        bare = model_id.rsplit("/", 1)[-1].lower()
        return {bare, bare.rsplit("-", 1)[-1]}

    segment_hits = [m for m in catalogue if low in _name_forms(m)]
    if len(segment_hits) == 1:
        return segment_hits[0], segment_hits

    substr_hits = [m for m in catalogue if low in m.lower()]
    if len(substr_hits) == 1:
        return substr_hits[0], substr_hits

    return None, segment_hits or substr_hits
