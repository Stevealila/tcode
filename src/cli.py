"""Argument parsing and the interactive REPL loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from pydantic_ai import ModelHTTPError, UsageLimits
from pydantic_ai.messages import ModelMessage

from . import ui
from .agent import build_agent
from .config import Config, ConfigError, load_config
from .runner import run_turn
from .sessions import list_sessions, load_latest_session, save_session


def _resolve_write_path(cfg: Config, raw: str) -> Path:
    """Resolve --write's target relative to cfg.cwd, refusing to escape it.

    Same sandboxing intent as FileSystem's own root_dir, applied to a path
    this process writes directly rather than one a tool call resolves.
    """
    p = (cfg.cwd / raw).resolve()
    try:
        p.relative_to(cfg.cwd)
    except ValueError:
        raise ConfigError(
            f"--write path must stay inside the workspace ({cfg.cwd}), got {raw!r}"
        ) from None
    return p


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


_LEAKED_TOOL_CALL_PREFIXES = ('{"name"', "{'name'")
_LEAKED_TOOL_CALL_MARKERS = ("<tool_call>", "<function=")


def _looks_like_leaked_tool_call(text: str) -> bool:
    """True if `text` is a model's own tool-call syntax leaking out as plain
    assistant text instead of triggering a real structured call.

    Observed on Groq's hosting of more than one model family at large
    (~40-file/~50K-token) reduce inputs: gpt-oss models sometimes emit the
    call as a literal `{"name": "write_file", "arguments": {...}` JSON
    envelope instead of invoking it; qwen3.6-27b sometimes emits its
    provider's own `<tool_call><function=write_file>...` pseudo-XML the same
    way. Either way the text that reaches here is the tool call itself, not
    an answer — real prose essentially never contains these exact markers,
    so this is a structural signal, not a guess.
    """
    stripped = text.strip()
    if stripped.startswith(_LEAKED_TOOL_CALL_PREFIXES):
        return True
    return any(marker in text for marker in _LEAKED_TOOL_CALL_MARKERS)


_LEAKED_CONTENT_KEY = re.compile(r'"(?:content|new_text)"\s*:\s*"')
_JSON_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}


def _unescape_json_string_prefix(raw: str) -> str:
    """Decode JSON string escapes in `raw`, tolerating an incomplete
    trailing escape (a lone backslash, or a cut-off `\\uXXXX`) by dropping
    it rather than raising. `raw` may be a genuinely truncated JSON string
    value, not a complete, valid one — see
    `_extract_leaked_write_payload`'s docstring for why that happens.
    """
    out: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            break  # dangling backslash at the very end — drop it
        nxt = raw[i + 1]
        if nxt == "u":
            if i + 6 <= n:
                try:
                    out.append(chr(int(raw[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    break
            break  # incomplete \uXXXX at the end — drop it
        out.append(_JSON_ESCAPES.get(nxt, nxt))
        i += 2
    return "".join(out)


def _extract_leaked_write_payload(text: str) -> tuple[str | None, bool]:
    """Best-effort recovery of a write_file/edit_file call's own
    `content`/`new_text` argument from a leaked JSON-shaped tool call.

    Returns `(extracted_text, was_truncated)` — `extracted_text` is `None`
    when nothing recoverable was found at all. Tries a strict JSON parse
    first (the leak is genuinely complete, just never turned into a real
    call); falls back to a raw scan for the `"content"`/`"new_text"` key's
    string value when that fails, which is what happens when the model's
    own generation was cut off mid-JSON — observed on google:gemma-4-31b-it
    composing a large --reduce rollup: the same tool-call payload, retried
    three times by run_turn, grew a little further each time before
    truncating at the same spot in its output budget, not a one-off
    formatting fluke. A truncated partial recovery is still returned
    (flagged, not silently passed off as complete) rather than discarded:
    an incomplete-but-real, actually-cited rollup is worth far more to a
    caller than nothing — the same "salvage over discard" call --write's
    whole existence already makes for a cleaner failure shape.
    """
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        args = obj.get("arguments")
        if isinstance(args, dict):
            for key in ("content", "new_text"):
                value = args.get(key)
                if isinstance(value, str):
                    return value, False
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    match = _LEAKED_CONTENT_KEY.search(stripped)
    if not match:
        return None, False
    raw = stripped[match.end() :]
    i = 0
    while i < len(raw):
        if raw[i] == "\\":
            i += 2
            continue
        if raw[i] == '"':
            return _unescape_json_string_prefix(raw[:i]), False
        i += 1
    # No closing quote found anywhere in the leaked text — the model's own
    # generation was cut off mid-string, not just mid-JSON-structure.
    return _unescape_json_string_prefix(raw), True


def _apply_write_fallback(
    write_path: Path | None, before_mtime: float | None, final_text: str, *, quiet: bool
) -> None:
    """A Python-level safety net for --write, see build_parser()'s help text.

    Mtime-based rather than "did a write_file call to this exact path
    happen": that's what the caller actually cares about (is the file on
    disk different now?), and it's the same check profile_scout.sh/
    story_intake.sh/state_of_market.sh already had to write themselves in
    bash before this existed — moved here once so no caller has to
    reimplement it.
    """
    if write_path is None:
        return
    if _mtime(write_path) != before_mtime:
        return
    if not final_text.strip():
        return
    if _looks_like_leaked_tool_call(final_text):
        # Not automatically unsalvageable: the leaked JSON/pseudo-XML is
        # usually the model's own genuine write_file/edit_file call, just
        # never turned into a real one — its `content`/`new_text` argument
        # is often a perfectly good answer sitting right there. Recover it
        # if we can; only fall through to refusing the write (below) when
        # nothing recoverable is found at all. Writing the RAW leaked text
        # verbatim would still be worse than writing nothing (see the
        # comment on that path) — the difference here is trying to salvage
        # the payload first instead of jumping straight to refusal.
        salvaged, truncated = _extract_leaked_write_payload(final_text)
        if salvaged is not None:
            if truncated:
                salvaged += (
                    "\n\n[tcode: recovered from a leaked, incomplete tool "
                    "call — the model's own generation was cut off "
                    "partway through, so this stops mid-sentence/"
                    "mid-section. Treat it as a partial answer, not a "
                    "finished one.]"
                )
            write_path.parent.mkdir(parents=True, exist_ok=True)
            write_path.write_text(salvaged)
            ui.print_notice(
                f"--write: the model's own write didn't land this turn, "
                f"and its final answer looked like a leaked, unexecuted "
                f"tool call — recovered the "
                f"{'truncated ' if truncated else ''}content from inside "
                f"it and wrote that to {write_path} instead of leaving it "
                "empty.",
                quiet=quiet,
            )
            return
        # Writing this would be worse than writing nothing: it dresses up
        # as a successful save (file now exists, mtime changed, a
        # reassuring notice) while the content is the model's own
        # unexecuted tool call, not an answer. A caller whose retry logic
        # is exactly "did the file change" (the callers this fallback
        # exists for) would take that as success and never retry — see
        # _looks_like_leaked_tool_call's docstring for where this was
        # observed. Leaving the file untouched keeps this turn looking like
        # what it was: nothing usable produced, same as an empty answer.
        ui.print_notice(
            f"--write: the model's final answer looks like its own "
            f"unexecuted tool call, not a real answer — leaving {write_path} "
            "untouched instead of saving it.",
            quiet=quiet,
        )
        return
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(final_text)
    ui.print_notice(
        f"--write: the model's own write didn't land this turn — wrote its "
        f"answer to {write_path} directly",
        quiet=quiet,
    )


def _friendly_error(e: Exception) -> str:
    """Turn a raw exception into something a terminal user can act on.

    A provider's rate-limit response (HTTP 413/429) usually carries a
    precise, well-formed message naming exactly which budget was hit —
    requests or tokens, per minute or per day — and how long to wait.
    Surface that directly rather than guessing from the status code alone:
    a 429 can mean requests-per-minute (our own throttle() should prevent
    that one, though it only paces the *active* provider's own budget — see
    config.py's `_rpm_for`), but it can just as easily mean the *daily*
    token quota is exhausted, which is a completely different situation
    with a completely different fix (wait, it's unrelated to conversation
    size, /clear won't help). Deliberately provider-neutral in the
    fallback branch below (no detail to surface): this fires for whichever
    provider `--model` is actually pointed at, not just Groq.
    """
    if isinstance(e, ModelHTTPError) and e.status_code in (413, 429):
        detail = e.body.get("error", {}).get("message") if isinstance(e.body, dict) else None
        if detail:
            return f"{e.model_name} rejected the request (HTTP {e.status_code}): {detail}"
        return (
            f"{e.model_name} rejected the request (HTTP {e.status_code}) — "
            "likely a rate limit. Wait and try again, or switch models "
            "with --model."
        )
    return str(e)


async def interactive(cfg: Config, message_history: list[ModelMessage]) -> None:
    agent = build_agent(cfg)
    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    ui.print_banner(cfg)

    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(str(cfg.history_file))
    )

    while True:
        try:
            user_input = await prompt_session.prompt_async("you › ")
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/help":
            ui.print_help()
            continue
        if user_input == "/clear":
            message_history = []
            ui.print_notice("conversation cleared")
            continue
        if user_input == "/memory":
            ui.show_memory(cfg)
            continue
        if user_input == "/sessions":
            ui.show_sessions(list_sessions(cfg))
            continue

        try:
            message_history, _ = await run_turn(agent, user_input, message_history, usage_limits, cfg)
        except KeyboardInterrupt:
            ui.print_notice("interrupted")
            continue
        except Exception as e:  # noqa: BLE001 - keep the REPL alive on turn failures
            ui.print_error(_friendly_error(e))
            continue

        save_session(cfg, message_history)

    ui.print_notice("bye")


async def one_shot(
    cfg: Config,
    prompt: str,
    message_history: list[ModelMessage],
    *,
    quiet: bool = False,
    write_path: Path | None = None,
) -> None:
    agent = build_agent(cfg)
    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    before_mtime = _mtime(write_path) if write_path else None
    try:
        message_history, final_text = await run_turn(
            agent, prompt, message_history, usage_limits, cfg, quiet=quiet
        )
    except Exception as e:  # noqa: BLE001
        ui.print_error(_friendly_error(e))
        raise SystemExit(1) from e
    _apply_write_fallback(write_path, before_mtime, final_text, quiet=quiet)
    save_session(cfg, message_history)


def _debug_preview(text: str, limit: int = 200) -> str:
    """A stderr-safe preview of model output: bounded, and with brace
    characters swapped for lookalike fullwidth ones (｛｝, not {}).

    These previews land in --verify's disagreement-path stderr notices,
    which a caller like a subprocess wrapper may reasonably capture as
    fallback diagnostic text when stdout comes back empty. If that text
    still contained real `{...}` braces, a naive "scan for the first
    balanced JSON object" caller downstream could reconstruct and act on
    the very primary answer `--verify` just deliberately withheld — silently
    defeating the disagreement contract documented on verify_mode below.
    Swapping the ASCII braces for fullwidth lookalikes keeps the preview
    readable to a human while making it un-parseable as JSON.
    """
    body = text.strip().replace("\n", " ")[:limit]
    return body.translate({ord("{"): "｛", ord("}"): "｝"})


async def verify_mode(cfg: Config, prompt: str) -> None:
    """Independent-verifier decision mode — see verify.py's module docstring.

    On agreement, prints the primary's answer to stdout, same contract as
    `--quiet`. On disagreement, prints *nothing* to stdout and exits 2 —
    deliberately not a hedge, a pick-one, or an error message on stdout: a
    caller built around "stdout has a clean answer or it doesn't" (e.g.
    scanning for the first parseable JSON object) already treats empty/
    unparseable stdout as "this attempt produced nothing usable" and moves
    to its own fallback — which is exactly the right response to a verifier
    disagreement too. Inventing a different signal would need every such
    caller to learn a second failure shape for what is, to them, the same
    situation: no trustworthy answer this attempt.

    The stderr notices below use `_debug_preview`, not the raw texts,
    specifically so this path's diagnostics can never be mistaken for the
    withheld answer by a downstream best-effort JSON scanner — see that
    helper's docstring for the concrete failure this closes.
    """
    from . import verify as verify_mod

    agent = build_agent(cfg)
    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    timeout = int(os.environ.get("TCODE_VERIFY_TIMEOUT", "150"))

    try:
        _, primary_text = await run_turn(agent, prompt, [], usage_limits, cfg, capture=True)
        verifier_text = await verify_mod.get_verifier_answer(cfg, prompt, usage_limits, timeout)
        agreed, verdict = await verify_mod.compare(cfg, primary_text, verifier_text)
    except Exception as e:  # noqa: BLE001
        ui.print_error(_friendly_error(e))
        raise SystemExit(1) from e

    ui.print_notice(f"verify: primary={_debug_preview(primary_text)!r}", quiet=True)
    ui.print_notice(f"verify: verifier={_debug_preview(verifier_text)!r}", quiet=True)
    ui.print_notice(f"verify: verdict={_debug_preview(verdict)!r}", quiet=True)

    if agreed:
        sys.stdout.write(primary_text)
        if not primary_text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        return

    ui.print_notice("verify: DISAGREEMENT — withholding the answer (stdout empty, exit 2)", quiet=True)
    raise SystemExit(2)


async def reduce_mode(
    cfg: Config,
    prompt: str,
    pattern: str,
    message_history: list[ModelMessage],
    *,
    quiet: bool,
    write_path: Path | None = None,
) -> None:
    """Map-reduce over many files in one call — see reduce.py's module docstring."""
    from . import reduce as reduce_mod

    usage_limits = UsageLimits(request_limit=cfg.request_limit)
    before_mtime = _mtime(write_path) if write_path else None
    try:
        message_history, final_text = await reduce_mod.run_reduce(
            cfg, prompt, pattern, message_history, usage_limits, quiet=quiet
        )
    except Exception as e:  # noqa: BLE001
        ui.print_error(_friendly_error(e))
        raise SystemExit(1) from e
    _apply_write_fallback(write_path, before_mtime, final_text, quiet=quiet)
    save_session(cfg, message_history)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcode",
        description="A system-wide coding-agent harness powered by Groq models.",
    )
    parser.add_argument(
        "prompt", nargs="*", help="run a single prompt non-interactively and exit"
    )
    parser.add_argument(
        "-c", "--continue", dest="cont", action="store_true",
        help="continue the last session in this directory",
    )
    parser.add_argument("--model", default=None, help="override the Groq model for this run")
    parser.add_argument(
        "--sessions", action="store_true", help="list saved sessions for this directory and exit"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="one-shot mode only: send tool-call activity, the usage footer, and "
        "notices to stderr instead of stdout, so stdout is just the model's answer "
        "— for a script parsing the output (a JSON decision, say), not a human",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="one-shot mode only: get the answer, independently re-derive it with a "
        "different model (or TCODE_VERIFY_CMD, an external command) and only print "
        "it if they agree — stdout is empty and the exit code is 2 on disagreement. "
        "For a caller that shouldn't act on a confidently-wrong-but-clean answer. "
        "Implies --quiet's stdout contract; ignored with -c/--continue (verification "
        "needs an independent re-derivation, not a continued conversation).",
    )
    parser.add_argument(
        "--write", metavar="PATH", default=None,
        help="one-shot/--reduce only: PATH the run is expected to write via its own "
        "write_file call. If PATH's mtime is unchanged when the turn ends — the "
        "model answered in text instead of actually calling the tool, or the "
        "tool call itself failed — tcode writes the model's final answer there "
        "directly, so a caller with one known output path doesn't depend on the "
        "model's own write succeeding. Relative to this directory.",
    )
    parser.add_argument(
        "--reduce", metavar="PATTERN", default=None,
        help="one-shot mode only: PATTERN is a glob (relative to this directory, ** "
        "allowed) matching many files to read and reduce to one answer — the prompt "
        "describes what to extract per file and how to synthesize the result. "
        "@listfile reads an explicit newline-separated file/glob list instead, for "
        "selection logic a bare glob can't express (e.g. a caller's own date-range "
        "filter). Chunks internally (map each file, group/digest if there are more "
        "than a handful, then one final turn) rather than asking a single turn to "
        "read a long file list, which is unreliable regardless of file size — see "
        "reduce.py.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cfg = load_config(Path.cwd(), model_override=args.model)
    except ConfigError as e:
        ui.print_error(str(e))
        raise SystemExit(1) from e

    if args.sessions:
        ui.show_sessions(list_sessions(cfg))
        return

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        # No positional prompt, and stdin isn't a terminal: something's
        # piped in (`echo "..." | tcode`, or a caller passing the prompt as
        # subprocess `input=` — Claude Code's own `-p` accepts a prompt
        # either way, and brain_call.py-shaped Python callers already use
        # `input=` uniformly for every provider they invoke). Falling
        # through to the interactive REPL here would hang forever reading
        # from a pipe that's never going to send REPL commands.
        prompt = sys.stdin.read().strip()

    if prompt and args.verify:
        asyncio.run(verify_mode(cfg, prompt))
        return

    try:
        write_path = _resolve_write_path(cfg, args.write) if args.write else None
    except ConfigError as e:
        ui.print_error(str(e))
        raise SystemExit(1) from e

    message_history = load_latest_session(cfg) if args.cont else []

    if prompt and args.reduce:
        asyncio.run(
            reduce_mode(cfg, prompt, args.reduce, message_history, quiet=args.quiet, write_path=write_path)
        )
        return

    if prompt:
        asyncio.run(one_shot(cfg, prompt, message_history, quiet=args.quiet, write_path=write_path))
    else:
        asyncio.run(interactive(cfg, message_history))


if __name__ == "__main__":
    main()
