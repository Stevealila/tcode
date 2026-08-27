"""Optional OS-level sandbox: re-exec tcode inside bubblewrap (preferred) or
firejail so the agent's shell and file tools are confined to the workspace
and ``~/.tcode`` at the kernel level, not just by tcode's in-process
guardrails — see README's "Shell access" section.

This is the backstop for the caller who never sets ``TCODE_SHELL=0``: the
in-process ``scope_shell_exploration`` / ``scope_writes_to`` /
``ClearToolResults`` guardrails all live inside the same process the model
can talk into misbehaving, whereas a write outside the workspace under
bubblewrap fails in the kernel regardless of what the model was persuaded
to do.

Deliberately *not* a substitute for ``TCODE_SHELL=0`` on genuinely
untrusted content: network stays reachable (the model API needs it), so a
shell that can still make outbound requests can still exfiltrate. What
``--sandbox`` buys is write/containment of the filesystem, cheaply, for the
common case where the operator wants a seatbelt without thinking hard about
the threat model.

Linux only. On any other platform, or when neither sandbox tool is on
PATH, this is a no-op with a warning — the run continues unsandboxed rather
than refusing to start.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from . import ui
from .config import GLOBAL_DIR

# Set on the re-exec'd child so a second maybe_reexec() call (the child
# parsing the same argv) doesn't recurse forever. Also the honest signal
# for "is this session actually sandboxed" — see config.py's `sandboxed`.
ACTIVE_ENV = "TCODE_SANDBOX_ACTIVE"


def sandbox_requested(flag: bool) -> bool:
    """True if --sandbox was passed or TCODE_SANDBOX is set truthy."""
    if flag:
        return True
    return os.environ.get("TCODE_SANDBOX", "").strip().lower() in ("1", "true", "yes")


def _bwrap_command(tool: str, workspace: Path, argv: list[str]) -> list[str]:
    # Whole filesystem read-only, then punch writable holes for exactly the
    # two trees tcode legitimately writes: the workspace and ~/.tcode
    # (sessions, memory, rpm-throttle state). /tmp becomes a private tmpfs
    # so scratch writes there neither leak out nor see the host's /tmp.
    # PID/IPC/UTS namespaces are unshared; the network namespace is *not*
    # (the model API needs it) — see the module docstring.
    return [
        tool,
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--tmpfs", "/dev/shm",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--bind", str(workspace), str(workspace),
        "--bind", str(GLOBAL_DIR), str(GLOBAL_DIR),
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--",
        *argv,
    ]


def _firejail_command(tool: str, workspace: Path, argv: list[str]) -> list[str]:
    # Weaker and fiddlier than bubblewrap (firejail's default is not a
    # read-only root), but a real boundary and the only fallback the plan
    # names. Kept simple on purpose: private /tmp, home read-only, the two
    # trees tcode writes re-enabled.
    return [
        tool,
        "--quiet",
        "--noprofile",
        "--private-tmp",
        f"--read-only={Path.home()}",
        f"--read-write={GLOBAL_DIR}",
        f"--read-write={workspace}",
        "--",
        *argv,
    ]


def maybe_reexec(flag: bool, *, quiet: bool = False) -> None:
    """Re-exec the current process under a sandbox if one was requested and
    is available; otherwise return (a warning is printed for the cases
    where a sandbox was asked for but can't be provided).

    Call this as early in ``main()`` as possible, before any real work —
    ``os.execvp`` replaces the process image, so anything done beforehand is
    thrown away.
    """
    if not sandbox_requested(flag):
        return
    if os.environ.get(ACTIVE_ENV) == "1":
        return  # already running inside the sandbox we spawned

    if sys.platform != "linux":
        ui.print_notice(
            f"--sandbox is Linux-only (bubblewrap/firejail); continuing "
            f"unsandboxed on {sys.platform}.",
            quiet=quiet,
        )
        return

    bwrap = shutil.which("bwrap")
    firejail = shutil.which("firejail")
    tool = bwrap or firejail
    if tool is None:
        ui.print_notice(
            "--sandbox needs bubblewrap (bwrap) or firejail on PATH; neither "
            "was found, continuing unsandboxed. Install one, or use "
            "TCODE_SHELL=0 for untrusted content.",
            quiet=quiet,
        )
        return

    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    workspace = Path.cwd().resolve()
    argv0 = shutil.which(sys.argv[0]) or sys.argv[0]
    inner = [argv0, *sys.argv[1:]]
    cmd = (
        _bwrap_command(tool, workspace, inner)
        if tool == bwrap
        else _firejail_command(tool, workspace, inner)
    )

    os.environ[ACTIVE_ENV] = "1"
    ui.print_notice(
        f"--sandbox: re-exec under {Path(tool).name} — {workspace} and "
        f"{GLOBAL_DIR} writable, the rest of the filesystem read-only, "
        "network still reachable.",
        quiet=quiet,
    )
    os.execvp(cmd[0], cmd)
