"""Tests for sandbox.py: request detection, command construction, and the
no-op guard paths of maybe_reexec (the actual os.execvp path can't be
exercised in-process). See betterment/plan.txt 3.5.c.
"""

from __future__ import annotations

from pathlib import Path

from src import sandbox
from src.sandbox import _bwrap_command, _firejail_command, maybe_reexec, sandbox_requested


class TestSandboxRequested:
    def test_flag_true(self, monkeypatch):
        monkeypatch.delenv("TCODE_SANDBOX", raising=False)
        assert sandbox_requested(True) is True

    def test_neither(self, monkeypatch):
        monkeypatch.delenv("TCODE_SANDBOX", raising=False)
        assert sandbox_requested(False) is False

    def test_env_truthy_values(self, monkeypatch):
        for v in ("1", "true", "YES", "Yes"):
            monkeypatch.setenv("TCODE_SANDBOX", v)
            assert sandbox_requested(False) is True

    def test_env_falsy_values(self, monkeypatch):
        for v in ("0", "false", "no", ""):
            monkeypatch.setenv("TCODE_SANDBOX", v)
            assert sandbox_requested(False) is False


class TestCommandConstruction:
    def test_bwrap_command_confines_writes_and_passes_argv(self):
        cmd = _bwrap_command("/usr/bin/bwrap", Path("/home/u/ws"), ["tcode", "--sessions"])
        assert cmd[0] == "/usr/bin/bwrap"
        assert cmd[-2:] == ["tcode", "--sessions"]
        # whole fs read-only, workspace + ~/.tcode punched back to writable
        assert "--ro-bind" in cmd and "/" in cmd
        joined = " ".join(cmd)
        assert "--bind /home/u/ws /home/u/ws" in joined
        assert str(sandbox.GLOBAL_DIR) in joined
        # network namespace deliberately NOT unshared
        assert "--unshare-net" not in cmd
        assert "--unshare-all" not in cmd

    def test_firejail_command_passes_argv(self):
        cmd = _firejail_command("/usr/bin/firejail", Path("/home/u/ws"), ["tcode", "x"])
        assert cmd[0] == "/usr/bin/firejail"
        assert cmd[-2:] == ["tcode", "x"]
        assert f"--read-write={Path('/home/u/ws')}" in cmd


class TestMaybeReexecNoOps:
    def test_no_op_when_not_requested(self, monkeypatch):
        monkeypatch.delenv("TCODE_SANDBOX", raising=False)
        called = []
        monkeypatch.setattr(sandbox.os, "execvp", lambda *a: called.append(a))
        maybe_reexec(False)
        assert called == []

    def test_no_op_when_already_active(self, monkeypatch):
        monkeypatch.setenv(sandbox.ACTIVE_ENV, "1")
        called = []
        monkeypatch.setattr(sandbox.os, "execvp", lambda *a: called.append(a))
        maybe_reexec(True)
        assert called == []

    def test_warns_and_continues_on_non_linux(self, monkeypatch):
        monkeypatch.delenv(sandbox.ACTIVE_ENV, raising=False)
        monkeypatch.setattr(sandbox.sys, "platform", "darwin")
        notices = []
        monkeypatch.setattr(sandbox.ui, "print_notice", lambda m, **k: notices.append(m))
        called = []
        monkeypatch.setattr(sandbox.os, "execvp", lambda *a: called.append(a))
        maybe_reexec(True)
        assert called == []
        assert any("Linux-only" in n for n in notices)

    def test_warns_and_continues_when_no_tool_installed(self, monkeypatch):
        monkeypatch.delenv(sandbox.ACTIVE_ENV, raising=False)
        monkeypatch.setattr(sandbox.sys, "platform", "linux")
        monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
        notices = []
        monkeypatch.setattr(sandbox.ui, "print_notice", lambda m, **k: notices.append(m))
        called = []
        monkeypatch.setattr(sandbox.os, "execvp", lambda *a: called.append(a))
        maybe_reexec(True)
        assert called == []
        assert any("bubblewrap" in n or "bwrap" in n for n in notices)
