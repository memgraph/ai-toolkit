"""#537: launch the real `memgraph-context-graph setup` TUI behind `cat shim | bash` in a pty, then press Escape.

Throwaway home, config and PATH (fake claude/codex only), so nothing real is touched. Run from the root of a
checkout of the setup branch, with its venv:

    .venv/bin/python <this file> .venv/bin/memgraph-context-graph

Expected: the plan screen draws in well under a second, Escape exits 0, no config is written and neither
agent CLI is called.
"""

import os
import pathlib
import pty
import re
import select
import sys
import tempfile
import time

CLI = pathlib.Path(sys.argv[1]).resolve()
FAKE = pathlib.Path("context-graph/memgraph-context-graph/tests/fake_agent_cli.py").resolve()

with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    (tmp / "home").mkdir()
    (tmp / "bin").mkdir()
    for agent in ("claude", "codex"):
        script = tmp / "bin" / agent
        script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" {agent} "$@"\n')
        script.chmod(0o755)
    project = tmp / "billing"
    project.mkdir()
    shim = tmp / "shim.sh"
    shim.write_text(
        f'if (exec </dev/tty) 2>/dev/null; then exec "{CLI}" setup "$@" </dev/tty; fi\nexec "{CLI}" setup --no-tui "$@"\n'
    )
    env = {
        "HOME": str(tmp / "home"),
        "PATH": f"{tmp / 'bin'}:/usr/bin:/bin",
        "CONTEXT_GRAPH_CONFIG": str(tmp / "config.toml"),
        "FAKE_AGENT_STATE": str(tmp / "state.json"),
        "TERM": "xterm-256color",
        "COLUMNS": "120",
        "LINES": "40",
    }
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(project)
        os.execve("/bin/bash", ["bash", "-c", f"cat {shim} | bash"], env)
    screen = b""
    start = time.monotonic()
    pressed = None
    status = None
    while time.monotonic() - start < 30:
        if select.select([fd], [], [], 0.1)[0]:
            try:
                screen += os.read(fd, 65536)
            except OSError:
                break
        if pressed is None and b"Install" in screen and b"Plan" in screen:
            pressed = time.monotonic() - start
            time.sleep(0.5)
            os.write(fd, b"\x1b")
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
    else:
        os.kill(pid, 9)
        print("never exited")
    text = re.sub(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][0-9A-Z]|\x1b[=>]", b" ", screen).decode(errors="replace")
    words = " ".join(text.split())
    print(
        f"plan screen drawn after {pressed:.1f}s; exit status {os.waitstatus_to_exitcode(status) if status is not None else '?'}"
    )
    for needle in (
        "Context Graph setup",
        "Claude Code",
        "Codex",
        "Cursor",
        "not found",
        "Install",
        "Identity",
        str(project),
    ):
        print(f"  {'found' if needle in words else 'MISSING'}: {needle}")
    print("config written:", (tmp / "config.toml").exists(), "| agent calls:", (tmp / "state.json").exists())
