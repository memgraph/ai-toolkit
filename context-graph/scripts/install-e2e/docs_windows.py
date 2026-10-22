"""Execute the docs' Windows one-liners in real consoles on a disposable runner."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from winpty import PtyProcess

ROOT = Path(os.environ["RUNNER_TEMP"]) / "docs-1847-evidence"
ROOT.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
SHELL = sys.argv[1]
if SHELL.casefold() == "powershell":
    # Python inherits PowerShell 7's module paths from the Actions shell; Windows PowerShell needs its own.
    os.environ["PSMODULEPATH"] = ";".join(
        str(path)
        for path in [
            Path(os.environ["WINDIR"]) / "System32/WindowsPowerShell/v1.0/Modules",
            Path(os.environ["PROGRAMFILES"]) / "WindowsPowerShell/Modules",
            Path(os.environ["USERPROFILE"]) / "Documents/WindowsPowerShell/Modules",
        ]
    )
URL = "https://install.memgraph.com/ai-memory/windows"


def run_console(
    name: str, command: str, *, expected: str = "Context Graph is set up.", startup_policy: bool = True
) -> None:
    """Answer setup's default questions while preserving the actual console output."""
    argv = [SHELL, "-NoProfile"]
    if startup_policy:
        argv += ["-ExecutionPolicy", "Bypass"]
    process = PtyProcess.spawn([*argv, "-Command", command], dimensions=(40, 160))
    chunks: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        while process.isalive():
            try:
                chunks.put(process.read(4096))
            except EOFError:
                break

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    output = ""
    answered: set[str] = set()
    deadline = time.monotonic() + 300
    questions = [
        "Your name",
        "Agents",
        "Projects",
        "Install the terminal viewer?",
        "Use Context Graph as the memory",
        "Install now?",
    ]
    try:
        while process.isalive() or not chunks.empty():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name}: installer did not finish in five minutes")
            try:
                output += chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
            if "[Y] Yes" in plain and "policy-confirm" not in answered:
                process.write("y\r")
                answered.add("policy-confirm")
            for question in questions:
                if question not in answered and re.search(r"◆[^\r\n]*" + re.escape(question), plain):
                    process.write("\r")
                    answered.add(question)
        reader.join(timeout=5)
        while not chunks.empty():
            output += chunks.get_nowait()
        (ROOT / f"{SHELL}-{name}.log").write_text(output, encoding="utf-8")
        print(output)
        assert expected in output, f"{name}: expected {expected!r} in command output"
        assert process.exitstatus == 0, f"{name}: exited {process.exitstatus}"
    finally:
        (ROOT / f"{SHELL}-{name}.log").write_text(output, encoding="utf-8")
        if process.isalive():
            process.terminate(force=True)


run_console(
    "execution-policy",
    "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; Write-Output ('DOCS_POLICY=' + (Get-ExecutionPolicy -Scope Process))",
    expected="DOCS_POLICY=Bypass",
    startup_policy=False,
)
if "--policy-only" in sys.argv:
    sys.exit(0)
run_console("first-one-liner", f"iwr {URL} -useb | iex")
run_console("setup-args-one-liner", f"$env:CONTEXT_GRAPH_SETUP_ARGS = 'codex --user-id alice'; iwr {URL} -useb | iex")
subprocess.run(
    [
        SHELL,
        "-NoProfile",
        "-Command",
        "$env:Path=[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User'); memgraph-context-graph config get identity.user_id; memgraph-context-graph doctor --runtime codex; exit $LASTEXITCODE",
    ],
    check=True,
)
