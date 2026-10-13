"""The packages the ``memgraph-context-graph`` uv tool needs beside itself, and installing the missing ones."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

#: Kept in step with the plugins' bootstrap scripts and install.sh, which install the same set first.
BASE = "memgraph-context-graph[mcp]>=0.6.0"
CONNECTORS = {
    "skills-graph": "skills-graph[memgraph-context-graph]>=0.2.1",
    "actions-graph": "actions-graph[memgraph-context-graph]>=0.5.0",
    "sessions-graph": "sessions-graph[memgraph-context-graph]>=0.8.1",
}
VIEWER = ("context-graph-viewer", "context-graph-viewer>=0.1.0")


class ToolError(Exception):
    """``uv tool install`` failed; the message carries its last line."""


def installed(package: str) -> bool:
    """Whether *package* is installed in this Python."""
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        return False
    return True


def missing(*, viewer: bool) -> list[str]:
    """The packages setup wants that aren't installed here, by distribution name."""
    wanted = [*CONNECTORS, *([VIEWER[0]] if viewer else [])]
    return [package for package in wanted if not installed(package)]


def uv_tool_environment() -> bool:
    """Whether this Python is the ``memgraph-context-graph`` uv tool's own environment.

    Setup only reinstalls into an environment uv manages as that tool; a development venv is the
    developer's to manage.
    """
    uv = shutil.which("uv")
    if uv is None:
        return False
    result = subprocess.run([uv, "tool", "dir"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False
    tool_dir = Path(result.stdout.strip()) / "memgraph-context-graph"
    return Path(sys.prefix).resolve() == tool_dir.resolve()


def install(*, viewer: bool, log: Callable[[str], None]) -> None:
    """Install the tool with every connector (and the viewer), keeping what is already there.

    The full set is always passed: ``uv tool install`` replaces the tool's requirements with what it is
    given, so naming only the missing package would drop the others.

    Raises:
        ToolError: uv isn't on PATH or the install failed.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise ToolError("uv isn't on PATH")
    argv = [uv, "tool", "install", BASE]
    for requirement in CONNECTORS.values():
        argv += ["--with", requirement]
    if viewer or installed(VIEWER[0]):
        argv += ["--with", VIEWER[1]]
    log(" ".join(["uv", *argv[1:]]))
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    assert process.stdout is not None
    for line in process.stdout:
        if line.strip():
            last = line.rstrip()
            log(last)
    if process.wait() != 0:
        raise ToolError(f"uv tool install failed: {last or f'exit {process.returncode}'}")
    importlib.invalidate_caches()
