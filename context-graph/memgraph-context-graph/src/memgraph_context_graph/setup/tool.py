"""The packages the ``memgraph-context-graph`` uv tool needs beside itself, and installing the missing ones."""

from __future__ import annotations

import importlib
import shlex
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from memgraph_context_graph.connectors import INSTALLABLE

if TYPE_CHECKING:
    from collections.abc import Callable

#: Kept in step with the plugins' bootstrap scripts and install.sh, which install the default connectors first.
BASE = "memgraph-context-graph[mcp]>=0.11.0"
VIEWER = ("context-graph-viewer", "context-graph-viewer>=0.1.1")
SPACES = ("context-graph-spaces", "context-graph-spaces>=0.1.2")
#: sessions-graph's local extraction, which setup doesn't install itself; ``unstructured2graph`` tells it is there.
RECONCILIATION = ("unstructured2graph", "sessions-graph[memgraph-context-graph,reconciliation]>=0.11.1")
#: Connectors whose own command people run: ``sessions-graph reconcile``, ``resources-graph sweep``.
EXECUTABLES = ("sessions-graph", "resources-graph")


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
    wanted = [*connectors(), *([VIEWER[0]] if viewer else [])]
    return [package for package in wanted if not installed(package)]


def uv_tool_environment() -> bool:
    """Whether this Python is the ``memgraph-context-graph`` uv tool's own environment.

    Setup only reinstalls into an environment uv manages as that tool; a development venv is the
    developer's to manage.
    """
    uv = shutil.which("uv")
    if uv is None:
        return False
    # --color never: FORCE_COLOR makes uv colour even captured output, and the path would no longer compare equal.
    result = subprocess.run([uv, "--color", "never", "tool", "dir"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False
    tool_dir = Path(result.stdout.strip()) / "memgraph-context-graph"
    return Path(sys.prefix).resolve() == tool_dir.resolve()


def connectors() -> list[str]:
    """The connectors setup installs: the configured ones (``[capture] connectors``), plus any already installed."""
    from memgraph_context_graph.connectors import configured_connectors

    configured = configured_connectors()
    return [name for name in INSTALLABLE if name in configured or installed(name)]


def requirements(*, viewer: bool = False, spaces: bool = False) -> list[str]:
    """The ``uv tool install`` arguments for the tool, its connectors (:func:`connectors`), and the optional packages.

    A package already installed is kept, and so is sessions-graph's reconciliation extra: ``uv tool
    install`` replaces the tool's requirements with what it is given, so leaving one out would remove
    it. The connectors' own commands (:data:`EXECUTABLES`) are put on PATH, since uv would drop them too.
    """
    argv = [BASE]
    names = connectors()
    for name in names:
        requirement = INSTALLABLE[name]
        if name == "sessions-graph" and installed(RECONCILIATION[0]):
            requirement = RECONCILIATION[1]
        argv += ["--with", requirement]
    if viewer or installed(VIEWER[0]):
        argv += ["--with", VIEWER[1]]
    if spaces or installed(SPACES[0]):
        argv += ["--with", SPACES[1]]
    for name in EXECUTABLES:
        if name in names:
            argv += ["--with-executables-from", name]
    return argv


def install_command(*, spaces: bool = False) -> str:
    """The whole ``uv tool install`` command as one line a person can paste, with *spaces* added."""
    return " ".join(["uv tool install", *map(shlex.quote, requirements(spaces=spaces))])


def install(*, viewer: bool, log: Callable[[str], None], spaces: bool = False, upgrade: bool = False) -> None:
    """Install the tool with its connectors (and the viewer, and the spaces package), keeping what is already there.

    With *upgrade*, the tool and its connectors move to the newest releases the requirements allow.

    The full set is always passed: ``uv tool install`` replaces the tool's requirements with what it is
    given, so naming only the missing package would drop the others.

    Raises:
        ToolError: uv isn't on PATH or the install failed.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise ToolError("uv isn't on PATH")
    # The tool's own Python version: left to choose, uv takes the first Python it finds, which may be too old.
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    argv = [uv, "--color", "never", "tool", "install", "--python", python, *requirements(viewer=viewer, spaces=spaces)]
    if upgrade:
        argv.append("--upgrade")
        # Bypass the cached index entries, so a release published minutes ago is found.
        for package in ("memgraph-context-graph", *connectors()):
            argv += ["--refresh-package", package]
    log(" ".join(["uv", *argv[3:]]))
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
