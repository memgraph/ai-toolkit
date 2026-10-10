"""Codex's built-in memories, as Context Graph switches away from and imports them.

Codex keeps one global memory store in ``$CODEX_HOME/memories/`` (default
``~/.codex``), written by its own background pipeline. Only
``memory_summary.md`` is injected into a session; the model opens
``MEMORY.md`` and the rest on demand. Memories are injected only while both
``[features] memories`` (off by default) and ``[memories] use_memories`` are
on, and ``generate_memories`` controls whether new threads feed the pipeline.

Switching to Context Graph turns all three off in the user-level
``config.toml``, the same scope ``[memory] backend`` has. Handing memory back
restores whatever they were before. The plugin's MCP server allows every tool
unless the user's own ``[mcp_servers.context-graph]`` narrows it with
``enabled_tools``; then ``memory`` is added there too.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_context_graph.memory_backend import ImportReport, MemoryBackendError

if TYPE_CHECKING:
    from agent_context_graph.memory_backend import MemoryFiles

_BACKUP_SUFFIX = ".context-graph.bak"
_SERVER = "context-graph"
#: (table, key) pairs this switch owns, and the value that turns each off.
_SWITCHES = (
    (("features", "memories"), False),
    (("memories", "use_memories"), False),
    (("memories", "generate_memories"), False),
)
#: Files a Codex memory store may hold, under v1's ``memories/`` or v2's ``memories_v2/``,
#: and where each lands in ``/memories``. Codex's store is global, so nothing is per project.
_IMPORTS = (
    ("memory_summary.md", "/memories/user/codex-memory-summary.md", "What Codex had learned about the user", "user"),
    ("MEMORY.md", "/memories/codex/memory.md", "Codex's task-by-task memory, as it consolidated it", "reference"),
)


@dataclass(frozen=True)
class CodexMemory:
    """Codex's built-in memories (``NativeMemory``)."""

    def enabled(self, home: Path) -> bool:
        """Whether Codex would inject its memories: ``features.memories`` on and ``use_memories`` not off."""
        config = _read(_config_path(home))
        return (
            bool(_get(config, ("features", "memories"), False))
            and _get(config, ("memories", "use_memories"), True) is not False
        )

    def disable(self, home: Path) -> list[str]:
        """Turn Codex's memories off in its user ``config.toml`` and let the ``memory`` tool through.

        Raises:
            MemoryBackendError: the config file isn't valid TOML.
        """
        path = _config_path(home)
        config = _read(path)
        for keys, off in _SWITCHES:
            _set(config, keys, off)
        server = config.get("mcp_servers", {}).get(_SERVER)
        if (
            server is not None
            and isinstance(server.get("enabled_tools"), list)
            and "memory" not in server["enabled_tools"]
        ):
            server["enabled_tools"].append("memory")
        return _write(path, config)

    def enable(self, home: Path) -> list[str]:
        """Restore the memory keys to what they were before :meth:`disable` first ran.

        Keys that didn't exist then are removed, which leaves Codex's own
        defaults (``features.memories`` is off by default).

        Raises:
            MemoryBackendError: the config file or its backup isn't valid TOML.
        """
        path = _config_path(home)
        backup = path.with_name(path.name + _BACKUP_SUFFIX)
        before = _read(backup) if backup.exists() else {}
        config = _read(path)
        for keys, _off in _SWITCHES:
            previous = _get(before, keys, None)
            if previous is None:
                _remove(config, keys)
            else:
                _set(config, keys, previous)
        return _write(path, config)

    def import_into(self, files: MemoryFiles, home: Path) -> ImportReport:
        """Copy Codex's consolidated memories into ``/memories``, never overwriting a path.

        The summary Codex injected each session becomes a ``user`` memory;
        its task-group ``MEMORY.md`` a ``reference`` file the model can open.
        Rollout summaries and raw memories are Codex's intermediate state and
        stay where they are.
        """
        report = ImportReport()
        codex_home = _codex_home(home)
        for store in ("memories", "memories_v2"):
            for source_name, target, description, memory_type in _IMPORTS:
                source = codex_home / store / source_name
                if not source.is_file() or not (text := source.read_text(encoding="utf-8")).strip():
                    continue
                if store == "memories_v2":
                    target = target.replace(".md", "-v2.md")
                try:
                    files.view(target)
                except Exception:
                    try:
                        files.create(target, f"---\ndescription: {description}\ntype: {memory_type}\n---\n{text}")
                    except Exception:  # Too large for one memory file: leave it in Codex's store.
                        report.skipped.append(target)
                        continue
                    report.imported.append(target)
                    continue
                report.skipped.append(target)
        return report


def _codex_home(home: Path) -> Path:
    override = os.environ.get("CODEX_HOME")
    return Path(override) if override else home / ".codex"


def _config_path(home: Path) -> Path:
    return _codex_home(home) / "config.toml"


def _read(path: Path) -> Any:
    import tomlkit
    from tomlkit.exceptions import ParseError

    if not path.exists():
        return tomlkit.document()
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except ParseError as exc:
        raise MemoryBackendError(f"{path} is not valid TOML ({exc}); fix it, then re-run.") from exc


def _get(config: Any, keys: tuple[str, str], default: Any) -> Any:
    table = config.get(keys[0])
    return table.get(keys[1], default) if hasattr(table, "get") else default


def _set(config: Any, keys: tuple[str, str], value: Any) -> None:
    import tomlkit

    table = config.get(keys[0])
    if not hasattr(table, "get"):
        table = tomlkit.table()
        config[keys[0]] = table
    table[keys[1]] = value


def _remove(config: Any, keys: tuple[str, str]) -> None:
    table = config.get(keys[0])
    if hasattr(table, "get") and keys[1] in table:
        del table[keys[1]]


def _write(path: Path, config: Any) -> list[str]:
    import tomlkit

    rendered = tomlkit.dumps(config)
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return [f"Unchanged {path}"]
    backup = path.with_name(path.name + _BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return [f"Wrote {path}"]


CODEX_MEMORY = CodexMemory()
