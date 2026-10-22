"""The built-in memories of the project-wired agents, as Context Graph switches away from them.

- **Copilot CLI** keeps "Copilot Memory" on GitHub's servers, injected into
  the system prompt and written by its ``store_memory`` tool. ``"memory":
  false`` in ``~/.copilot/settings.json`` turns it off. It can't be read
  locally, so nothing is imported.
- **Grok Build** keeps Markdown topic notes under ``~/.grok/memory-v2/`` (one
  global folder, one per workspace), on by default through a managed remote
  setting. ``[memory_v2] enabled = false`` in the user's ``~/.grok/config.toml``
  overrides that; it is the only scope Grok reads memory keys from.
- **Cursor** removed its Memories feature, and **Antigravity CLI** and
  **OpenCode** have none, so switching them only turns the ``memory`` tool on.

Each object has the shape :class:`~memgraph_context_graph.memory_backend.NativeMemory` describes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memgraph_context_graph.adapters.agent_mcp import _read_json, _read_toml, _write_json, _write_toml
from memgraph_context_graph.memory_backend import ExportReport, ImportReport, keep_in_graph

if TYPE_CHECKING:
    from pathlib import Path

    from memgraph_context_graph.memory_backend import MemoryFiles, MemorySource

_BACKUP_SUFFIX = ".context-graph.bak"
_HEADING_RE = re.compile(r"^#\s+(?P<title>.+)$", re.MULTILINE)
#: Grok names a workspace folder ``<slug>-<hash8>``; the hash isn't derivable from the checkout's path.
_WORKSPACE_HASH_RE = re.compile(r"-[0-9a-f]{8}$")


@dataclass(frozen=True)
class CopilotMemory:
    """Copilot CLI's Copilot Memory (``NativeMemory``)."""

    def enabled(self, home: Path) -> bool:
        """On unless the user-level ``memory`` setting is ``false`` (Copilot's default is on)."""
        return _read_json(_copilot_settings(home)).get("memory") is not False

    def disable(self, home: Path) -> list[str]:
        """Set ``"memory": false`` in ``~/.copilot/settings.json``.

        Raises:
            MemoryBackendError: the settings file isn't a JSON object.
        """
        return _set_copilot_memory(home, False)

    def enable(self, home: Path) -> list[str]:
        """Restore ``memory`` to what it was before :meth:`disable`: removed, unless the user had set it."""
        backup = _copilot_settings(home).with_name("settings.json" + _BACKUP_SUFFIX)
        before = _read_json(backup).get("memory") if backup.exists() else None
        return _set_copilot_memory(home, before)

    def import_into(self, files: MemoryFiles, home: Path) -> ImportReport:
        """Nothing to import: Copilot Memory lives on GitHub, readable only in its settings page."""
        return ImportReport()

    def export_from(self, source: MemorySource, home: Path) -> ExportReport:
        """Nothing written back: Copilot Memory lives on GitHub, writable only by Copilot's own tool."""
        return keep_in_graph(source)


@dataclass(frozen=True)
class GrokMemory:
    """Grok Build's memory (``NativeMemory``)."""

    def enabled(self, home: Path) -> bool:
        """On unless the user config turns ``memory_v2`` off; Grok's managed default is on."""
        return _read_toml(_grok_config(home)).get("memory_v2", {}).get("enabled", True) is not False

    def disable(self, home: Path) -> list[str]:
        """Set ``[memory_v2] enabled = false`` in ``~/.grok/config.toml``.

        Raises:
            MemoryBackendError: the config isn't valid TOML.
        """
        import tomlkit

        path = _grok_config(home)
        config = _read_toml(path)
        if "memory_v2" not in config:
            config["memory_v2"] = tomlkit.table()
        config["memory_v2"]["enabled"] = False
        return _write_toml(path, config)

    def enable(self, home: Path) -> list[str]:
        """Restore ``memory_v2.enabled`` to its value before :meth:`disable`, or Grok's default.

        Raises:
            MemoryBackendError: the config or its backup isn't valid TOML.
        """
        path = _grok_config(home)
        if not path.exists():
            return []
        backup = path.with_name(path.name + _BACKUP_SUFFIX)
        before = _read_toml(backup).get("memory_v2", {}).get("enabled") if backup.exists() else None
        config = _read_toml(path)
        table = config.get("memory_v2")
        if table is None:
            return [f"Unchanged {path}"]
        if before is None:
            table.pop("enabled", None)
            if not table:
                del config["memory_v2"]
        else:
            table["enabled"] = before
        return _write_toml(path, config)

    def import_into(self, files: MemoryFiles, home: Path) -> ImportReport:
        """Copy Grok's topic notes into ``/memories/grok/``, never overwriting a path.

        Global topics land in ``/memories/grok/``, each workspace's in
        ``/memories/grok/<workspace>/``: Grok's workspace folder can't be traced
        back to a checkout, so they can't go into a project folder. Unconsolidated
        observations and Grok's generated ``MEMORY.md`` index stay behind.
        """
        report = ImportReport()
        root = home / ".grok" / "memory-v2"
        sources = [(root / "global" / "topics", "/memories/grok")]
        for workspace in sorted((root / "workspaces").glob("*/")):
            sources.append((workspace / "topics", f"/memories/grok/{_WORKSPACE_HASH_RE.sub('', workspace.name)}"))
        for folder, target_dir in sources:
            for source in sorted(folder.glob("*.md")):
                text = source.read_text(encoding="utf-8")
                if not text.strip():
                    continue
                target = f"{target_dir}/{source.name}"
                if _exists(files, target):
                    report.skipped.append(target)
                    continue
                try:
                    files.create(target, _with_frontmatter(text))
                except Exception:  # Too large, or the path is taken by a directory: keep going.
                    report.skipped.append(target)
                    continue
                report.imported.append(target)
        return report

    def export_from(self, source: MemorySource, home: Path) -> ExportReport:
        """Nothing written back: Grok names each workspace's folder by a hash only Grok can derive."""
        return keep_in_graph(source)


@dataclass(frozen=True)
class NoBuiltInMemory:
    """An agent with no memory of its own: switching only turns the ``memory`` tool on (``NativeMemory``)."""

    def enabled(self, home: Path) -> bool:
        """Never on: there is nothing to turn off."""
        return False

    def disable(self, home: Path) -> list[str]:
        """Nothing to turn off."""
        return []

    def enable(self, home: Path) -> list[str]:
        """Nothing to turn back on."""
        return []

    def import_into(self, files: MemoryFiles, home: Path) -> ImportReport:
        """Nothing to import."""
        return ImportReport()

    def export_from(self, source: MemorySource, home: Path) -> ExportReport:
        """Nothing to write back into, and nothing to report: the agent never had a memory to hand back."""
        return ExportReport()


def _copilot_settings(home: Path) -> Path:
    return home / ".copilot" / "settings.json"


def _set_copilot_memory(home: Path, value: bool | None) -> list[str]:
    path = _copilot_settings(home)
    settings = _read_json(path)
    if settings.get("memory") == value or (value is None and "memory" not in settings):
        return [f"Unchanged {path}"]
    if value is None:
        settings.pop("memory")
    else:
        settings["memory"] = value
    return _write_json(path, settings)


def _grok_config(home: Path) -> Path:
    return home / ".grok" / "config.toml"


def _exists(files: MemoryFiles, path: str) -> bool:
    try:
        files.view(path)
    except Exception:
        return False
    return True


def _with_frontmatter(text: str) -> str:
    """Grok's note, headed with the frontmatter the session-start index reads its description from."""
    if text.startswith("---"):
        return text
    match = _HEADING_RE.search(text)
    description = match["title"].strip() if match else "Imported from Grok Build's memory"
    return f"---\ndescription: {description}\ntype: reference\n---\n{text}"


COPILOT_MEMORY = CopilotMemory()
GROK_MEMORY = GrokMemory()
NO_BUILT_IN_MEMORY = NoBuiltInMemory()
