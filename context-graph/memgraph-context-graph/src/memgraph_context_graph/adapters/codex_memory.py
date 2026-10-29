"""Codex's built-in memories, as Context Graph switches away from and imports them.

Codex keeps one global memory store in ``$CODEX_HOME/memories/`` (default
``~/.codex``), written by its own background pipeline. Only
``memory_summary.md`` is injected into a session; the model opens
``MEMORY.md`` and the rest on demand. Memories are injected only while both
``[features] memories`` (off by default) and ``[memories] use_memories`` are
on, and ``generate_memories`` controls whether new threads feed the pipeline.

Switching to Context Graph turns all three off in the user-level
``config.toml``, the same scope ``[memory] backend`` has, after copying the file
to its own memory-switch backup. Handing memory back restores whatever they
were before from that backup, then removes it, after leaving the graph's memories as
ad-hoc notes: the one place Codex takes outside writes, which its
consolidation pass folds into the files it injects. The plugin's MCP server allows every tool
unless the user's own ``[mcp_servers.context-graph]`` narrows it with
``enabled_tools``; then ``memory`` is added there too.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memgraph_context_graph.memory_backend import ExportReport, ImportReport, MemoryBackendError

if TYPE_CHECKING:
    from memgraph_context_graph.memory_backend import MemoryFiles, MemorySource

_BACKUP_SUFFIX = ".context-graph.bak"
#: The config as it was before the memory switch, apart from the backup any write leaves: setup writes
#: ``config.toml`` too, and its older backup must never be taken for the state the switch should restore.
_SWITCH_BACKUP_SUFFIX = ".context-graph-memory.bak"
_SERVER = "context-graph"
#: (table, key) pairs this switch owns, and the value that turns each off.
_SWITCHES = (
    (("features", "memories"), False),
    (("memories", "use_memories"), False),
    (("memories", "generate_memories"), False),
)
#: Files a Codex memory store may hold, under v1's ``memories/`` or v2's ``memories_v2/``,
#: and where each lands in ``/memories``. Codex's store is global, so nothing is per project, and both stay
#: under ``/memories/codex/``: a summary written across every project isn't one the user wrote for all of them.
_IMPORTS = (
    ("memory_summary.md", "/memories/codex/memory-summary.md", "What Codex had learned about the user", "reference"),
    ("MEMORY.md", "/memories/codex/memory.md", "Codex's task-by-task memory, as it consolidated it", "reference"),
)
#: Codex's note-file contract (``ext/memories/src/local/ad_hoc_note.rs``): ``YYYY-MM-DDTHH-MM-SS-<slug>.md``,
#: a slug of 1 to 80 lowercase letters, digits and hyphens.
_NOTES_DIR = ("extensions", "ad_hoc", "notes")
_NOTE_SLUG_MAX = 80
_NOTE_PREFIX = "context-graph-"
_SLUG_UNSAFE_RE = re.compile(r"[^a-z0-9]+")


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

        The config is first copied to the switch's own backup, unless one is there from a switch not yet
        handed back: then the memory keys are already off, and the backup still holds what they were.

        Raises:
            MemoryBackendError: the config file isn't valid TOML.
        """
        path = _config_path(home)
        config = _read(path)
        switch_backup = _switch_backup(path)
        if not switch_backup.exists():
            switch_backup.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                shutil.copy2(path, switch_backup)
            else:
                switch_backup.touch()
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
        """Restore the memory keys to what they were before :meth:`disable` first ran, then drop its backup.

        Keys that didn't exist then are removed, which leaves Codex's own
        defaults (``features.memories`` is off by default), and so is a table
        the switch created and left empty.

        Raises:
            MemoryBackendError: the config file or its backup isn't valid TOML.
        """
        path = _config_path(home)
        switch_backup = _switch_backup(path)
        before = _read(switch_backup)
        config = _read(path)
        for keys, _off in _SWITCHES:
            previous = _get(before, keys, None)
            if previous is None:
                _remove(config, keys)
            else:
                _set(config, keys, previous)
        created = [
            t for t in {keys[0] for keys, _off in _SWITCHES} if t in config and not config[t] and t not in before
        ]
        for table in created:
            del config[table]
        lines = _write(path, config, trim_end=bool(created)) if path.exists() or config else []
        switch_backup.unlink(missing_ok=True)
        return lines

    def import_into(self, files: MemoryFiles, home: Path) -> ImportReport:
        """Copy Codex's consolidated memories into ``/memories``, never overwriting a path.

        The summary Codex injected each session and its task-group
        ``MEMORY.md`` become ``reference`` files under ``/memories/codex/``,
        which the model can open.
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

    def export_from(self, source: MemorySource, home: Path) -> ExportReport:
        """Leave each of the graph's memories as an ad-hoc note in Codex's store, for its pipeline to fold in.

        Codex's injected files are its consolidation pass's output and are rewritten by it, so the graph's
        memories go where Codex takes outside writes: one note each under ``extensions/ad_hoc/notes/``, which
        the next pass consolidates. A note is never overwritten, and one already left with the same text is
        not left again. Files the import took from Codex and that the graph never changed aren't noted, since
        Codex still has them.
        """
        notes = _memory_root(home).joinpath(*_NOTES_DIR)
        left = {path.read_text(encoding="utf-8") for path in notes.glob(f"*-{_NOTE_PREFIX}*.md")}
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        report = ExportReport()
        for memory in source.files():
            if _still_in_codex(memory.path, memory.content, home):
                continue
            note = _note(memory.path, memory.content)
            if note not in left:
                notes.mkdir(parents=True, exist_ok=True)
                _create_note(notes, stamp, _slug(memory.path), note)
                left.add(note)
            report.exported.append(memory.path)
        if report.exported:
            report.notes.append(_when_folded_in(notes, home))
        return report


def allow_plugin_tools(home: Path, plugin_id: str) -> list[str]:
    """Let the plugin's MCP tools run without a prompt, unless the user already set a mode for them.

    Codex asks before every MCP tool call, and under ``approval_policy = "never"`` an ask is a refusal, so
    ``recall`` would fail for exactly the users who turned prompts off. The tools are read-only over the user's
    own memory (``memory`` writes only ``/memories`` files), which is what ``approve`` is for.

    Raises:
        MemoryBackendError: the config file isn't valid TOML.
    """
    path = _config_path(home)
    config = _read(path)
    server = _plugin_server(config, plugin_id, create=True)
    if "default_tools_approval_mode" in server:
        return [f"Unchanged {path}"]
    server["default_tools_approval_mode"] = "approve"
    return _write(path, config)


def disallow_plugin_tools(home: Path, plugin_id: str) -> list[str]:
    """Drop what :func:`allow_plugin_tools` wrote, leaving the rest of the plugin's table as the user has it."""
    path = _config_path(home)
    if not path.exists():
        return []
    config = _read(path)
    server = _plugin_server(config, plugin_id, create=False)
    if server is None or server.get("default_tools_approval_mode") != "approve":
        return []
    del server["default_tools_approval_mode"]
    plugin = config["plugins"][plugin_id]
    if not server:
        del plugin["mcp_servers"][_SERVER]
    if not plugin["mcp_servers"]:
        del plugin["mcp_servers"]
    if not plugin:
        del config["plugins"][plugin_id]
    return _write(path, config)


def _plugin_server(config: Any, plugin_id: str, *, create: bool) -> Any:
    import tomlkit

    node = config
    for key in ("plugins", plugin_id, "mcp_servers", _SERVER):
        if key not in node:
            if not create:
                return None
            node[key] = tomlkit.table(is_super_table=key != _SERVER)
        node = node[key]
    return node


def _memory_root(home: Path) -> Path:
    """The store of the memory version Codex runs: ``memories/`` for v1, its default, or ``memories_v2/``."""
    version = _get(_read(_config_path(home)), ("memories", "version"), "v1")
    return _codex_home(home) / ("memories_v2" if version == "v2" else "memories")


def _still_in_codex(path: str, text: str, home: Path) -> bool:
    """Whether *path* is one :meth:`CodexMemory.import_into` copied, unchanged from the file Codex still holds."""
    codex_home = _codex_home(home)
    for store in ("memories", "memories_v2"):
        for source_name, target, _description, _type in _IMPORTS:
            if store == "memories_v2":
                target = target.replace(".md", "-v2.md")
            source = codex_home / store / source_name
            if path == target and source.is_file():
                original = source.read_text(encoding="utf-8")
                return bool(original.strip()) and text.endswith(original)
    return False


def _note(path: str, text: str) -> str:
    parts = path.removeprefix("/memories/").split("/")
    if len(parts) > 2 and parts[0] == "projects":
        scope = f"It applies only in the project `{parts[1]}` (its git remote, or else its folder name)."
    else:
        scope = "It applies to the user everywhere."
    return (
        f"Remember this. The user kept it in Context Graph's memory as `{path}` while that replaced Codex's"
        f" memory, and has switched back to Codex's. {scope}\n\n{text.rstrip()}\n"
    )


def _slug(path: str) -> str:
    stem = _SLUG_UNSAFE_RE.sub("-", path.removeprefix("/memories/").removesuffix(".md").lower()).strip("-")
    return (_NOTE_PREFIX + stem)[:_NOTE_SLUG_MAX].rstrip("-")


def _create_note(notes: Path, stamp: str, slug: str, text: str) -> None:
    """Write a new note, numbering its slug when the name is taken; Codex never overwrites one either."""
    for attempt in range(1000):
        suffix = f"-{attempt}" if attempt else ""
        name = f"{stamp}-{slug[: _NOTE_SLUG_MAX - len(suffix)]}{suffix}.md"
        try:
            with (notes / name).open("x", encoding="utf-8", newline="") as file:
                file.write(text)
        except FileExistsError:
            continue
        return
    raise MemoryBackendError(f"Couldn't name a note for {slug} in {notes}; remove old ones and re-run.")


def _when_folded_in(notes: Path, home: Path) -> str:
    """When Codex takes the notes in: its next consolidation, which runs only while its memories are on again."""
    path = _config_path(home)
    before = _read(_switch_backup(path))
    on = (
        bool(_get(before, ("features", "memories"), False))
        and _get(before, ("memories", "generate_memories"), True) is not False
    )
    if on:
        return f"Left them as notes in {notes}; Codex folds them into its memory at its next consolidation."
    return (
        f"Left them as notes in {notes}; Codex folds them in only once its memories are on"
        f" (features.memories and memories.generate_memories in {path}), which they weren't before."
    )


def _switch_backup(path: Path) -> Path:
    return path.with_name(path.name + _SWITCH_BACKUP_SUFFIX)


def _read_text(path: Path) -> str:
    """The file's text with its own line endings, which a default ``read_text`` would turn into ``\\n``."""
    with path.open(encoding="utf-8", newline="") as file:
        return file.read()


def _write_text(path: Path, text: str) -> None:
    """Write *text* as given, so a file keeps the line endings it was read with; Windows would add ``\\r``."""
    path.write_text(text, encoding="utf-8", newline="")


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
        return tomlkit.parse(_read_text(path))
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


def _write(path: Path, config: Any, *, trim_end: bool = False) -> list[str]:
    """Write *config* to *path*, backing up the first version; *trim_end* drops the blank lines a removed table left."""
    import tomlkit

    current = _read_text(path) if path.exists() else None
    rendered = tomlkit.dumps(config)
    if trim_end and rendered.strip():
        rendered = rendered.rstrip() + "\n"
    if current is not None and "\r\n" in current:
        # tomlkit renders the keys it adds with "\n" even in a CRLF file; keep the file's one style.
        rendered = rendered.replace("\r\n", "\n").replace("\n", "\r\n")
    if current == rendered:
        return [f"Unchanged {path}"]
    backup = path.with_name(path.name + _BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(path, rendered)
    return [f"Wrote {path}"]


CODEX_MEMORY = CodexMemory()
