"""Claude Code's auto memory, as Context Graph switches away from and imports it.

Auto memory is Markdown files in ``~/.claude/projects/<project>/memory/``, one
folder per project. ``<project>`` is the project's absolute path with every
character other than a letter or digit replaced by ``-``. ``MEMORY.md`` there
is the index Claude Code loads at session start; the other files are
memories, each with frontmatter naming its ``type``. The ``autoMemoryEnabled``
setting turns it off; a plugin can't set it, so this edits the user's own
``~/.claude/settings.json``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_context_graph.memory_backend import ImportReport, MemoryBackendError

if TYPE_CHECKING:
    from agent_context_graph.memory_backend import MemoryFiles

_SETTING = "autoMemoryEnabled"
_INDEX_FILE = "MEMORY.md"
_BACKUP_SUFFIX = ".context-graph.bak"
_TYPE_RE = re.compile(r"^\s*type:\s*[\"']?(?P<type>[A-Za-z]+)", re.MULTILINE)
_PREFIX_RE = re.compile(r"^(user|feedback|project|reference)_")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]")
#: Types about the user, which apply in every project; the rest are about one project.
_USER_WIDE_TYPES = frozenset({"user", "feedback"})


@dataclass(frozen=True)
class ClaudeCodeMemory:
    """Claude Code's auto memory (``NativeMemory``)."""

    def enabled(self, home: Path) -> bool:
        """On unless a user-level setting turns it off."""
        settings = _read_settings(_settings_path(home))
        return settings.get(_SETTING) is not False

    def disable(self, home: Path) -> list[str]:
        """Set ``autoMemoryEnabled: false`` in ``~/.claude/settings.json``, keeping every other setting.

        Raises:
            MemoryBackendError: the settings file isn't a JSON object.
        """
        return _set(_settings_path(home), False)

    def enable(self, home: Path) -> list[str]:
        """Drop the ``autoMemoryEnabled: false`` that :meth:`disable` wrote.

        Raises:
            MemoryBackendError: the settings file isn't a JSON object.
        """
        return _set(_settings_path(home), None)

    def import_into(self, files: MemoryFiles, home: Path) -> ImportReport:
        """Copy every project's auto-memory files into the graph's ``/memories`` tree.

        ``user``/``feedback`` memories go under ``/memories/<type>/``, since they
        apply everywhere; ``project``/``reference`` (and untyped) ones under
        that project's ``/memories/projects/<key>/``, keyed as the session-start
        index keys the checkout. ``MEMORY.md`` is skipped: the index is
        generated now. The original files are left as they are.
        """
        from sessions_graph.memory_index import project_key_for

        report = ImportReport()
        for memory_dir in sorted((home / ".claude" / "projects").glob("*/memory")):
            checkout = _checkout_for(memory_dir.parent.name)
            project = (project_key_for(checkout) if checkout else None) or _fallback_key(memory_dir.parent.name)
            for source in sorted(memory_dir.glob("*.md")):
                if source.name == _INDEX_FILE:
                    continue
                text = source.read_text(encoding="utf-8")
                if not text.strip():
                    continue
                target = _target_path(source.name, text, project)
                if _exists(files, target):
                    report.skipped.append(target)
                    continue
                try:
                    files.create(target, text)
                except Exception:  # Too large, or the path is taken by a directory: keep going.
                    report.skipped.append(target)
                    continue
                report.imported.append(target)
        return report


def _target_path(filename: str, text: str, project: str) -> str:
    match = _TYPE_RE.search(text.split("\n---", 1)[0]) if text.startswith("---") else None
    memory_type = match["type"].lower() if match else "project"
    name = _PREFIX_RE.sub("", filename)
    if memory_type in _USER_WIDE_TYPES:
        return f"/memories/{memory_type}/{name}"
    return f"/memories/projects/{project}/{name}"


def _exists(files: MemoryFiles, path: str) -> bool:
    try:
        files.view(path)
    except Exception:
        return False
    return True


def _checkout_for(slug: str) -> Path | None:
    """The directory Claude Code named *slug*, found by walking the filesystem from ``/``.

    The slug is lossy: ``/repos/ai-toolkit`` and ``/repos/ai/toolkit`` give
    the same one. So each step matches the directory's real entries by their
    own slugs, preferring the longest, and backtracks when a path dead-ends.
    """
    tokens = slug.lstrip("-").split("-")
    return _walk(Path("/"), tokens) if tokens and tokens[0] else None


def _walk(directory: Path, tokens: list[str]) -> Path | None:
    if not tokens:
        return directory
    try:
        entries = [entry for entry in directory.iterdir() if entry.is_dir()]
    except OSError:
        return None
    by_slug: dict[str, Path] = {}
    for entry in entries:
        by_slug.setdefault(_SLUG_RE.sub("-", entry.name), entry)
    for width in range(len(tokens), 0, -1):
        entry = by_slug.get("-".join(tokens[:width]))
        if entry is not None and (found := _walk(entry, tokens[width:])) is not None:
            return found
    return None


def _fallback_key(slug: str) -> str:
    """A key for a project whose checkout is gone: the whole slug, since its last part alone may collide."""
    from sessions_graph.memory_index import safe_key

    return safe_key(slug.strip("-"))


def _settings_path(home: Path) -> Path:
    return home / ".claude" / "settings.json"


def _read_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        settings = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise MemoryBackendError(f"{path} is not valid JSON ({exc}); fix it, then re-run.") from exc
    if not isinstance(settings, dict):
        raise MemoryBackendError(f"{path} must hold a JSON object; fix it, then re-run.")
    return settings


def _set(path: Path, value: bool | None) -> list[str]:
    """Write ``autoMemoryEnabled`` (None removes it), backing the file up once before the first change."""
    settings = _read_settings(path)
    if settings.get(_SETTING) == value or (value is None and _SETTING not in settings):
        return [f"Unchanged {path}"]
    if value is None:
        settings.pop(_SETTING)
    else:
        settings[_SETTING] = value
    backup = path.with_name(path.name + _BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    state = "removed" if value is None else json.dumps(value)
    return [f"Wrote {path} ({_SETTING} {state})"]


CLAUDE_CODE_MEMORY = ClaudeCodeMemory()
