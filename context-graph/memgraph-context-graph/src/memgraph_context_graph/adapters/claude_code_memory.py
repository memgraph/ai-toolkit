"""Claude Code's auto memory, as Context Graph switches away from and imports it.

Auto memory is Markdown files in ``~/.claude/projects/<project>/memory/``, one
folder per git repository, shared by its worktrees and subdirectories (outside
a repository, per project folder). ``<project>`` is that folder's absolute path
with every character other than a letter or digit replaced by ``-``; the other
``~/.claude/projects/`` folders, one per directory a session started in, hold
transcripts and are never read for memory. ``MEMORY.md`` there
is the index Claude Code loads at session start; the other files are
memories, each with frontmatter naming its ``type``. The ``autoMemoryEnabled``
setting turns it off; a plugin can't set it, so this edits the user's own
``~/.claude/settings.json``. A project's ``.claude/settings.json`` or
``.claude/settings.local.json`` outranks it there, so :meth:`ClaudeCodeMemory.overridden_in`
finds a project that turns it back on.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memgraph_context_graph.memory_backend import ExportReport, ImportReport, MemoryBackendError

if TYPE_CHECKING:
    from memgraph_context_graph.memory_backend import MemoryFiles, MemorySource

_SETTING = "autoMemoryEnabled"
_INDEX_FILE = "MEMORY.md"
_BACKUP_SUFFIX = ".context-graph.bak"
_TYPE_RE = re.compile(r"^\s*type:\s*[\"']?(?P<type>[A-Za-z]+)", re.MULTILINE)
_PREFIX_RE = re.compile(r"^(user|feedback|project|reference)_")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]")
_NAME_RE = re.compile(r"^\s*name:\s*(?P<value>.+?)\s*$", re.MULTILINE)
_DESCRIPTION_RE = re.compile(r"^\s*description:\s*(?P<value>.+?)\s*$", re.MULTILINE)
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

    def overridden_in(self, project: Path) -> Path | None:
        """The settings file in *project* that turns auto memory back on there, or None.

        ``settings.local.json`` outranks ``settings.json``, so a ``false`` there wins over a ``true`` beside it.

        Raises:
            MemoryBackendError: one of the files isn't a JSON object.
        """
        for name in ("settings.local.json", "settings.json"):
            path = project / ".claude" / name
            value = _read_settings(path).get(_SETTING)
            if value is not None:
                return path if value is True else None
        return None

    def import_into(self, files: MemoryFiles, home: Path) -> ImportReport:
        """Copy every project's auto-memory files into the graph's ``/memories`` tree.

        ``user``/``feedback`` memories go under ``/memories/<type>/``, since they
        apply everywhere; ``project``/``reference`` (and untyped) ones under
        that project's ``/memories/projects/<key>/``, keyed as the session-start
        index keys the checkout. A ``user``/``feedback`` memory whose name the
        graph already holds with other text, so another project's own version,
        goes under its project's folder under its own filename instead, where
        handing memory back finds it again. ``MEMORY.md`` is skipped: the index
        is generated now. The original files are left as they are.
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
                lineage = _lineage(files, target)
                if lineage and text not in lineage and not target.startswith("/memories/projects/"):
                    target = f"/memories/projects/{project}/{source.name}"
                    lineage = _lineage(files, target)
                if lineage:
                    report.skipped.append(target)
                    continue
                try:
                    files.create(target, text)
                except Exception:  # Too large, or the path is taken by a directory: keep going.
                    report.skipped.append(target)
                    continue
                report.imported.append(target)
        return report

    def export_from(self, source: MemorySource, home: Path) -> ExportReport:
        """Write the graph's memories back into auto memory, the reverse of :meth:`import_into`.

        A project's files go to its repository's memory folder. A ``user``/``feedback`` file goes back
        to the folders it was imported from, which still hold it; one first written in the graph goes
        to the repositories of the sessions that wrote it. A file there is overwritten only when it is
        an earlier text of the graph's file, so a newer or unrelated native file is kept and reported.
        Each folder written to gets its ``MEMORY.md`` rebuilt. Files with no folder to go to (a project
        Claude Code never opened, nested paths, other agents' files) stay in the graph only.
        """
        folders = _memory_folders(home)
        every_folder = sorted({folder for found in folders.values() for folder in found})
        report = ExportReport()
        written: set[Path] = set()
        for memory in source.files():
            parts = memory.path.removeprefix("/memories/").split("/")
            if len(parts) == 2 and parts[0] in _USER_WIDE_TYPES:
                name = _native_name(parts[1], memory.content, parts[0])
                targets = [folder for folder in every_folder if (folder / name).is_file()] or [
                    folder for key in source.written_in(memory.path) for folder in folders.get(key, [])
                ]
            elif len(parts) == 3 and parts[0] == "projects":
                name = _native_name(parts[2], memory.content, "project")
                targets = folders.get(parts[1], [])
            else:
                targets = []
            if not targets:
                report.skipped.append(memory.path)
                continue
            lineage = {memory.content} | {version.content for version in source.versions(memory.path)}
            kept = []
            for folder in targets:
                if _write_memory(folder, name, memory.content, lineage):
                    written.add(folder)
                elif (folder / name).read_text(encoding="utf-8") != memory.content:
                    kept.append(str(folder / name))
            report.kept += kept
            if len(kept) < len(targets):
                report.exported.append(memory.path)
        for folder in sorted(written):
            _write_index(folder)
        return report


def _memory_folders(home: Path) -> dict[str, list[Path]]:
    """Each project key's auto-memory folders, keyed as the import keys them: one per repository Claude Code opened.

    Every ``~/.claude/projects/`` folder whose checkout is in a repository leads to the one folder named after
    that repository's main checkout, the only one Claude Code reads memory from; a folder whose checkout is
    gone counts only when it already holds memory.
    """
    from sessions_graph.memory_index import project_key_for

    projects = home / ".claude" / "projects"
    folders: dict[str, set[Path]] = {}
    for project in sorted(projects.iterdir() if projects.is_dir() else []):
        if not project.is_dir():
            continue
        checkout = _checkout_for(project.name)
        if checkout is None:
            if (project / "memory").is_dir():
                folders.setdefault(_fallback_key(project.name), set()).add(project / "memory")
            continue
        root = _repository_root(checkout) or checkout
        folder = projects / _SLUG_RE.sub("-", str(root)) / "memory"
        key = project_key_for(checkout) or _fallback_key(folder.parent.name)
        folders.setdefault(key, set()).add(folder)
    return {key: sorted(found) for key, found in folders.items()}


def _repository_root(checkout: Path) -> Path | None:
    """The main checkout of the repository *checkout* is in, which its worktrees share; None outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    common = Path(result.stdout.strip())
    return common.parent if result.returncode == 0 and common.name == ".git" else None


def _native_name(name: str, text: str, default_type: str) -> str:
    """The auto-memory filename for a graph file: the ``<type>_`` prefix the import stripped, put back."""
    match = _TYPE_RE.search(_frontmatter(text))
    memory_type = match["type"].lower() if match else default_type
    return name if _PREFIX_RE.match(name) else f"{memory_type}_{name}"


def _write_memory(folder: Path, name: str, text: str, lineage: set[str]) -> bool:
    """Write *text* to *folder*/*name* unless a file there holds text outside *lineage*; returns whether it wrote.

    *lineage* is the graph file's current and earlier texts. A native file among them is one the graph's copy
    descends from, so writing over it loses nothing; any other was written beside the graph and is kept.
    """
    path = folder / name
    if path.is_file():
        current = path.read_text(encoding="utf-8")
        if current == text or current not in lineage:
            return False
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _write_index(folder: Path) -> None:
    """Rebuild ``MEMORY.md`` as Claude Code writes it: one ``- [Title](file.md) — hook`` line per memory."""
    lines = []
    for path in sorted(folder.glob("*.md")):
        if path.name == _INDEX_FILE:
            continue
        frontmatter = _frontmatter(path.read_text(encoding="utf-8"))
        title = match["value"].strip("\"'") if (match := _NAME_RE.search(frontmatter)) else path.stem
        hook = match["value"].strip("\"'") if (match := _DESCRIPTION_RE.search(frontmatter)) else ""
        lines.append(f"- [{title}]({path.name})" + (f" — {hook}" if hook else ""))
    (folder / _INDEX_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frontmatter(text: str) -> str:
    return text.split("\n---", 1)[0] if text.startswith("---") else ""


def _target_path(filename: str, text: str, project: str) -> str:
    match = _TYPE_RE.search(_frontmatter(text))
    memory_type = match["type"].lower() if match else "project"
    name = _PREFIX_RE.sub("", filename)
    if memory_type in _USER_WIDE_TYPES:
        return f"/memories/{memory_type}/{name}"
    return f"/memories/projects/{project}/{name}"


def _lineage(files: MemoryFiles, path: str) -> set[str]:
    """The graph file at *path*'s current and earlier texts; empty when there is no such file."""
    current = files.read(path)
    if current is None:
        return set()
    return {current} | {version.content for version in files.versions(path)}


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
