"""The Claude API memory tool, backed by Context Graph, for apps built on the Anthropic SDK.

``MemgraphMemoryTool`` (and its async twin) plug into the SDK's tool runner
the same way its local-filesystem memory tool does, but read and write a
user's ``(:Memory)`` files in Memgraph. Those are the files the ``memory``
tool serves to Claude Code and Codex, so an app and the user's coding agents
share one memory.

The user is fixed when the tool is built, by the application, never by the
model: a multi-user app builds one tool per request or per user.

Needs the ``sessions-graph[anthropic]`` extra::

    from anthropic import Anthropic
    from sessions_graph import SessionsGraph
    from sessions_graph.anthropic_memory import MemgraphMemoryTool

    graph = SessionsGraph()
    graph.setup()
    memory = MemgraphMemoryTool(graph, user_id="alice")
    runner = Anthropic().beta.messages.tool_runner(
        model="claude-opus-5-5", max_tokens=1024, tools=[memory],
        messages=[{"role": "user", "content": "Remember that I prefer uv."}],
    )
    runner.until_done()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anthropic.lib.tools as _sdk_tools
from anthropic.lib.tools import BetaAbstractMemoryTool, BetaAsyncAbstractMemoryTool
from anyio import to_thread

from .memory_store import MemoryCommandError, MemoryStore
from .models import MEMORY_ROOT


class _MemoryToolError(Exception):
    """A memory command that can't be carried out, on SDKs that predate ``ToolError``."""


# Newer SDKs return a ToolError's message to Claude verbatim; older ones send
# repr() of whatever a tool raises, so the spec's error text still arrives, wrapped.
ToolError: type[Exception] = getattr(_sdk_tools, "ToolError", _MemoryToolError)

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaMemoryTool20250818CreateCommand,
        BetaMemoryTool20250818DeleteCommand,
        BetaMemoryTool20250818InsertCommand,
        BetaMemoryTool20250818RenameCommand,
        BetaMemoryTool20250818StrReplaceCommand,
        BetaMemoryTool20250818ViewCommand,
    )

    from .core import SessionsGraph


class _Commands:
    """The six commands mapped onto a :class:`MemoryStore`, errors raised as the SDK's ``ToolError``."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def run(self, command: str, *args: Any) -> str:
        try:
            return getattr(self._store, command)(*args)
        except MemoryCommandError as exc:
            raise ToolError(str(exc)) from exc

    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        view_range = command.view_range
        return self.run("view", command.path, (view_range[0], view_range[1]) if view_range else None)

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        return self.run("create", command.path, command.file_text)

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        return self.run("str_replace", command.path, command.old_str, command.new_str or "")

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        return self.run("insert", command.path, command.insert_line, command.insert_text)

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        return self.run("delete", command.path)

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        return self.run("rename", command.old_path, command.new_path)

    def clear_all_memory(self) -> str:
        """Delete every memory file the user has. History is kept as ``MemoryVersion``s."""
        for top in {file.path.split("/")[2] for file in self._store.files()}:
            self._store.delete(f"{MEMORY_ROOT}/{top}")
        return "All memory cleared"


class MemgraphMemoryTool(BetaAbstractMemoryTool):
    """The memory tool over one user's Context Graph memory files.

    Args:
        graph:      A set-up :class:`~sessions_graph.SessionsGraph`.
        user_id:    Whose memory; supplied by the application, never by the model.
        session_id: Recorded as ``PRODUCED_MEMORY`` provenance on every write, when given.
        cache_control: Passed to the tool definition, as the SDK's own memory tools accept.
    """

    def __init__(
        self,
        graph: SessionsGraph,
        user_id: str,
        *,
        session_id: str | None = None,
        cache_control: BetaCacheControlEphemeralParam | None = None,
    ) -> None:
        super().__init__(cache_control=cache_control)
        self._commands = _Commands(graph.memory_store(user_id, session_id=session_id))

    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        """A directory listing or a file's numbered lines."""
        return self._commands.view(command)

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        """Create or overwrite a file."""
        return self._commands.create(command)

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        """Replace one unique occurrence."""
        return self._commands.str_replace(command)

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        """Insert after a line."""
        return self._commands.insert(command)

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        """Delete a file or directory."""
        return self._commands.delete(command)

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        """Move a file or directory."""
        return self._commands.rename(command)

    def clear_all_memory(self) -> str:
        """Delete every memory file the user has."""
        return self._commands.clear_all_memory()


class AsyncMemgraphMemoryTool(BetaAsyncAbstractMemoryTool):
    """:class:`MemgraphMemoryTool` for the async client; each command runs in a worker thread."""

    def __init__(
        self,
        graph: SessionsGraph,
        user_id: str,
        *,
        session_id: str | None = None,
        cache_control: BetaCacheControlEphemeralParam | None = None,
    ) -> None:
        super().__init__(cache_control=cache_control)
        self._commands = _Commands(graph.memory_store(user_id, session_id=session_id))

    async def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        """A directory listing or a file's numbered lines."""
        return await to_thread.run_sync(self._commands.view, command)

    async def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        """Create or overwrite a file."""
        return await to_thread.run_sync(self._commands.create, command)

    async def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        """Replace one unique occurrence."""
        return await to_thread.run_sync(self._commands.str_replace, command)

    async def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        """Insert after a line."""
        return await to_thread.run_sync(self._commands.insert, command)

    async def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        """Delete a file or directory."""
        return await to_thread.run_sync(self._commands.delete, command)

    async def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        """Move a file or directory."""
        return await to_thread.run_sync(self._commands.rename, command)

    async def clear_all_memory(self) -> str:
        """Delete every memory file the user has."""
        return await to_thread.run_sync(self._commands.clear_all_memory)
