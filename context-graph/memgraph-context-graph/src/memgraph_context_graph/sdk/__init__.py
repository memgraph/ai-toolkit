"""Context Graph as the memory of an app built on an agent SDK.

A coding agent reaches Context Graph through command hooks and the stdio MCP
server; an app built on an agent SDK runs the agent in its own process, so it
gets the same three things in-process instead:

- **capture**: the SDK's own hooks emit the Event Protocol through the
  configured connectors, so the app's sessions land in the same graph as the
  user's coding-agent sessions and are embedded and reconciled the same way;
- **context**: the session-start text a hook would inject (the memory index,
  where the user left off in this project), handed to the SDK's own
  context-injection point;
- **tools**: ``recall`` and ``memory``, wrapped as the SDK's native tools.

:class:`ContextGraphMemory` is that, SDK-agnostic. The per-SDK wiring lives in
:mod:`memgraph_context_graph.sdk.claude` (Claude Agent SDK) and
:mod:`memgraph_context_graph.sdk.openai` (OpenAI Agents SDK).

The Memgraph connection, capture policy and connectors come from the config
file, as for hooks: the detached embed and reconcile processes a session end
starts read that same file, so an app can't point capture somewhere they
don't follow. What an app may set is *whose* memory it is (``user_id``, for a
multi-user app) and the project (``cwd``).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from memgraph_context_graph import capture_log
from memgraph_context_graph.adapters._identity import MEMORY_BACKEND_GRAPH, load_config
from memgraph_context_graph.capture_policy import policy_from_config
from memgraph_context_graph.connectors import configured_connectors, supported_connectors
from memgraph_context_graph.events import SessionEndEvent, SessionStartEvent
from memgraph_context_graph.tools import ToolError, available_tools, normalize_connector_name, session_hints

if TYPE_CHECKING:
    from collections.abc import Iterable

    from memgraph_context_graph.adapters._identity import HookConfig
    from memgraph_context_graph.events import Event
    from memgraph_context_graph.link import AgentLink
    from memgraph_context_graph.tools import Tool

__all__ = ["ContextGraphMemory"]

#: How long capture stays off after Memgraph couldn't be reached, so each event doesn't wait out a connect timeout.
RECONNECT_AFTER_SECONDS = 30.0


class ContextGraphMemory:
    """One user's Context Graph memory for an SDK app: capture, session context and tools.

    Capture never fails the app: a connector that can't be built or fails an
    event goes to the capture-failure log, as a hook's would. Events are
    written one at a time, in the order they are emitted, whichever thread
    emits them. Tools never raise: a failure is text for the model.

    Args:
        user_id: Whose memory; set by the application, never by the model.
            Defaults to the config file's ``identity.user_id``.
        cwd: The project the sessions belong to, as a hook's ``cwd`` does.
            Defaults to the current directory.
        connectors: The graph connectors capture writes through. Defaults to
            the config file's ``[capture] connectors``.
        memory_tool: Whether the model gets the ``memory`` tool and the
            session context its index. An SDK app has no other memory to
            switch off, so it is on unless the app says otherwise.
        link: Capture through this link, with whatever connectors it holds,
            instead of building one from the config file; for a library
            caller wiring its own graphs.
    """

    def __init__(
        self,
        *,
        user_id: str | None = None,
        cwd: str | None = None,
        connectors: Iterable[str] | None = None,
        memory_tool: bool = True,
        link: AgentLink | None = None,
    ) -> None:
        base = load_config()
        self.config: HookConfig = replace(
            base,
            user_id=user_id or base.user_id,
            memory_backend=MEMORY_BACKEND_GRAPH if memory_tool else base.memory_backend,
        )
        self.cwd = cwd or os.getcwd()
        self.connectors = supported_connectors(connectors if connectors is not None else configured_connectors())
        self._link: AgentLink | None = link
        self._unreachable_since: float | None = None
        self._lock = threading.Lock()
        self._started: dict[str, str] = {}
        self._tools: dict[str, Tool] | None = None

    @property
    def user_id(self) -> str | None:
        """Whose memory this is."""
        return self.config.user_id

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @property
    def link(self) -> AgentLink:
        """The link capture emits through, connected on first use.

        Raises:
            ValueError: Memgraph can't be reached.
        """
        if self._link is None:
            from memgraph_context_graph.hooks.runner import create_link
            from memgraph_context_graph.spaces import spaces_for

            policy, _problems = policy_from_config(self.config)
            self._link = create_link(
                self.connectors,
                memgraph_env=spaces_for(self.config).personal().target.env(),
                on_error=self._connector_failed,
                policy=policy,
            )
        return self._link

    def emit(self, event: Event) -> None:
        """Send *event* through the configured connectors; dropped while Memgraph is unreachable."""
        with self._lock:
            unreachable = self._unreachable_since
            if unreachable is not None and time.monotonic() - unreachable < RECONNECT_AFTER_SECONDS:
                return
            try:
                link = self.link
            except Exception as exc:  # The app carries on uncaptured, as a hook's session would.
                self._unreachable_since = time.monotonic()
                capture_log.record_failure("connect", exc, runtime=event.source_sdk, session_id=event.session_id)
                return
            self._unreachable_since = None
            try:
                link.emit(event)
            except Exception as exc:  # A caller's own link may let a connector's failure through.
                capture_log.record_failure(
                    "emit",
                    exc,
                    runtime=event.source_sdk,
                    session_id=event.session_id,
                    event_type=event.event_type.value,
                )

    async def emit_async(self, event: Event) -> None:
        """:meth:`emit` off the event loop; the connectors' writes are blocking."""
        await asyncio.to_thread(self.emit, event)

    def start_session(self, session_id: str, *, source_sdk: str, model: str | None = None) -> bool:
        """Record that *session_id* started, once; returns whether this call started it."""
        if session_id in self._started:
            return False
        self._started[session_id] = source_sdk
        self.emit(
            SessionStartEvent(
                session_id=session_id,
                source_sdk=source_sdk,
                model=model,
                working_directory=self.cwd,
                user_id=self.user_id,
            )
        )
        return True

    def end_session(self, session_id: str | None = None, *, status: str = "completed") -> None:
        """End *session_id*, or every session this memory started: embed what is left and mark it for reconciliation."""
        ending = [session_id] if session_id is not None else list(self._started)
        for sid in ending:
            source_sdk = self._started.pop(sid, "")
            self.emit(SessionEndEvent(session_id=sid, source_sdk=source_sdk, status=status))

    def _connector_failed(self, name: str, event: Event | None, exc: Exception) -> None:
        """Log a connector that couldn't be built (``setup``) or failed an event (``emit``), as the hook runner does."""
        if event is None:
            capture_log.record_failure("setup", exc, connector=name, cwd=self.cwd)
            return
        capture_log.record_failure(
            "emit",
            exc,
            connector=name,
            runtime=event.source_sdk,
            session_id=event.session_id,
            event_type=event.event_type.value,
            cwd=self.cwd,
        )

    # ------------------------------------------------------------------
    # Context and tools
    # ------------------------------------------------------------------

    def session_context(self, session_id: str | None = None) -> str | None:
        """What a session starts knowing: which memory tools it has, the memory index, where the user left off.

        None when no tool has anything to say. A failure to read it never fails
        the app; the model just isn't told.
        """
        payload = {"cwd": self.cwd, "session_id": session_id or ""}
        try:
            return "\n".join(session_hints(self.connectors, self.config, payload=payload)) or None
        except Exception as exc:
            capture_log.record_failure("respond", exc, level="warning", session_id=session_id, cwd=self.cwd)
            return None

    def tools(self) -> dict[str, Tool]:
        """The tools this memory gives the model, keyed by name: those reading an enabled connector's graph."""
        if self._tools is None:
            enabled = {normalize_connector_name(name) for name in self.connectors}
            self._tools = {
                name: tool
                for name, tool in available_tools(self.config).items()
                if normalize_connector_name(tool.connector) in enabled
            }
        return self._tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run tool *name* for this memory's user; returns the text for the model and whether it is an error."""
        tool = self.tools().get(name)
        if tool is None:
            return f"Unknown tool: {name}", True
        try:
            return tool.call(arguments, self.config).text, False
        except ToolError as exc:
            return str(exc), True
        except Exception as exc:  # Memgraph down, say: the model is told, and the run goes on.
            capture_log.record_failure("respond", exc, level="warning", tool_name=name, cwd=self.cwd)
            return f"{name} failed: {type(exc).__name__}: {exc}", True

    async def call_tool_async(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """:meth:`call_tool` off the event loop."""
        return await asyncio.to_thread(self.call_tool, name, arguments)
