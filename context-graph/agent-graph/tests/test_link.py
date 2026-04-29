"""Unit tests for AgentLink and the adapter/connector protocol."""

import asyncio

from agent_graph import AgentLink
from agent_graph.events import (
    Event,
    EventType,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_graph.protocols import GraphConnector


class _RecordingConnector(GraphConnector):
    """Test connector that records all received events."""

    def __init__(self, *, filter_types: set[EventType] | None = None):
        self.events: list[Event] = []
        self._filter = filter_types

    def supports(self, event: Event) -> bool:
        if self._filter is None:
            return True
        return event.event_type in self._filter

    def on_event(self, event: Event) -> None:
        self.events.append(event)


class TestAgentLink:
    """Tests for the AgentLink hub."""

    def test_emit_routes_to_connector(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        link.emit(ToolStartEvent(session_id="s1", tool_name="Read"))
        assert len(rec.events) == 1
        assert rec.events[0].event_type == EventType.TOOL_START

    def test_emit_routes_to_multiple_connectors(self):
        link = AgentLink()
        rec1 = _RecordingConnector()
        rec2 = _RecordingConnector()
        link.add_connector(rec1)
        link.add_connector(rec2)

        link.emit(ToolStartEvent(session_id="s1", tool_name="Read"))
        assert len(rec1.events) == 1
        assert len(rec2.events) == 1

    def test_supports_filtering(self):
        link = AgentLink()
        tool_only = _RecordingConnector(filter_types={EventType.TOOL_START, EventType.TOOL_END})
        link.add_connector(tool_only)

        link.emit(ToolStartEvent(session_id="s1", tool_name="Read"))
        link.emit(MessageEvent(session_id="s1", role="user", content="hi"))
        link.emit(ToolEndEvent(session_id="s1", tool_name="Read", result="ok"))

        assert len(tool_only.events) == 2

    def test_remove_connector(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)
        link.remove_connector(rec)

        link.emit(ToolStartEvent(session_id="s1", tool_name="Read"))
        assert len(rec.events) == 0

    def test_connectors_property(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        assert len(link.connectors) == 1
        assert link.connectors[0] is rec


class TestClaudeAdapter:
    """Tests for the Claude adapter (no real SDK dependency)."""

    def test_auto_session_emits_session_start(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        from agent_graph.adapters.claude import ClaudeAdapter

        ClaudeAdapter(
            link,
            "s-test",
            session_kwargs={"model": "claude-sonnet-4-20250514", "tags": ["test"]},
        )

        assert len(rec.events) == 1
        assert rec.events[0].event_type == EventType.SESSION_START
        assert rec.events[0].session_id == "s-test"
        assert rec.events[0].model == "claude-sonnet-4-20250514"

    def test_no_auto_session(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        from agent_graph.adapters.claude import ClaudeAdapter

        ClaudeAdapter(link, "s-test", auto_session=False)
        assert len(rec.events) == 0

    def test_hooks_dict_keys(self):
        link = AgentLink()
        from agent_graph.adapters.claude import ClaudeAdapter

        adapter = ClaudeAdapter(link, "s-test")
        hooks = adapter.get_sdk_hooks()

        expected_keys = {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "UserPromptSubmit",
            "SubagentStart",
            "SubagentStop",
            "Notification",
            "Stop",
        }
        assert set(hooks.keys()) == expected_keys

    def test_pre_tool_use_emits_event(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        from agent_graph.adapters.claude import ClaudeAdapter

        adapter = ClaudeAdapter(link, "s-test", auto_session=False)

        asyncio.get_event_loop().run_until_complete(
            adapter._pre_tool_use(
                {"tool_name": "Read", "tool_input": {"path": "/f.py"}, "tool_use_id": "tu-1"},
                "tu-1",
                {},
            )
        )

        assert len(rec.events) == 1
        e = rec.events[0]
        assert e.event_type == EventType.TOOL_START
        assert e.tool_name == "Read"
        assert e.tool_use_id == "tu-1"

    def test_post_tool_use_emits_event(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        from agent_graph.adapters.claude import ClaudeAdapter

        adapter = ClaudeAdapter(link, "s-test", auto_session=False)

        asyncio.get_event_loop().run_until_complete(
            adapter._post_tool_use(
                {"tool_name": "Read", "tool_use_id": "tu-1", "tool_response": "file contents"},
                "tu-1",
                {},
            )
        )

        assert len(rec.events) == 1
        e = rec.events[0]
        assert e.event_type == EventType.TOOL_END
        assert e.result == "file contents"

    def test_user_prompt_emits_message(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        from agent_graph.adapters.claude import ClaudeAdapter

        adapter = ClaudeAdapter(link, "s-test", auto_session=False)

        asyncio.get_event_loop().run_until_complete(adapter._user_prompt_submit({"prompt": "Hello!"}, None, {}))

        assert len(rec.events) == 1
        e = rec.events[0]
        assert e.event_type == EventType.MESSAGE
        assert e.role == "user"
        assert e.content == "Hello!"

    def test_stop_emits_session_end(self):
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        from agent_graph.adapters.claude import ClaudeAdapter

        adapter = ClaudeAdapter(link, "s-test", auto_session=False)

        asyncio.get_event_loop().run_until_complete(adapter._stop({}, None, {}))

        assert len(rec.events) == 1
        assert rec.events[0].event_type == EventType.SESSION_END

    def test_full_lifecycle(self):
        """Simulate a full Claude session lifecycle."""
        link = AgentLink()
        rec = _RecordingConnector()
        link.add_connector(rec)

        from agent_graph.adapters.claude import ClaudeAdapter

        adapter = ClaudeAdapter(
            link,
            "s-lifecycle",
            session_kwargs={"model": "claude-sonnet-4-20250514"},
        )

        loop = asyncio.get_event_loop()

        # User sends prompt
        loop.run_until_complete(adapter._user_prompt_submit({"prompt": "Read main.py"}, None, {}))

        # Tool call
        loop.run_until_complete(
            adapter._pre_tool_use(
                {"tool_name": "Read", "tool_input": {"path": "main.py"}, "tool_use_id": "tu-1"},
                "tu-1",
                {},
            )
        )

        # Tool result
        loop.run_until_complete(
            adapter._post_tool_use(
                {"tool_name": "Read", "tool_use_id": "tu-1", "tool_response": "print('hi')"},
                "tu-1",
                {},
            )
        )

        # Stop
        loop.run_until_complete(adapter._stop({}, None, {}))

        types = [e.event_type for e in rec.events]
        assert types == [
            EventType.SESSION_START,
            EventType.MESSAGE,
            EventType.TOOL_START,
            EventType.TOOL_END,
            EventType.SESSION_END,
        ]
        # All events share the same session_id
        assert all(e.session_id == "s-lifecycle" for e in rec.events)
        # All events report source SDK
        assert all(e.source_sdk == "claude" for e in rec.events)


class TestMultiConnector:
    """Tests for routing to multiple connectors with different filters."""

    def test_mixed_connectors(self):
        link = AgentLink()
        all_rec = _RecordingConnector()
        tools_rec = _RecordingConnector(filter_types={EventType.TOOL_START, EventType.TOOL_END})
        sessions_rec = _RecordingConnector(filter_types={EventType.SESSION_START, EventType.SESSION_END})

        link.add_connector(all_rec)
        link.add_connector(tools_rec)
        link.add_connector(sessions_rec)

        link.emit(SessionStartEvent(session_id="s1"))
        link.emit(ToolStartEvent(session_id="s1", tool_name="Read"))
        link.emit(ToolEndEvent(session_id="s1", tool_name="Read", result="ok"))
        link.emit(SessionEndEvent(session_id="s1"))

        assert len(all_rec.events) == 4
        assert len(tools_rec.events) == 2
        assert len(sessions_rec.events) == 2
