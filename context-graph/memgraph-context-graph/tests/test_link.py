"""Unit tests for AgentLink and the connector protocol."""

from memgraph_context_graph import AgentLink
from memgraph_context_graph.events import (
    Event,
    EventType,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from memgraph_context_graph.protocols import GraphConnector


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
