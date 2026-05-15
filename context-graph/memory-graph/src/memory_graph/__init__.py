"""Memory Graph: Cross-session memory store for agents, backed by Memgraph.

Quick start::

    from memory_graph import MemoryGraph, Memory

    graph = MemoryGraph()
    graph.setup()

    mem = graph.save_memory(user_id="alice", content="Prefers Python over TypeScript")
    results = graph.search_memories(user_id="alice", query="Python")

Integration with Agent Context Graph::

    from memory_graph import MemoryGraph
    from memory_graph.connector import MemoryGraphConnector
    from agent_context_graph import AgentLink
    from agent_context_graph.adapters.claude import ClaudeAdapter

    graph = MemoryGraph()
    graph.setup()

    connector = MemoryGraphConnector(graph)
    link = AgentLink()
    link.add_connector(connector)
    adapter = ClaudeAdapter(link, session_id="s-1", session_kwargs={"user_id": "alice"})

    # Memory writes and recalls go through the Python API:
    graph.save_memory(
        user_id=connector.active_user_id,
        content="User works in the ai-toolkit repo",
        session_id=connector.active_session_id,
    )
"""

from .core import MemoryGraph
from .models import Memory, MemoryValidationError

__all__ = ["Memory", "MemoryGraph", "MemoryValidationError"]
