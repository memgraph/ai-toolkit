"""Sessions Graph: Cross-session memory store for agents, backed by Memgraph.

Quick start::

    from sessions_graph import SessionsGraph

    graph = SessionsGraph()
    graph.setup()

    mem = graph.save_memory(user_id="alice", content="Prefers Python over TypeScript")
    results = graph.search_memories(user_id="alice", query="Python")
"""

from .core import SessionsGraph
from .models import Memory, MemoryValidationError
from .recall import RecallConfig, Recalled
from .reconciliation import ReconciliationSummary

__all__ = ["Memory", "MemoryValidationError", "RecallConfig", "Recalled", "ReconciliationSummary", "SessionsGraph"]
