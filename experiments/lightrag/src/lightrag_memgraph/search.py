"""Search functionality for LightRAG Memgraph integration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import LightRAGMemgraph


async def naive_search(rag: "LightRAGMemgraph", query: str) -> str:
    """Perform naive search on the knowledge graph.

    Args:
        rag: Initialized LightRAGMemgraph instance
        query: Search query string

    Returns:
        Search results as string
    """
    return await rag.query_naive(query)


async def hybrid_search(rag: "LightRAGMemgraph", query: str) -> str:
    """Perform hybrid search on the knowledge graph.

    Args:
        rag: Initialized LightRAGMemgraph instance
        query: Search query string

    Returns:
        Search results as string
    """
    return await rag.query_hybrid(query)
