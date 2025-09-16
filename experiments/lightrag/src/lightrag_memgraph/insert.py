"""Text insertion functionality for LightRAG Memgraph integration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import LightRAGMemgraph


# TODO(gitbuda): This wrapper doesn't make much sense -> we should just use the LightRAG class directly.
async def insert_text(
    rag: "LightRAGMemgraph", text: str, file_path: str = None
) -> None:
    """Insert text into the knowledge graph.

    Args:
        rag: Initialized LightRAGMemgraph instance
        text: Text to insert and process
    """
    await rag.insert_text(text, file_path=file_path)
