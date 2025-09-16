"""Core LightRAG Memgraph integration functionality."""

import os
from typing import Optional

import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger


class LightRAGMemgraph:
    """LightRAG wrapper with Memgraph storage backend."""

    def __init__(
        self, working_dir: str = "./lightrag_storage", log_level: str = "INFO"
    ):
        """Initialize LightRAG with Memgraph storage.

        Args:
            working_dir: Directory for LightRAG storage
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.working_dir = working_dir
        self.log_level = log_level
        self.rag: Optional[LightRAG] = None

    async def initialize(self) -> None:
        """Initialize the LightRAG instance with Memgraph storage."""
        setup_logger("lightrag", level=self.log_level)

        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        self.rag = LightRAG(
            working_dir=self.working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
            # TODO: This should be read from environment variables
            # TODO: MemgraphStorage is missing from LightRAG documentation
            graph_storage="MemgraphStorage",
        )

        # IMPORTANT: Both initialization calls are required!
        await self.rag.initialize_storages()  # Initialize storage backends
        await initialize_pipeline_status()  # Initialize processing pipeline

    async def query_naive(self, query: str) -> str:
        """Perform naive search on the knowledge graph.

        Args:
            query: Search query string

        Returns:
            Search results as string

        Raises:
            RuntimeError: If LightRAG is not initialized
        """
        if self.rag is None:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        return await self.rag.aquery(query, param=QueryParam(mode="naive"))

    async def query_hybrid(self, query: str) -> str:
        """Perform hybrid search on the knowledge graph.

        Args:
            query: Search query string

        Returns:
            Search results as string

        Raises:
            RuntimeError: If LightRAG is not initialized
        """
        if self.rag is None:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        return await self.rag.aquery(query, param=QueryParam(mode="hybrid"))

    async def insert_text(self, text: str, file_path: str = None) -> None:
        """Insert text into the knowledge graph.

        Args:
            text: Text to insert and process

        Raises:
            RuntimeError: If LightRAG is not initialized
        """
        if self.rag is None:
            raise RuntimeError("LightRAG not initialized. Call initialize() first.")
        await self.rag.ainsert(text, file_paths=file_path)


async def initialize_lightrag_memgraph(
    working_dir: str = "./lightrag_storage",
) -> LightRAGMemgraph:
    """Convenience function to initialize LightRAG with Memgraph storage.

    Args:
        working_dir: Directory for LightRAG storage

    Returns:
        Initialized LightRAGMemgraph instance
    """
    rag = LightRAGMemgraph(working_dir=working_dir)
    await rag.initialize()
    return rag
