"""Pluggable entity/relation extraction backends for unstructured2graph.

An ExtractionBackend takes one Chunk's text and ensures its entities (and,
for backends that support it, relations) end up in Memgraph -- tagged under
a stable workspace label with an `entity_type` property and a `file_path`
property equal to the chunk's hash. That's the only contract downstream code
relies on: connect_chunks_to_entities() and the ontology-gated label
promotion helpers in memgraph.py both operate purely on those properties and
a workspace-label string, so they work unmodified no matter which backend
produced the data.
"""

import logging
from typing import TYPE_CHECKING, Protocol

from lightrag_memgraph import MemgraphLightRAGWrapper
from memgraph_toolbox.api.memgraph import Memgraph

if TYPE_CHECKING:
    from .loaders import Chunk

logger = logging.getLogger(__name__)


class ExtractionBackend(Protocol):
    """Extracts entities/relations from one Chunk's text and persists them.

    `memgraph` is passed explicitly to aingest_chunk() even though
    LightRAGBackend ignores it -- LightRAG owns a completely separate
    storage layer, bridged via MEMGRAPH_URI/MEMGRAPH_USERNAME env vars in
    lightrag_memgraph.core, so it never touches this argument. A backend
    with no storage layer of its own (e.g. a local-model backend) needs it
    to write directly via this package's own Cypher helpers in memgraph.py.
    """

    @property
    def workspace_label(self) -> str:
        """The Memgraph label this backend's entity nodes are written under."""
        ...

    async def aingest_chunk(self, memgraph: Memgraph, chunk: "Chunk") -> None:
        """Extract from chunk.text and persist the result under workspace_label,
        tagging every entity node with file_path=chunk.hash."""
        ...


class LightRAGBackend:
    """Adapts a MemgraphLightRAGWrapper to the ExtractionBackend protocol.

    LightRAG performs its own LLM-based extraction and writes nodes/edges
    via its own Memgraph storage backend (a separate driver from
    memgraph_toolbox's Memgraph client) -- aingest_chunk() only drives that;
    it never writes anything itself.
    """

    def __init__(self, wrapper: MemgraphLightRAGWrapper) -> None:
        self.wrapper = wrapper

    @property
    def workspace_label(self) -> str:
        try:
            return self.wrapper.workspace
        except Exception as e:
            logger.warning(f"Could not auto-derive LightRAG entity workspace, falling back to 'base': {e}")
            return "base"

    async def aingest_chunk(self, memgraph: Memgraph, chunk: "Chunk") -> None:
        await self.wrapper.ainsert(input=chunk.text, file_paths=[chunk.hash])

    async def afinalize(self) -> None:
        """Finalize the underlying MemgraphLightRAGWrapper.

        Not part of the ExtractionBackend protocol -- GLiNER2Backend has no
        persistent resources needing an equivalent teardown, so this isn't a
        general contract every backend must implement. Exposed here so
        callers that do need to finalize a LightRAGBackend (e.g. between
        repeated runs in the same process) can call it directly instead of
        reaching through `.wrapper` into MemgraphLightRAGWrapper themselves.
        """
        await self.wrapper.afinalize()
