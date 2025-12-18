"""
Unstructured2Graph - Convert unstructured documents into knowledge graphs.

This package provides utilities for parsing various document formats and
ingesting them into Memgraph knowledge graphs using LightRAG.
"""

from .loaders import (
    parse_source,
    make_chunks,
    from_unstructured,
    Chunk,
    ChunkedDocument,
)
from .memgraph import (
    create_nodes_from_list,
    connect_chunks_to_entities,
    create_vector_search_index,
    compute_embeddings,
    create_index,
    create_label_index,
    link_nodes_in_order,
)

__version__ = "0.1.0"
__all__ = [
    "parse_source",
    "make_chunks",
    "from_unstructured",
    "Chunk",
    "ChunkedDocument",
    "create_nodes_from_list",
    "connect_chunks_to_entities",
    "create_vector_search_index",
    "compute_embeddings",
    "create_index",
    "create_label_index",
    "link_nodes_in_order",
]
