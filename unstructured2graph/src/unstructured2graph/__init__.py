"""
Unstructured2Graph - Convert unstructured documents into knowledge graphs.

This package provides utilities for parsing various document formats and
ingesting them into Memgraph knowledge graphs using LightRAG.
"""

from .loaders import (
    Chunk,
    ChunkedDocument,
    from_unstructured,
    make_chunks,
    parse_source,
)
from .memgraph import (
    compute_embeddings,
    connect_chunks_to_entities,
    create_index,
    create_label_index,
    create_nodes_from_list,
    create_vector_search_index,
    link_nodes_in_order,
)

__version__ = "0.1.0"
__all__ = [
    "Chunk",
    "ChunkedDocument",
    "compute_embeddings",
    "connect_chunks_to_entities",
    "create_index",
    "create_label_index",
    "create_nodes_from_list",
    "create_vector_search_index",
    "from_unstructured",
    "link_nodes_in_order",
    "make_chunks",
    "parse_source",
]
