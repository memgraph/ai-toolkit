"""
Unstructured2Graph - Convert unstructured documents into knowledge graphs.

This package provides utilities for parsing various document formats and
ingesting them into Memgraph knowledge graphs using LightRAG.
"""

from .loaders import (
    Chunk,
    ChunkedDocument,
    from_texts,
    from_unstructured,
    make_chunks,
    parse_source,
    parse_text,
)
from .memgraph import (
    compute_embeddings,
    connect_chunks_to_entities,
    create_label_index,
    create_nodes_from_list,
    create_property_index,
    create_unique_constraint,
    create_vector_search_index,
    link_nodes_in_order,
    promote_entity_types_to_labels,
)
from .ontology import DEFAULT_ONTOLOGY, EntityType, Ontology

__version__ = "0.5.0"
__all__ = [
    "DEFAULT_ONTOLOGY",
    "Chunk",
    "ChunkedDocument",
    "EntityType",
    "Ontology",
    "compute_embeddings",
    "connect_chunks_to_entities",
    "create_label_index",
    "create_nodes_from_list",
    "create_property_index",
    "create_unique_constraint",
    "create_vector_search_index",
    "from_texts",
    "from_unstructured",
    "link_nodes_in_order",
    "make_chunks",
    "parse_source",
    "parse_text",
    "promote_entity_types_to_labels",
]
