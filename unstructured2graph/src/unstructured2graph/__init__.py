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
    promote_all_entity_types_to_labels,
    promote_entity_types_to_labels,
)
from .ontology import DEFAULT_ONTOLOGY, DEFAULT_ONTOLOGY_PATH, EntityType, Ontology, load_ontology

__version__ = "0.6.0"
__all__ = [
    "DEFAULT_ONTOLOGY",
    "DEFAULT_ONTOLOGY_PATH",
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
    "load_ontology",
    "make_chunks",
    "parse_source",
    "parse_text",
    "promote_all_entity_types_to_labels",
    "promote_entity_types_to_labels",
]
