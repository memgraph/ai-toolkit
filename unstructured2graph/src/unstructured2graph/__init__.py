"""
Unstructured2Graph - Convert unstructured documents into knowledge graphs.

This package provides utilities for parsing various document formats and
ingesting them into Memgraph knowledge graphs via a pluggable extraction
backend (LightRAG by default; see gliner2_backend.GLiNER2Backend for a
local, LLM-free alternative -- not exported here since it requires the
optional `gliner2` dependency).
"""

from importlib import metadata

from .extraction_backend import ExtractionBackend, LightRAGBackend
from .loaders import (
    Chunk,
    ChunkedDocument,
    Document,
    Segment,
    enqueue_texts,
    from_documents,
    from_texts,
    from_unstructured,
    make_chunks,
    parse_source,
    parse_text,
    process_enqueued_and_finalize,
)
from .memgraph import (
    OntologyReport,
    compute_embeddings,
    connect_chunks_to_entities,
    create_entity_type_constraint,
    create_label_index,
    create_nodes_from_list,
    create_property_index,
    create_unique_constraint,
    create_vector_search_index,
    enforce_relation_domain_range,
    ensure_lookup,
    link_nodes_in_order,
    ontology_report,
    promote_all_entity_types_to_labels,
    promote_entity_types_to_labels,
    upsert_typed_relationships,
)
from .ontology import (
    DEFAULT_ONTOLOGY,
    DEFAULT_ONTOLOGY_PATH,
    EntityType,
    Ontology,
    RelationType,
    load_ontology,
)

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "DEFAULT_ONTOLOGY",
    "DEFAULT_ONTOLOGY_PATH",
    "Chunk",
    "ChunkedDocument",
    "Document",
    "EntityType",
    "ExtractionBackend",
    "LightRAGBackend",
    "Ontology",
    "OntologyReport",
    "RelationType",
    "Segment",
    "__version__",
    "compute_embeddings",
    "connect_chunks_to_entities",
    "create_entity_type_constraint",
    "create_label_index",
    "create_nodes_from_list",
    "create_property_index",
    "create_unique_constraint",
    "create_vector_search_index",
    "enforce_relation_domain_range",
    "enqueue_texts",
    "ensure_lookup",
    "from_documents",
    "from_texts",
    "from_unstructured",
    "link_nodes_in_order",
    "load_ontology",
    "make_chunks",
    "ontology_report",
    "parse_source",
    "parse_text",
    "process_enqueued_and_finalize",
    "promote_all_entity_types_to_labels",
    "promote_entity_types_to_labels",
    "upsert_typed_relationships",
]
