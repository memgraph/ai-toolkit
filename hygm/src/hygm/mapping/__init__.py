"""Layer 3: mapping files binding a model's types to the sources their data comes from (docs/contract.md, section 6)."""

from .format import load_mapping, mapping_from_mapping, mapping_schema, mapping_to_mapping
from .rendering import Renderer
from .types import (
    SOURCE_KINDS,
    BindingKind,
    ColumnEndpoint,
    GraphRef,
    Join,
    JoinEndpoint,
    Mapping,
    NodeBinding,
    PropertyBinding,
    RelationBinding,
    Source,
    SourceKind,
    TableRef,
)
from .validation import check_mapping

__all__ = [
    "SOURCE_KINDS",
    "BindingKind",
    "ColumnEndpoint",
    "GraphRef",
    "Join",
    "JoinEndpoint",
    "Mapping",
    "NodeBinding",
    "PropertyBinding",
    "RelationBinding",
    "Renderer",
    "Source",
    "SourceKind",
    "TableRef",
    "check_mapping",
    "load_mapping",
    "mapping_from_mapping",
    "mapping_schema",
    "mapping_to_mapping",
]
