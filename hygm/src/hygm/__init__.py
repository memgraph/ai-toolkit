"""hygm: a shared graph schema/ontology model, and the strategies that produce one."""

from .default import CATCH_ALL_LABELS, CORE_LABELS, VALUE_LABELS, default_model, with_core
from .identifiers import CYPHER_IDENTIFIER_PATTERN, is_valid_identifier, require_valid_identifier
from .strategies import LlmRecommendationStrategy, ManualStrategy, Observer, OwlImportStrategy
from .types import IDENTITIES, HygmModel, Identity, NodeType, RelationType
from .validation import (
    PERSON_LABEL,
    USER_LABEL,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    validate_model,
)

__all__ = [
    "CATCH_ALL_LABELS",
    "CORE_LABELS",
    "CYPHER_IDENTIFIER_PATTERN",
    "IDENTITIES",
    "PERSON_LABEL",
    "USER_LABEL",
    "VALUE_LABELS",
    "HygmModel",
    "Identity",
    "LlmRecommendationStrategy",
    "ManualStrategy",
    "NodeType",
    "Observer",
    "OwlImportStrategy",
    "RelationType",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    "default_model",
    "is_valid_identifier",
    "require_valid_identifier",
    "validate_model",
    "with_core",
]
