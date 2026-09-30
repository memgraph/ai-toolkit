"""hygm: a shared graph schema/ontology model, and the strategies that produce one."""

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
    "CYPHER_IDENTIFIER_PATTERN",
    "IDENTITIES",
    "PERSON_LABEL",
    "USER_LABEL",
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
    "is_valid_identifier",
    "require_valid_identifier",
    "validate_model",
]
