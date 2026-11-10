"""hygm: a shared graph schema/ontology model, and the strategies that produce one."""

from .default import default_model, with_core
from .errors import ModelError
from .format import model_schema, needs_version_2
from .identifiers import CYPHER_IDENTIFIER_PATTERN, is_valid_identifier, require_valid_identifier
from .labels import CATCH_ALL_LABELS, CORE_LABELS, VALUE_LABELS
from .strategies import (
    Change,
    Derivation,
    DerivationError,
    DerivationLimits,
    Llm,
    LlmRecommendationStrategy,
    ManualStrategy,
    Observer,
    OwlImportStrategy,
    model_from_mapping,
    model_to_mapping,
)
from .types import IDENTITIES, PROPERTY_TYPES, HygmModel, Identity, NodeType, Property, PropertyType, RelationType
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
    "PROPERTY_TYPES",
    "USER_LABEL",
    "VALUE_LABELS",
    "Change",
    "Derivation",
    "DerivationError",
    "DerivationLimits",
    "HygmModel",
    "Identity",
    "Llm",
    "LlmRecommendationStrategy",
    "ManualStrategy",
    "ModelError",
    "NodeType",
    "Observer",
    "OwlImportStrategy",
    "Property",
    "PropertyType",
    "RelationType",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    "default_model",
    "is_valid_identifier",
    "model_from_mapping",
    "model_schema",
    "model_to_mapping",
    "needs_version_2",
    "require_valid_identifier",
    "validate_model",
    "with_core",
]
