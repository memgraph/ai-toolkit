"""
Validation module for HyGM graph models and Memgraph schemas.

This module provides comprehensive validation capabilities including:
- Pre-migration validation (Graph Schema Validation)
- Post-migration validation (Memgraph Schema Validation)
"""

from .base import (
    ValidationSeverity,
    ValidationCategory,
    ValidationIssue,
    ValidationMetrics,
    ValidationResult,
    BaseValidator,
    create_validation_issue,
)
from .graph_schema_validator import GraphSchemaValidator
from .validation import (
    MemgraphSchemaValidator,
    validate_migration_result,
)

__all__ = [
    # Base classes and types
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationMetrics",
    "ValidationResult",
    "BaseValidator",
    "create_validation_issue",
    # Pre-migration validation
    "GraphSchemaValidator",
    # Post-migration validation
    "MemgraphSchemaValidator",
    "validate_migration_result",
]
