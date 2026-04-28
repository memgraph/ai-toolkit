"""
Validation module for HyGM graph models and Memgraph schemas.

This module provides comprehensive validation capabilities including:
- Pre-migration validation (Graph Schema Validation)
- Post-migration validation (Memgraph Data Validation)
"""

from .base import (
    BaseValidator,
    ValidationCategory,
    ValidationIssue,
    ValidationMetrics,
    ValidationResult,
    ValidationSeverity,
    create_validation_issue,
)
from .graph_schema_validator import GraphSchemaValidator
from .memgraph_data_validator import (
    MemgraphDataValidator,
    validate_memgraph_data,
)

__all__ = [
    "BaseValidator",
    # Pre-migration validation
    "GraphSchemaValidator",
    # Post-migration validation
    "MemgraphDataValidator",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationMetrics",
    "ValidationResult",
    # Base classes and types
    "ValidationSeverity",
    "create_validation_issue",
    "validate_memgraph_data",
]
