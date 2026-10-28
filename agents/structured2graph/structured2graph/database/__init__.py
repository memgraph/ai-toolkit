"""
Database analysis package interface.

This module provides the main entry point for database analysis functionality,
importing and exposing all the necessary components.
"""

# Core data models
# Base interfaces
from .analyzer import DatabaseAnalyzer

# Factory (imported from existing factory module)
from .factory import DatabaseAnalyzerFactory
from .models import (
    ColumnInfo,
    DatabaseStructure,
    ForeignKeyInfo,
    RelationshipInfo,
    TableInfo,
    TableType,
)

__all__ = [
    "ColumnInfo",
    # Interfaces
    "DatabaseAnalyzer",
    # Factory
    "DatabaseAnalyzerFactory",
    "DatabaseStructure",
    "ForeignKeyInfo",
    "RelationshipInfo",
    "TableInfo",
    # Data models
    "TableType",
]
