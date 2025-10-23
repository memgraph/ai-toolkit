"""
Database analysis package interface.

This module provides the main entry point for database analysis functionality,
importing and exposing all the necessary components.
"""

# Core data models
from .models import (
    TableType,
    ColumnInfo,
    ForeignKeyInfo,
    TableInfo,
    RelationshipInfo,
    DatabaseStructure,
)

# Base interfaces
from .analyzer import DatabaseAnalyzer

# Factory (imported from existing factory module)
from .factory import DatabaseAnalyzerFactory

__all__ = [
    # Data models
    "TableType",
    "ColumnInfo",
    "ForeignKeyInfo",
    "TableInfo",
    "RelationshipInfo",
    "DatabaseStructure",
    # Interfaces
    "DatabaseAnalyzer",
    # Factory
    "DatabaseAnalyzerFactory",
]
