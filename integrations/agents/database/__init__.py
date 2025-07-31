"""
Database analysis and abstraction layer.

This package provides database-agnostic interfaces and implementations
for analyzing SQL database structures.
"""

import sys
from pathlib import Path

# Add agents root to path for absolute imports
sys.path.append(str(Path(__file__).parent.parent))

from database.interface import (
    DatabaseAnalyzer,
    ColumnInfo,
    ForeignKeyInfo,
    TableInfo,
    TableType,
    RelationshipInfo,
    DatabaseStructure,
)
from database.factory import DatabaseAnalyzerFactory
from database.hygm_adapter import DatabaseDataInterface

__all__ = [
    "DatabaseAnalyzer",
    "ColumnInfo",
    "ForeignKeyInfo",
    "TableInfo",
    "TableType",
    "RelationshipInfo",
    "DatabaseStructure",
    "DatabaseAnalyzerFactory",
    "DatabaseDataInterface",
]
