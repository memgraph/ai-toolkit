"""
Base strategy interface for Hypothetical Graph Modeling (HyGM).

This module defines the base interface that all modeling strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.graph_models import GraphModel


class BaseModelingStrategy(ABC):
    """Base class for all graph modeling strategies."""

    @abstractmethod
    def create_model(self, database_structure: dict[str, Any], domain_context: str | None = None) -> "GraphModel":
        """
        Create a graph model from database structure.

        Args:
            database_structure: Analyzed database structure
            domain_context: Optional domain-specific context

        Returns:
            GraphModel: Generated graph model
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        pass
