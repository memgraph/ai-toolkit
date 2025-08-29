"""
User operation tracking for preserving user intentions in graph modeling.

This module tracks user operations to ensure they are preserved when
LLM strategies regenerate models for validation fixes.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .operations import OperationType
else:
    try:
        from .operations import OperationType
    except ImportError:
        # Fallback for circular imports
        pass


class UserOperationRecord:
    """Record of a user operation."""

    def __init__(self, operation: "OperationType"):
        self.operation = operation


class UserOperationHistory:
    """Container for tracking all user operations in a session."""

    def __init__(self, session_id: str):
        self.operations: List[UserOperationRecord] = []
        self.session_id = session_id

    def add_operation(self, operation: "OperationType") -> None:
        """Add a new user operation to the history."""
        record = UserOperationRecord(operation)
        self.operations.append(record)

    def get_operations_by_type(self, operation_type: str) -> List[UserOperationRecord]:
        """Get all operations of a specific type."""
        return [
            op
            for op in self.operations
            if op.operation.operation_type == operation_type
        ]

    def has_label_changes(self) -> bool:
        """Check if user has made any label changes."""
        return len(self.get_operations_by_type("change_node_label")) > 0

    def to_llm_context(self) -> str:
        """Generate context string for LLM about user operations."""
        if not self.operations:
            return ""

        context_parts = ["USER CHANGES TO PRESERVE:", ""]

        for i, record in enumerate(self.operations, 1):
            op = record.operation

            if op.operation_type == "change_node_label":
                old_label = getattr(op, "old_label", "unknown")
                new_label = getattr(op, "new_label", "unknown")
                context_parts.append(f"{i}. Node label: '{old_label}' → '{new_label}'")
            elif op.operation_type == "rename_property":
                node_label = getattr(op, "node_label", "unknown")
                old_prop = getattr(op, "old_property", "unknown")
                new_prop = getattr(op, "new_property", "unknown")
                context_parts.append(
                    f"{i}. Property: {node_label}.{old_prop} → {new_prop}"
                )
            elif op.operation_type == "drop_property":
                node_label = getattr(op, "node_label", "unknown")
                prop_name = getattr(op, "property_name", "unknown")
                context_parts.append(f"{i}. Remove: {node_label}.{prop_name}")
            elif op.operation_type == "add_property":
                node_label = getattr(op, "node_label", "unknown")
                prop_name = getattr(op, "property_name", "unknown")
                context_parts.append(f"{i}. Add: {node_label}.{prop_name}")

        context_parts.extend(
            ["", "PRESERVE these user changes exactly when improving the model."]
        )

        return "\n".join(context_parts)
