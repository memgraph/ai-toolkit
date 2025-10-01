"""
Base validation interfaces and common types for HyGM validation system.

This module provides common validation interfaces, result types, and utilities
that are shared between pre-migration and post-migration validation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationCategory(Enum):
    """Categories of validation issues."""

    COVERAGE = "coverage"
    STRUCTURE = "structure"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"
    DATA_INTEGRITY = "data_integrity"
    SCHEMA_MISMATCH = "schema_mismatch"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during schema validation."""

    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    expected: Any = None
    actual: Any = None
    recommendation: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationMetrics:
    """Metrics collected during validation."""

    tables_covered: int = 0
    tables_total: int = 0
    properties_covered: int = 0
    properties_total: int = 0
    relationships_covered: int = 0
    relationships_total: int = 0
    indexes_covered: int = 0
    indexes_total: int = 0
    constraints_covered: int = 0
    constraints_total: int = 0
    coverage_percentage: float = 0.0

    def calculate_coverage(self):
        """Calculate overall coverage percentage."""
        if self.tables_total == 0:
            self.coverage_percentage = 0.0
        else:
            self.coverage_percentage = (self.tables_covered / self.tables_total) * 100


@dataclass
class ValidationResult:
    """Base result class for all validation types."""

    validation_type: str
    success: bool
    summary: str
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get only critical issues."""
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.CRITICAL
        ]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning issues."""
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]

    @property
    def info_issues(self) -> List[ValidationIssue]:
        """Get only info issues."""
        return [
            issue for issue in self.issues if issue.severity == ValidationSeverity.INFO
        ]

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)

        # Update success status based on critical issues
        if issue.severity == ValidationSeverity.CRITICAL:
            self.success = False


class BaseValidator(ABC):
    """Abstract base class for all validators."""

    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.metrics = ValidationMetrics()

    @abstractmethod
    def validate(self, *args, **kwargs) -> ValidationResult:
        """Perform validation and return results."""
        pass

    def add_issue(
        self,
        severity: ValidationSeverity,
        category: ValidationCategory,
        message: str,
        expected: Any = None,
        actual: Any = None,
        recommendation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Helper method to add validation issues."""
        issue = ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            expected=expected,
            actual=actual,
            recommendation=recommendation,
            details=details or {},
        )
        self.issues.append(issue)
        logger.debug("Added %s issue: %s", severity.value, message)

    def reset(self):
        """Reset validator state for new validation."""
        self.issues = []
        self.metrics = ValidationMetrics()

    def _generate_summary(self) -> str:
        """Generate a summary of validation results."""
        critical_count = len(
            [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
        )
        warning_count = len(
            [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
        )

        if critical_count > 0:
            return (
                f"Validation FAILED: {critical_count} critical issues, "
                f"{warning_count} warnings"
            )
        elif warning_count > 0:
            return f"Validation PASSED with {warning_count} warnings"
        else:
            return "Validation PASSED: No issues found"


def create_validation_issue(
    severity: Union[str, ValidationSeverity],
    category: Union[str, ValidationCategory],
    message: str,
    **kwargs,
) -> ValidationIssue:
    """
    Helper function to create validation issues with string or enum inputs.
    """
    if isinstance(severity, str):
        severity = ValidationSeverity(severity.upper())
    if isinstance(category, str):
        category = ValidationCategory(category.lower())

    return ValidationIssue(
        severity=severity, category=category, message=message, **kwargs
    )
