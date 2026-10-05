"""Validation issues and the model's own hard gate.

The issue types are lifted from agents/sql2graph's HyGM
(core/hygm/validation/base.py) in the same shape, so a later consolidation
there is a type swap rather than a rewrite. Its SQL-coverage metrics are left
behind.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import HygmModel

#: The node label for the person a conversation is with. A mention the
#: extractor types `User` may be a third party, which the mention resolver
#: re-types to `Person` (#358); that re-typed edge must still conform.
USER_LABEL = "User"
PERSON_LABEL = "Person"


class ValidationSeverity(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationCategory(Enum):
    COVERAGE = "coverage"
    STRUCTURE = "structure"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"
    DATA_INTEGRITY = "data_integrity"
    SCHEMA_MISMATCH = "schema_mismatch"


@dataclass
class ValidationIssue:
    """One problem found in a model or in data checked against one."""

    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    expected: Any = None
    actual: Any = None
    recommendation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """The issues one validation pass found. `success` is False once any issue is CRITICAL."""

    validation_type: str
    success: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.CRITICAL:
            self.success = False

    def summary(self) -> str:
        if not self.issues:
            return f"{self.validation_type}: no issues"
        return f"{self.validation_type}: {len(self.critical_issues)} critical, {len(self.warnings)} warnings"


def _critical(category: ValidationCategory, message: str, **kwargs: Any) -> ValidationIssue:
    return ValidationIssue(ValidationSeverity.CRITICAL, category, message, **kwargs)


def validate_model(model: "HygmModel") -> ValidationResult:
    """The hard gate every model must pass before anything extracts against it.

    Checks that labels are unique, that every relation endpoint names a
    declared node type, and that `User` never appears in an endpoint without
    `Person` beside it (#358). Identifier syntax and `identity` values are
    enforced when the types are constructed.

    Returns:
        A ValidationResult; `success` is False if any issue is CRITICAL.
    """
    result = ValidationResult("model")
    declared = model.node_labels()
    for kind, labels in (("node label", declared), ("relation type", model.relation_labels())):
        for label in sorted({label for label in labels if labels.count(label) > 1}):
            result.add_issue(_critical(ValidationCategory.CONSISTENCY, f"Duplicate {kind} {label!r}"))

    for relation in model.relation_types:
        for side, labels in (("start_labels", relation.start_labels), ("end_labels", relation.end_labels)):
            for label in labels:
                if label not in declared:
                    result.add_issue(
                        _critical(
                            ValidationCategory.STRUCTURE,
                            f"Relation type {relation.label!r}: {side} names undeclared node type {label!r}",
                            expected=declared,
                            actual=label,
                        )
                    )
            resolved = labels or declared
            if USER_LABEL in resolved and PERSON_LABEL not in resolved:
                result.add_issue(
                    _critical(
                        ValidationCategory.STRUCTURE,
                        f"Relation type {relation.label!r}: {side} includes {USER_LABEL!r} but not {PERSON_LABEL!r}",
                        recommendation=(
                            f"A {USER_LABEL} mention that is not the user is re-typed to {PERSON_LABEL}, "
                            "so every endpoint that accepts one must accept the other"
                        ),
                    )
                )
    return result
