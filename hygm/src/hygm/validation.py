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
    from .types import HygmModel, NodeType, Property

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
    """One problem found in a model or in data checked against one.

    `code` is the contract's error code (docs/contract.md, section 9) for an
    issue in a model; issues about extracted data have none.
    """

    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    expected: Any = None
    actual: Any = None
    recommendation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    code: str | None = None


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


def _critical(code: str, category: ValidationCategory, message: str, **kwargs: Any) -> ValidationIssue:
    return ValidationIssue(ValidationSeverity.CRITICAL, category, message, code=code, **kwargs)


def validate_model(model: "HygmModel") -> ValidationResult:
    """The hard gate every model must pass before anything extracts against it.

    Checks, each issue carrying its contract error code (docs/contract.md,
    section 9): labels and property names are unique; `parents` and every
    relation endpoint name declared node types; the parent graph has no
    cycle; redeclared and doubly inherited properties agree; every key names
    required properties and no type has two keys; and `User` never appears in
    an endpoint without `Person` beside it (#358). Identifier syntax, property
    types, `values` and `identity` are enforced when the types are constructed.

    Returns:
        A ValidationResult; `success` is False if any issue is CRITICAL.
    """
    result = ValidationResult("model")
    declared = model.node_labels()
    for kind, labels in (("node label", declared), ("relation type", model.relation_labels())):
        for label in sorted({label for label in labels if labels.count(label) > 1}):
            result.add_issue(
                _critical("duplicate_label", ValidationCategory.CONSISTENCY, f"Duplicate {kind} {label!r}")
            )

    for owner in (*model.node_types, *model.relation_types):
        names = [p.name for p in owner.properties]
        for name in sorted({name for name in names if names.count(name) > 1}):
            result.add_issue(
                _critical(
                    "duplicate_property",
                    ValidationCategory.CONSISTENCY,
                    f"Type {owner.label!r} declares property {name!r} twice",
                )
            )

    _check_parents(model, result)
    if not any(issue.code in {"unknown_parent", "parent_cycle"} for issue in result.issues):
        _check_inheritance(model, result)

    for relation in model.relation_types:
        for side, labels in (("start_labels", relation.start_labels), ("end_labels", relation.end_labels)):
            for label in labels:
                if label not in declared:
                    result.add_issue(
                        _critical(
                            "undeclared_endpoint",
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
                        "profile_rule",
                        ValidationCategory.STRUCTURE,
                        f"Relation type {relation.label!r}: {side} includes {USER_LABEL!r} but not {PERSON_LABEL!r}",
                        recommendation=(
                            f"A {USER_LABEL} mention that is not the user is re-typed to {PERSON_LABEL}, "
                            "so every endpoint that accepts one must accept the other"
                        ),
                    )
                )
    return result


def _check_parents(model: "HygmModel", result: ValidationResult) -> None:
    declared = set(model.node_labels())
    for node in model.node_types:
        for parent in node.parents:
            if parent not in declared:
                result.add_issue(
                    _critical(
                        "unknown_parent",
                        ValidationCategory.STRUCTURE,
                        f"Node type {node.label!r}: parents names undeclared node type {parent!r}",
                        actual=parent,
                    )
                )
    in_cycle = sorted(node.label for node in model.node_types if _reaches(model, node.label, node.label))
    if in_cycle:
        result.add_issue(
            _critical(
                "parent_cycle",
                ValidationCategory.STRUCTURE,
                f"Node types {in_cycle} are their own ancestors: parents must not form a cycle",
            )
        )


def _reaches(model: "HygmModel", start: str, target: str) -> bool:
    seen: set[str] = set()
    frontier = list(getattr(model.node_type(start), "parents", ()))
    while frontier:
        label = frontier.pop()
        if label == target:
            return True
        if label not in seen:
            seen.add(label)
            frontier.extend(getattr(model.node_type(label), "parents", ()))
    return False


def _check_inheritance(model: "HygmModel", result: ValidationResult) -> None:
    """Property agreement and keys; only meaningful once the parent graph is a DAG of declared types."""
    for node in model.node_types:
        _check_property_agreement(model, node, result)
        effective = {p.name: p for p in model.properties_of(node.label)}
        for name in node.key:
            prop = effective.get(name)
            if prop is None:
                result.add_issue(
                    _critical(
                        "unknown_key_property",
                        ValidationCategory.STRUCTURE,
                        f"Node type {node.label!r}: key names {name!r}, which it neither declares nor inherits",
                    )
                )
            elif not prop.required:
                result.add_issue(
                    _critical(
                        "key_not_required",
                        ValidationCategory.STRUCTURE,
                        f"Node type {node.label!r}: key property {name!r} must be required",
                    )
                )
        if _key_holders(model, node.label) > 1 and all(_key_holders(model, p) <= 1 for p in node.parents):
            result.add_issue(
                _critical(
                    "two_keys",
                    ValidationCategory.STRUCTURE,
                    f"Node type {node.label!r} has more than one key, counting its own and its ancestors'",
                )
            )


def _key_holders(model: "HygmModel", label: str) -> int:
    return sum(1 for owner in (label, *model.ancestors(label)) if getattr(model.node_type(owner), "key", ()))


def _check_property_agreement(model: "HygmModel", node: "NodeType", result: ValidationResult) -> None:
    inherited: dict[str, tuple[str, Property]] = {}
    for parent in node.parents:
        for prop in model.properties_of(parent):
            seen = inherited.get(prop.name)
            if seen and not _agree(seen[1], prop):
                result.add_issue(
                    _critical(
                        "property_conflict",
                        ValidationCategory.CONSISTENCY,
                        f"Node type {node.label!r} inherits property {prop.name!r} from {seen[0]!r} and "
                        f"{parent!r} with different types or values",
                    )
                )
            inherited.setdefault(prop.name, (parent, prop))
    for prop in node.properties:
        seen = inherited.get(prop.name)
        if seen and not _agree(seen[1], prop):
            result.add_issue(
                _critical(
                    "property_conflict",
                    ValidationCategory.CONSISTENCY,
                    f"Node type {node.label!r} redeclares property {prop.name!r}, inherited from {seen[0]!r}, "
                    "with a different type or values",
                )
            )


def _agree(a: "Property", b: "Property") -> bool:
    return a.type == b.type and a.values == b.values
