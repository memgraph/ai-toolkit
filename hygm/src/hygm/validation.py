"""Validation issues and the model's own hard gate.

The issue types are lifted from agents/sql2graph's HyGM
(core/hygm/validation/base.py) in the same shape, so a later consolidation
there is a type swap rather than a rewrite. Its SQL-coverage metrics are left
behind.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .facets import FacetConflictError, merge, narrowing_problem

if TYPE_CHECKING:
    from .types import HygmModel, NodeType, Property, RelationRestriction, RelationType


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
    cycle; a redeclared property only narrows what it inherits, and one
    inherited from several parents leaves some value all of them allow
    (`property_conflict`, 4.15); every key names required, single-valued
    properties (`count_conflict` for a list) and no type has two keys; `disjoint_with` names
    declared node types (`unknown_type`), no type is disjoint with itself or
    an ancestor, and no two disjoint types share a descendant
    (`disjoint_conflict`); no synonym equals, ignoring case, another type's
    label or synonym, nor another property's name or synonym on the same
    type (`duplicate_name`); `inverse_of` and restrictions name declared
    relation types (`unknown_type`); inverse relations have swapped
    endpoints, a relation has at most one inverse, and a symmetric one equal
    endpoints (`inverse_conflict`); a relation restriction names a relation
    its type may start, at most once, and only narrows its end labels and
    counts, as do the restrictions a type inherits from several parents
    (`relation_conflict`). A profile's own rules are
    its owner's to check (docs/contract.md, 5.6). Identifier syntax, property
    types, `values` and one property's own facets (`count_conflict`,
    `invalid_value`, 4.14) are enforced when the types are constructed,
    profile settings when a model file is read.

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
    _check_disjoint_declared(model, result)
    _check_inverses(model, result)
    if not any(issue.code in {"unknown_parent", "parent_cycle", "unknown_type"} for issue in result.issues):
        _check_inheritance(model, result)
        _check_disjointness(model, result)
        _check_restrictions(model, result)
    _check_synonyms(model, result)

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


def _check_disjoint_declared(model: "HygmModel", result: ValidationResult) -> None:
    declared = set(model.node_labels())
    for node in model.node_types:
        for other in node.disjoint_with:
            if other not in declared:
                result.add_issue(
                    _critical(
                        "unknown_type",
                        ValidationCategory.STRUCTURE,
                        f"Node type {node.label!r}: disjoint_with names undeclared node type {other!r}",
                        actual=other,
                    )
                )


def _check_inverses(model: "HygmModel", result: ValidationResult) -> None:
    """Inverse relations swap their endpoints and pair up one to one; a symmetric relation's are equal (4.16)."""

    def conflict(message: str) -> None:
        result.add_issue(_critical("inverse_conflict", ValidationCategory.CONSISTENCY, message))

    claimed: dict[str, list[str]] = {}
    for relation in model.relation_types:
        if relation.symmetric and set(relation.start_labels) != set(relation.end_labels):
            conflict(f"Relation type {relation.label!r} is symmetric, so its start and end labels must be equal")
        if relation.inverse_of is None:
            continue
        other = model.relation_type(relation.inverse_of)
        if other is None:
            result.add_issue(
                _critical(
                    "unknown_type",
                    ValidationCategory.STRUCTURE,
                    f"Relation type {relation.label!r}: inverse_of names undeclared relation type {relation.inverse_of!r}",
                )
            )
            continue
        claimed.setdefault(other.label, []).append(relation.label)
        if set(relation.start_labels) != set(other.end_labels) or set(relation.end_labels) != set(other.start_labels):
            conflict(
                f"Relation types {relation.label!r} and {other.label!r} are inverses, so one's start labels "
                "must be the other's end labels"
            )
        if other.inverse_of not in (None, relation.label):
            conflict(f"Relation type {relation.label!r} is {other.label!r}'s inverse, which names another inverse")
    for label, claims in claimed.items():
        if len(claims) > 1:
            conflict(f"Relation type {label!r} has more than one inverse: {sorted(set(claims))}")


def _check_restrictions(model: "HygmModel", result: ValidationResult) -> None:
    """A type's relation restrictions only narrow what its instances inherit (4.16)."""

    def conflict(message: str) -> None:
        result.add_issue(_critical("relation_conflict", ValidationCategory.CONSISTENCY, message))

    declared = set(model.node_labels())
    for node in model.node_types:
        relations = [r.relation for r in node.relation_restrictions]
        for name in sorted({name for name in relations if relations.count(name) > 1}):
            conflict(f"Node type {node.label!r} restricts relation type {name!r} more than once")
        for restriction in node.relation_restrictions:
            where = f"Node type {node.label!r}: the restriction of {restriction.relation!r}"
            relation = model.relation_type(restriction.relation)
            if relation is None:
                result.add_issue(
                    _critical(
                        "unknown_type", ValidationCategory.STRUCTURE, f"{where} names an undeclared relation type"
                    )
                )
                continue
            if not model.may_start(node.label, relation):
                conflict(f"{where} narrows a relation its instances can't start")
                continue
            undeclared = [label for label in restriction.end_labels or () if label not in declared]
            for label in undeclared:
                result.add_issue(
                    _critical(
                        "undeclared_endpoint",
                        ValidationCategory.STRUCTURE,
                        f"{where} names undeclared node type {label!r}",
                    )
                )
            if undeclared:
                continue
            problem = _restriction_problem(model, node, relation, restriction)
            if problem:
                conflict(f"{where} {problem}: a restriction may only narrow")
    for node in model.node_types:
        for relation in model.relation_types:
            problem = _parents_disagree(model, node, relation)
            if problem:
                conflict(f"Node type {node.label!r} inherits restrictions of {relation.label!r} {problem}")


def _inherited_relation(model: "HygmModel", node: "NodeType", relation: "RelationType") -> "RelationType":
    inherited = relation
    for parent in node.parents:
        from_parent = model.relation_for(parent, relation.label)
        if from_parent is not None:
            inherited = model.narrowed(inherited, from_parent.end_labels, from_parent.min_count, from_parent.max_count)
    return inherited


def _restriction_problem(
    model: "HygmModel", node: "NodeType", relation: "RelationType", restriction: "RelationRestriction"
) -> str | None:
    inherited = _inherited_relation(model, node, relation)
    if restriction.end_labels is not None and inherited.end_labels:
        wider = [e for e in restriction.end_labels if not any(model.is_a(e, x) for x in inherited.end_labels)]
        if wider:
            return f"ends at {wider}, beyond the inherited {list(inherited.end_labels)}"
    if restriction.end_labels == ():
        return "has an empty end_labels, which reads as any label"
    if restriction.min_count is not None and restriction.min_count < inherited.min_count:
        return f"lowers min_count below {inherited.min_count}"
    if (
        restriction.max_count is not None
        and inherited.max_count is not None
        and restriction.max_count > inherited.max_count
    ):
        return f"raises max_count above {inherited.max_count}"
    least = max(inherited.min_count, restriction.min_count or 0)
    limits = [m for m in (inherited.max_count, restriction.max_count) if m is not None]
    if limits and least > min(limits):
        return f"leaves no count: at least {least} and at most {min(limits)}"
    return None


def _parents_disagree(model: "HygmModel", node: "NodeType", relation: "RelationType") -> str | None:
    """Why the restrictions `node` inherits from its parents leave no end label or count, or None."""
    narrowed = [model.relation_for(parent, relation.label) for parent in node.parents]
    narrowed = [r for r in narrowed if r is not None and r != relation]
    if len(narrowed) < 2:
        return None
    for i, a in enumerate(narrowed):
        for b in narrowed[i + 1 :]:
            if (
                a.end_labels
                and b.end_labels
                and not any(model.is_a(x, y) or model.is_a(y, x) for x in a.end_labels for y in b.end_labels)
            ):
                return f"from several parents with no end label in common ({list(a.end_labels)}, {list(b.end_labels)})"
    least = max(r.min_count for r in narrowed)
    limits = [r.max_count for r in narrowed if r.max_count is not None]
    if limits and least > min(limits):
        return f"from several parents with no count in common: at least {least} and at most {min(limits)}"
    return None


def _check_disjointness(model: "HygmModel", result: ValidationResult) -> None:
    """Disjoint types share no instance, so neither may be the other's ancestor nor share a descendant."""
    for node in model.node_types:
        for other in node.disjoint_with:
            if model.is_a(node.label, other) or model.is_a(other, node.label):
                result.add_issue(
                    _critical(
                        "disjoint_conflict",
                        ValidationCategory.CONSISTENCY,
                        f"Node type {node.label!r} is disjoint with {other!r}, its own ancestor, descendant or self",
                    )
                )
                continue
            shared = [
                t.label for t in model.node_types if model.is_a(t.label, node.label) and model.is_a(t.label, other)
            ]
            if shared:
                result.add_issue(
                    _critical(
                        "disjoint_conflict",
                        ValidationCategory.CONSISTENCY,
                        f"Node types {node.label!r} and {other!r} are disjoint but share the descendant {shared[0]!r}",
                    )
                )


def _check_synonyms(model: "HygmModel", result: ValidationResult) -> None:
    """One concept, one name: a synonym can't name another type, or another property of the same type."""
    _check_names_among("node type", model.node_types, result)
    _check_names_among("relation type", model.relation_types, result)
    for owner in model.node_types:
        _check_names_among(f"property of {owner.label!r}", _distinct(model.properties_of(owner.label)), result)
    for relation in model.relation_types:
        _check_names_among(f"property of {relation.label!r}", _distinct(relation.properties), result)


def _distinct(properties: "tuple[Property, ...]") -> "tuple[Property, ...]":
    return tuple({p.name: p for p in properties}.values())


def _check_names_among(kind: str, named: Any, result: ValidationResult) -> None:
    # Two labels differing only in case are not this rule's: it is about synonyms, and older models may have them.
    owners: dict[str, dict[str, bool]] = {}
    for item in named:
        identifier = getattr(item, "label", None) or item.name
        owners.setdefault(identifier.casefold(), {}).setdefault(identifier, False)
        for synonym in item.synonyms:
            owners.setdefault(synonym.casefold(), {})[identifier] = True
    for name, holders in owners.items():
        if len(holders) > 1 and any(holders.values()):
            result.add_issue(
                _critical(
                    "duplicate_name",
                    ValidationCategory.CONSISTENCY,
                    f"The name {name!r} is the label or a synonym of more than one {kind}: {sorted(holders)}",
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
            elif prop.max_count != 1:
                result.add_issue(
                    _critical(
                        "count_conflict",
                        ValidationCategory.STRUCTURE,
                        f"Node type {node.label!r}: key property {name!r} must be single-valued (max_count 1)",
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
    """Facets inherited from several parents must leave some value (merge), and a redeclaration only narrows (4.15)."""
    inherited: dict[str, tuple[str, Property]] = {}
    for parent in node.parents:
        for prop in model.properties_of(parent):
            seen = inherited.get(prop.name)
            if seen is None or seen[1] == prop:
                inherited.setdefault(prop.name, (parent, prop))
                continue
            try:
                inherited[prop.name] = (seen[0], merge(seen[1], prop))
            except FacetConflictError as e:
                result.add_issue(
                    _critical(
                        "property_conflict",
                        ValidationCategory.CONSISTENCY,
                        f"Node type {node.label!r} inherits property {prop.name!r} from {seen[0]!r} and "
                        f"{parent!r}, and no value satisfies both: {e}",
                    )
                )
    for prop in node.properties:
        seen = inherited.get(prop.name)
        problem = narrowing_problem(seen[1], prop) if seen else None
        if seen and problem:
            result.add_issue(
                _critical(
                    "property_conflict",
                    ValidationCategory.CONSISTENCY,
                    f"Node type {node.label!r} redeclares property {prop.name!r}, inherited from {seen[0]!r}, "
                    f"and {problem}: a redeclaration may only narrow it",
                )
            )
