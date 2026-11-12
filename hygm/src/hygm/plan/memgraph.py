"""The Memgraph plan: the constraints, indexes and enums a model implies (docs/contract.md, 7.1).

A plan is derived from a model, never stored. Each fact is constrained once,
on the label of the type that declares it: a `Customer` node also carries the
`Party` label (4.7), so a key, required property or type declared on `Party`
is enforced on customers through `Party`'s constraint, and repeating it on
`Customer` would only add a redundant constraint.

| Model | Plan |
|---|---|
| a type's `key` | a uniqueness constraint (composite for several properties) and a label-property index |
| `required: true` | an existence constraint |
| a property's `type` | a data type constraint, where Memgraph has a matching type (`MEMGRAPH_TYPES`) |
| `values` | an enum named `<Label>_<property>`, and the property typed `ENUM` |
| a multi-valued property (`max_count` above 1 or none) | a `LIST` data type constraint; Memgraph has no per-element type, so the elements' type and enum go unchecked |
| `parents` | no DDL: writers set every label `labels_for` returns |
| a subtype refining an inherited property (4.15) | what the refinement adds, on the subtype's label: an existence constraint when it makes the property required; the type, enum and list constraint stay on the declaring ancestor's label, which the subtype's nodes carry |
| `min_count` above 1, `max_count`, bounds, `value`, `default` | nothing: Memgraph can't enforce them; consumers that write data check them |
| `abstract`, `disjoint_with`, relation counts, `inverse_of`, `symmetric`, relation restrictions | nothing: Memgraph has no constraint for them; consumers that write data check them |

Memgraph can only type a property as `ENUM`, not as one named enum, so the
enum constraint accepts a value of any enum; writers store
`ToEnum(enum_name, value)` (see `enum_for`), and a plain string is rejected.
Memgraph has no constraints on relationship properties, so relation types add
nothing.

Security boundary: labels and property names are interpolated into the DDL
because Cypher can't parameterize them. That is safe only because hygm's types
refuse anything but a valid Cypher identifier when they are constructed
(docs/contract.md, 4.10); they are backtick-quoted as well, so an identifier
that is also a Cypher keyword (`Order`, `MATCH`) still parses. Enum values are
arbitrary strings, so each is backtick-quoted with backticks doubled, Cypher's
only escape inside a quoted name. DDL takes no parameters, so there is no
safer channel for them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..errors import ModelError
from ..validation import validate_model

if TYPE_CHECKING:
    from ..types import HygmModel, NodeType, Property, PropertyType

StatementKind = Literal["enum", "unique", "index", "exists", "typed"]

#: The Memgraph data type each logical property type is constrained to (4.6).
#: `datetime` has none: a value may be a LocalDateTime or a ZonedDateTime, and
#: Memgraph allows one type constraint per label and property, so it stays
#: unconstrained rather than reject half the valid values. A string property
#: with `values` is typed `ENUM` instead of `STRING`.
MEMGRAPH_TYPES: Mapping[PropertyType, str | None] = {
    "string": "STRING",
    "integer": "INTEGER",
    "float": "FLOAT",
    "decimal": "FLOAT",
    "boolean": "BOOLEAN",
    "date": "DATE",
    "datetime": None,
    "json": "MAP",
}

#: Runs one Cypher statement and returns its rows; the caller's Memgraph client.
Run = Callable[[str], Iterable[Mapping[str, Any]]]


def _name(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


@dataclass(frozen=True)
class Statement:
    """One DDL statement of a plan.

    Attributes:
        kind: `enum`, `unique` (uniqueness constraint), `index`
            (label-property index), `exists` (existence constraint) or `typed`
            (data type constraint).
        target: The node label it applies to, or the enum's name.
        properties: The property names it covers, in key order; empty for an enum.
        data_type: The Memgraph type of a `typed` statement, else None.
        values: The values of an `enum` statement, else empty.
    """

    kind: StatementKind
    target: str
    properties: tuple[str, ...] = ()
    data_type: str | None = None
    values: tuple[str, ...] = ()

    @property
    def cypher(self) -> str:
        """The statement as Cypher Memgraph accepts."""
        label = _name(self.target)
        if self.kind == "enum":
            return f"CREATE ENUM {label} VALUES {{ {', '.join(_name(v) for v in self.values)} }}"
        if self.kind == "index":
            return f"CREATE INDEX ON :{label}({', '.join(_name(p) for p in self.properties)})"
        if self.kind == "unique":
            return f"CREATE CONSTRAINT ON (n:{label}) ASSERT {', '.join(f'n.{_name(p)}' for p in self.properties)} IS UNIQUE"
        if self.kind == "exists":
            return f"CREATE CONSTRAINT ON (n:{label}) ASSERT EXISTS (n.{_name(self.properties[0])})"
        return f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{_name(self.properties[0])} IS TYPED {self.data_type}"


@dataclass(frozen=True)
class MemgraphPlan:
    """What a Memgraph needs to hold a model: its statements, in model declaration order."""

    statements: tuple[Statement, ...]

    def cypher(self) -> tuple[str, ...]:
        """Every statement as Cypher, in order."""
        return tuple(statement.cypher for statement in self.statements)


class PlanConflictError(ValueError):
    """The database already holds a constraint or enum that contradicts the plan.

    Attributes:
        conflicts: One message per contradiction.
    """

    def __init__(self, conflicts: Iterable[str]) -> None:
        self.conflicts = tuple(conflicts)
        super().__init__("; ".join(self.conflicts))


def _require_valid(model: HygmModel) -> None:
    critical = validate_model(model).critical_issues
    if critical:
        raise ModelError(
            [issue.code or "invalid_format" for issue in critical],
            "; ".join(issue.message for issue in critical),
        )


def _require_declared(model: HygmModel, label: str) -> None:
    if model.node_type(label) is None:
        raise ValueError(f"{label!r} is not a node type of the model")


def labels_for(model: HygmModel, label: str) -> tuple[str, ...]:
    """The labels a node of type `label` carries: the type, then its ancestors nearest first (4.7).

    Raises:
        ValueError: if `label` is not a declared node type.
    """
    _require_declared(model, label)
    return (label, *model.ancestors(label))


def _declaring_types(model: HygmModel, label: str, name: str) -> tuple[str, ...]:
    """The topmost types in `label`'s hierarchy that declare property `name`: where it is constrained."""
    owners = [owner for owner in (*reversed(model.ancestors(label)), label) if _own(model, owner, name)]
    return tuple(o for o in owners if not any(_has(model, a, name) for a in model.ancestors(o)))


def _own(model: HygmModel, label: str, name: str) -> Property | None:
    node = model.node_type(label)
    return next((p for p in node.properties if p.name == name), None) if node else None


def _has(model: HygmModel, label: str, name: str) -> Property | None:
    return next((p for p in model.properties_of(label) if p.name == name), None)


def enum_name(label: str, prop: str) -> str:
    """The enum a `values` property declared on `label` gets: `<label>_<prop>`, a valid identifier."""
    return f"{label}_{prop}"


def enum_for(model: HygmModel, label: str, prop: str) -> str | None:
    """The enum a writer stores `label`'s `prop` as, or None if the property has no `values`.

    Under multiple inheritance two parents may each declare the property and
    own an enum; this is the first of them in `properties_of` order.

    Raises:
        ValueError: if `label` is not a declared node type.
    """
    _require_declared(model, label)
    found = _has(model, label, prop)
    if found is None or found.values is None:
        return None
    return enum_name(_declaring_types(model, label, prop)[0], prop)


def _type_statements(model: HygmModel, node: NodeType) -> list[Statement]:
    label = node.label
    statements: list[Statement] = []
    if node.key:
        statements.append(Statement("unique", label, node.key))
        statements.append(Statement("index", label, node.key))
    ancestors = model.ancestors(label)
    inherited = {p.name for a in ancestors for p in model.properties_of(a)}
    inherited_required = {p.name for a in ancestors for p in model.properties_of(a) if p.required}
    for prop in model.properties_of(label):
        if prop.required and prop.name not in inherited_required:
            statements.append(Statement("exists", label, (prop.name,)))
    for prop in node.properties:
        if prop.name in inherited:
            continue
        if prop.values is not None:
            if not prop.values:
                raise ModelError("invalid_format", f"{label}.{prop.name}: values must be a non-empty list")
            statements.append(Statement("enum", enum_name(label, prop.name), values=tuple(dict.fromkeys(prop.values))))
        if prop.multi_valued:
            statements.append(Statement("typed", label, (prop.name,), data_type="LIST"))
        elif prop.values is not None:
            statements.append(Statement("typed", label, (prop.name,), data_type="ENUM"))
        elif (data_type := MEMGRAPH_TYPES[prop.type]) is not None:
            statements.append(Statement("typed", label, (prop.name,), data_type=data_type))
    return statements


def memgraph_plan(model: HygmModel) -> MemgraphPlan:
    """The constraints, indexes and enums Memgraph needs to hold `model`.

    Deterministic: node types in declaration order, each with its key's
    uniqueness constraint and index, then existence constraints, then enums
    and data type constraints in property order. Each is placed on the label
    of the type that declares it (see the module docstring).

    Raises:
        ModelError: if `model` fails validate_model.
        ValueError: if two enums would share a name (`A_b` + `c` and `A` + `b_c`).
    """
    _require_valid(model)
    statements = [s for node in model.node_types for s in _type_statements(model, node)]
    enums = [s.target for s in statements if s.kind == "enum"]
    clashes = sorted({name for name in enums if enums.count(name) > 1})
    if clashes:
        raise ValueError(f"Enum names {clashes} come from more than one label and property; rename one of them")
    return MemgraphPlan(tuple(statements))


# The SHOW output spells some types differently from the DDL that creates them.
_SHOWN_TYPES = {"BOOL": "BOOLEAN"}


def _shown_type(value: str) -> str:
    compact = value.replace(" ", "").upper()
    return _SHOWN_TYPES.get(compact, compact)


def _properties(value: Any) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


@dataclass(frozen=True)
class _Existing:
    unique: frozenset[tuple[str, frozenset[str]]]
    index: frozenset[tuple[str, tuple[str, ...]]]
    exists: frozenset[tuple[str, str]]
    typed: Mapping[tuple[str, str], str]
    enums: Mapping[str, tuple[str, ...]]

    @classmethod
    def read(cls, run: Run) -> _Existing:
        unique, exists, typed = set(), set(), {}
        for row in run("SHOW CONSTRAINT INFO"):
            kind, label, props = row["constraint type"], row["label"], _properties(row["properties"])
            if kind == "unique":
                unique.add((label, frozenset(props)))
            elif kind == "exists":
                exists.add((label, props[0]))
            elif kind == "data_type":
                typed[(label, props[0])] = _shown_type(row["data_type"])
        index = {
            (row["label"], _properties(row["property"]))
            for row in run("SHOW INDEX INFO")
            if row["index type"] == "label+property"
        }
        enums = {row["Enum Name"]: tuple(row["Enum Values"]) for row in run("SHOW ENUMS")}
        return cls(frozenset(unique), frozenset(index), frozenset(exists), typed, enums)

    def holds(self, statement: Statement) -> bool:
        """Whether `statement`'s constraint, index or enum already exists, whatever its definition."""
        label, props = statement.target, statement.properties
        if statement.kind == "unique":
            return (label, frozenset(props)) in self.unique
        if statement.kind == "index":
            return (label, props) in self.index
        if statement.kind == "exists":
            return (label, props[0]) in self.exists
        if statement.kind == "typed":
            return (label, props[0]) in self.typed
        return label in self.enums

    def conflict(self, statement: Statement) -> str | None:
        """Why `statement` contradicts what exists, or None."""
        label, props = statement.target, statement.properties
        if statement.kind == "typed":
            shown = self.typed.get((label, props[0]))
            if shown is not None and shown != statement.data_type:
                return f":{label}({props[0]}) is already typed {shown}, the model says {statement.data_type}"
        if statement.kind == "enum":
            values = self.enums.get(label)
            if values is not None and values != statement.values:
                return f"enum {label} already has values {list(values)}, the model says {list(statement.values)}"
        return None


def apply_plan(plan: MemgraphPlan, run: Run) -> tuple[Statement, ...]:
    """Create whatever `plan` holds that the database doesn't yet; applying a plan twice is a no-op.

    Reads what exists (`SHOW CONSTRAINT INFO`, `SHOW INDEX INFO`, `SHOW
    ENUMS`) and runs only the missing statements. Reading first, rather than
    running everything and tolerating "already exists" errors, because
    Memgraph is inconsistent there: re-creating a uniqueness or existence
    constraint or an index is silently accepted, while re-creating a data type
    constraint or an enum is an error whose only distinguishing mark is its
    message text. The SHOW output also tells "already exists" apart from a
    contradiction, which an error message can't.

    Args:
        plan: The plan to apply.
        run: Runs one Cypher statement and returns its rows as mappings, e.g.
            `lambda q: [r.data() for r in session.run(q)]`. hygm needs no client.

    Returns:
        The statements that were run.

    Raises:
        PlanConflictError: before running anything, if a property is already
            typed differently or an enum of the same name has other values.
            Changing either is model evolution, not a plan.
    """
    existing = _Existing.read(run)
    conflicts = [c for c in (existing.conflict(s) for s in plan.statements) if c]
    if conflicts:
        raise PlanConflictError(conflicts)
    missing = tuple(s for s in plan.statements if not existing.holds(s))
    for statement in missing:
        list(run(statement.cypher))
    return missing


# TODO(#699): a mapping's `join` relation adds a label-property index on each
# endpoint's join property; it lands with the mapping layer.
