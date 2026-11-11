"""The operations between two models (docs/contract.md, section 11)."""

from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ModelError
from ..types import HygmModel, NodeType, Property, RelationType
from .operations import (
    AddNodeType,
    AddProperty,
    AddRelationType,
    Kind,
    Operation,
    OrderNodeTypes,
    OrderProperties,
    OrderRelationTypes,
    ProfileLevel,
    RemoveNodeType,
    RemoveProperty,
    RemoveRelationType,
    SetDescription,
    SetEndpoints,
    SetIri,
    SetKey,
    SetParents,
    SetProfile,
    SetProperty,
)


def diff(a: HygmModel, b: HygmModel) -> tuple[Operation, ...]:
    """The operations that turn `a` into `b`: ``apply(a, diff(a, b)) == b``.

    A rename can't be told from a removal and an addition with the same
    definition, so diff never emits one: a renamed type or property comes
    out as remove plus add. A caller that knows better (a user's edit,
    derivation's merges) records its own operations instead.

    The list is meant to be applied whole, as apply() does, validating only
    the result: it removes what `b` lacks first, then adds and changes types,
    then fixes the order, and a step in between may name something a later
    step adds.

    Raises:
        ModelError: `unknown_operation` when `a` and `b` differ in a key no
            operation sets yet (the `ontology` block, `imports`, a node type's
            `abstract`, `disjoint_with`, `relation_restrictions`, a type's
            `synonyms` or `names`, a relation type's counts, `inverse_of` or
            `symmetric`),
            rather than leave the change out. A type added whole, or a
            property set whole, carries every key.
    """
    _require_expressible(a, b)
    ops: list[Operation] = []
    b_nodes = {t.label: t for t in b.node_types}
    b_relations = {r.label: r for r in b.relation_types}
    a_nodes = {t.label: t for t in a.node_types}
    a_relations = {r.label: r for r in a.relation_types}
    ops += [RemoveNodeType(t.label) for t in a.node_types if t.label not in b_nodes]
    ops += [RemoveRelationType(r.label) for r in a.relation_types if r.label not in b_relations]
    for node in b.node_types:
        ops += _node_changes(a_nodes[node.label], node) if node.label in a_nodes else [AddNodeType(node)]
    for relation in b.relation_types:
        before = a_relations.get(relation.label)
        ops += _relation_changes(before, relation) if before else [AddRelationType(relation)]
    ops += _profile_changes(a.profiles, b.profiles, "model", None)
    if _kept_order(a.node_labels(), b.node_labels()) != b.node_labels():
        ops.append(OrderNodeTypes(b.node_labels()))
    if _kept_order(a.relation_labels(), b.relation_labels()) != b.relation_labels():
        ops.append(OrderRelationTypes(b.relation_labels()))
    return tuple(ops)


# TODO(#731): operations that set these keys, so diff and apply cover the whole model (contract, section 11).
_MODEL_KEYS_WITHOUT_OPERATIONS = ("ontology", "imports")
_NODE_KEYS_WITHOUT_OPERATIONS = ("abstract", "disjoint_with", "synonyms", "names", "relation_restrictions")
_RELATION_KEYS_WITHOUT_OPERATIONS = ("synonyms", "names", "min_count", "max_count", "inverse_of", "symmetric")


def _unexpressible(key: str, where: str = "") -> ModelError:
    at = f" on {where}" if where else ""
    return ModelError("unknown_operation", f"diff can't express a change to {key!r}{at}: no operation sets it yet")


def _require_expressible(a: HygmModel, b: HygmModel) -> None:
    for key in _MODEL_KEYS_WITHOUT_OPERATIONS:
        if getattr(a, key) != getattr(b, key):
            raise _unexpressible(key)


def _require_expressible_type(a: NodeType | RelationType, b: NodeType | RelationType, keys: tuple[str, ...]) -> None:
    for key in keys:
        if getattr(a, key) != getattr(b, key):
            raise _unexpressible(key, repr(b.label))


def _kept_order(before: Sequence[str], after: Sequence[str]) -> tuple[str, ...]:
    """The order removals and appended additions leave, before any reordering."""
    return (*(label for label in before if label in after), *(label for label in after if label not in before))


def _node_changes(a: NodeType, b: NodeType) -> list[Operation]:
    _require_expressible_type(a, b, _NODE_KEYS_WITHOUT_OPERATIONS)
    ops: list[Operation] = []
    if a.description != b.description:
        ops.append(SetDescription("node", b.label, b.description))
    if a.iri != b.iri:
        ops.append(SetIri("node", b.label, b.iri))
    if a.parents != b.parents:
        ops.append(SetParents(b.label, b.parents))
    if a.key != b.key:
        ops.append(SetKey(b.label, b.key))
    ops += _profile_changes(_node_profiles(a), _node_profiles(b), "node", b.label)
    ops += _property_changes("node", b.label, a.properties, b.properties)
    return ops


def _relation_changes(a: RelationType, b: RelationType) -> list[Operation]:
    _require_expressible_type(a, b, _RELATION_KEYS_WITHOUT_OPERATIONS)
    ops: list[Operation] = []
    if a.description != b.description:
        ops.append(SetDescription("relation", b.label, b.description))
    if a.iri != b.iri:
        ops.append(SetIri("relation", b.label, b.iri))
    if (a.start_labels, a.end_labels) != (b.start_labels, b.end_labels):
        ops.append(SetEndpoints(b.label, b.start_labels, b.end_labels))
    ops += _profile_changes(a.profiles, b.profiles, "relation", b.label)
    ops += _property_changes("relation", b.label, a.properties, b.properties)
    return ops


def _node_profiles(node: NodeType) -> dict[str, dict[str, Any]]:
    return {name: dict(settings) for name, settings in node.profiles.items()}


def _profile_changes(
    a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]], kind: ProfileLevel, label: str | None
) -> list[Operation]:
    # A property's profiles change with the rest of it, through SetProperty.
    return [
        SetProfile(name, b.get(name), kind, label)
        for name in (*a, *(name for name in b if name not in a))
        if a.get(name) != b.get(name)
    ]


def _property_changes(kind: Kind, label: str, a: tuple[Property, ...], b: tuple[Property, ...]) -> list[Operation]:
    before = {p.name: p for p in a}
    after = {p.name: p for p in b}
    ops: list[Operation] = [RemoveProperty(kind, label, p.name) for p in a if p.name not in after]
    for prop in b:
        if prop.name not in before:
            ops.append(AddProperty(kind, label, prop))
        elif before[prop.name] != prop:
            ops.append(SetProperty(kind, label, prop))
    names = tuple(p.name for p in b)
    if _kept_order([p.name for p in a], names) != names:
        ops.append(OrderProperties(kind, label, names))
    return ops
