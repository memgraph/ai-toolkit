"""The operations between two models (docs/contract.md, section 11)."""

from collections.abc import Mapping, Sequence
from typing import Any

from ..types import HygmModel, NodeType, Property, RelationType
from .operations import (
    AddIndividual,
    AddNodeType,
    AddProperty,
    AddRelationType,
    Kind,
    Operation,
    OrderIndividuals,
    OrderNodeTypes,
    OrderProperties,
    OrderRelationTypes,
    ProfileLevel,
    RemoveIndividual,
    RemoveNodeType,
    RemoveProperty,
    RemoveRelationType,
    SetAbstract,
    SetDescription,
    SetDisjointWith,
    SetEndpoints,
    SetImports,
    SetIndividual,
    SetInverse,
    SetIri,
    SetKey,
    SetNames,
    SetOntology,
    SetParents,
    SetProfile,
    SetProperty,
    SetRelationCounts,
    SetRelationRestrictions,
    SetSymmetric,
    SetSynonyms,
)


def diff(a: HygmModel, b: HygmModel) -> tuple[Operation, ...]:
    """The operations that turn `a` into `b`: ``apply(a, diff(a, b)) == b``.

    A rename can't be told from a removal and an addition with the same
    definition, so diff never emits one: a renamed type, property or
    individual comes out as remove plus add. A caller that knows better (a
    user's edit, derivation's merges) records its own operations instead.

    The list is meant to be applied whole, as apply() does, validating only
    the result: it removes what `b` lacks first, then adds and changes types
    and individuals, then fixes the order, and a step in between may name
    something a later step adds. A type added whole, or a property or
    individual set whole, carries every key.
    """
    ops: list[Operation] = []
    b_nodes = {t.label: t for t in b.node_types}
    b_relations = {r.label: r for r in b.relation_types}
    b_individuals = {i.id: i for i in b.individuals}
    a_nodes = {t.label: t for t in a.node_types}
    a_relations = {r.label: r for r in a.relation_types}
    a_individuals = {i.id: i for i in a.individuals}
    ops += [RemoveNodeType(t.label) for t in a.node_types if t.label not in b_nodes]
    ops += [RemoveRelationType(r.label) for r in a.relation_types if r.label not in b_relations]
    ops += [RemoveIndividual(i.id) for i in a.individuals if i.id not in b_individuals]
    for node in b.node_types:
        ops += _node_changes(a_nodes[node.label], node) if node.label in a_nodes else [AddNodeType(node)]
    for relation in b.relation_types:
        before = a_relations.get(relation.label)
        ops += _relation_changes(before, relation) if before else [AddRelationType(relation)]
    ops += _profile_changes(a.profiles, b.profiles, "model", None)
    if a.ontology != b.ontology:
        ops.append(SetOntology(b.ontology))
    if a.imports != b.imports:
        ops.append(SetImports(b.imports))
    for individual in b.individuals:
        before = a_individuals.get(individual.id)
        if before is None:
            ops.append(AddIndividual(individual))
        elif before != individual:
            ops.append(SetIndividual(individual))
    if _kept_order(a.node_labels(), b.node_labels()) != b.node_labels():
        ops.append(OrderNodeTypes(b.node_labels()))
    if _kept_order(a.relation_labels(), b.relation_labels()) != b.relation_labels():
        ops.append(OrderRelationTypes(b.relation_labels()))
    b_ids = tuple(b_individuals)
    if _kept_order(tuple(a_individuals), b_ids) != b_ids:
        ops.append(OrderIndividuals(b_ids))
    return tuple(ops)


def _kept_order(before: Sequence[str], after: Sequence[str]) -> tuple[str, ...]:
    """The order removals and appended additions leave, before any reordering."""
    return (*(label for label in before if label in after), *(label for label in after if label not in before))


def _node_changes(a: NodeType, b: NodeType) -> list[Operation]:
    ops: list[Operation] = []
    if a.description != b.description:
        ops.append(SetDescription("node", b.label, b.description))
    if a.iri != b.iri:
        ops.append(SetIri("node", b.label, b.iri))
    if a.parents != b.parents:
        ops.append(SetParents(b.label, b.parents))
    if a.key != b.key:
        ops.append(SetKey(b.label, b.key))
    if a.abstract != b.abstract:
        ops.append(SetAbstract(b.label, b.abstract))
    if a.disjoint_with != b.disjoint_with:
        ops.append(SetDisjointWith(b.label, b.disjoint_with))
    if a.relation_restrictions != b.relation_restrictions:
        ops.append(SetRelationRestrictions(b.label, b.relation_restrictions))
    ops += _name_changes("node", a, b)
    ops += _profile_changes(_node_profiles(a), _node_profiles(b), "node", b.label)
    ops += _property_changes("node", b.label, a.properties, b.properties)
    return ops


def _relation_changes(a: RelationType, b: RelationType) -> list[Operation]:
    ops: list[Operation] = []
    if a.description != b.description:
        ops.append(SetDescription("relation", b.label, b.description))
    if a.iri != b.iri:
        ops.append(SetIri("relation", b.label, b.iri))
    if (a.start_labels, a.end_labels) != (b.start_labels, b.end_labels):
        ops.append(SetEndpoints(b.label, b.start_labels, b.end_labels))
    if (a.min_count, a.max_count) != (b.min_count, b.max_count):
        ops.append(SetRelationCounts(b.label, b.min_count, b.max_count))
    if a.inverse_of != b.inverse_of:
        ops.append(SetInverse(b.label, b.inverse_of))
    if a.symmetric != b.symmetric:
        ops.append(SetSymmetric(b.label, b.symmetric))
    ops += _name_changes("relation", a, b)
    ops += _profile_changes(a.profiles, b.profiles, "relation", b.label)
    ops += _property_changes("relation", b.label, a.properties, b.properties)
    return ops


def _name_changes(kind: Kind, a: NodeType | RelationType, b: NodeType | RelationType) -> list[Operation]:
    ops: list[Operation] = []
    if a.synonyms != b.synonyms:
        ops.append(SetSynonyms(kind, b.label, b.synonyms))
    if a.names != b.names:
        ops.append(SetNames(kind, b.label, b.names))
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
