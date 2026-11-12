"""Applying operations: each a pure function from a model to a model (docs/contract.md, section 11)."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from ..errors import ModelError
from ..profiles import label_settings
from ..types import HygmModel, NodeType, Property, RelationType
from ..validation import validate_model
from .operations import (
    AddNodeType,
    AddProperty,
    AddRelationType,
    MergeNodeTypes,
    MergeRelationTypes,
    Operation,
    OrderNodeTypes,
    OrderProperties,
    OrderRelationTypes,
    RemoveNodeType,
    RemoveProperty,
    RemoveRelationType,
    RenameNodeType,
    RenameProperty,
    RenameRelationType,
    SetDescription,
    SetEndpoints,
    SetIri,
    SetKey,
    SetParents,
    SetProfile,
    SetProperty,
)

Owner = NodeType | RelationType


def apply(model: HygmModel, operations: Iterable[Operation], *, validate: bool = True) -> HygmModel:
    """`model` with `operations` applied in order; `model` itself is never changed.

    Only the result is validated, so a list may pass through invalid models
    on the way: removing a type leaves what names it for a later operation
    to change. Renames and merges carry through to everything naming what
    they rename.

    Args:
        model: The model to start from.
        operations: What to apply, in order.
        validate: False to skip validate_model() on the result, for a caller
            that applies a list in parts.

    Raises:
        ModelError: unknown_type, undeclared_property, duplicate_label,
            duplicate_property or invalid_operation when an operation doesn't
            apply to the model it meets, naming which one; or the codes of
            validate_model() when the result fails it.
    """
    for index, operation in enumerate(operations):
        try:
            model = _APPLY[type(operation)](model, operation)
        except ModelError as e:
            raise ModelError(e.codes, f"Operation {index} ({operation.op}): {e}") from e
    if validate:
        result = validate_model(model)
        if not result.success:
            problems = "; ".join(issue.message for issue in result.critical_issues)
            codes = [issue.code or "invalid_format" for issue in result.critical_issues]
            raise ModelError(codes, f"The operations leave an invalid model: {problems}")
    return model


def _node(model: HygmModel, label: str) -> NodeType:
    node = model.node_type(label)
    if node is None:
        raise ModelError("unknown_type", f"no node type {label!r}")
    return node


def _relation(model: HygmModel, label: str) -> RelationType:
    relation = next((r for r in model.relation_types if r.label == label), None)
    if relation is None:
        raise ModelError("unknown_type", f"no relation type {label!r}")
    return relation


def _owner(model: HygmModel, kind: str, label: str) -> Owner:
    return _node(model, label) if kind == "node" else _relation(model, label)


def _with(model: HygmModel, changed: Owner) -> HygmModel:
    """`model` with the type of `changed`'s kind and label replaced by it, in place."""
    if isinstance(changed, NodeType):
        return replace(model, node_types=tuple(changed if t.label == changed.label else t for t in model.node_types))
    return replace(
        model, relation_types=tuple(changed if r.label == changed.label else r for r in model.relation_types)
    )


def _declared(owner: Owner, name: str) -> Property:
    prop = next((p for p in owner.properties if p.name == name), None)
    if prop is None:
        raise ModelError("undeclared_property", f"{owner.label!r} declares no property {name!r}")
    return prop


def _free(model: HygmModel, kind: str, label: str) -> None:
    taken = model.node_labels() if kind == "node" else model.relation_labels()
    if label in taken:
        raise ModelError("duplicate_label", f"{kind} type {label!r} already exists")


def _relabel(labels: tuple[str, ...], source: str, target: str, *, drop: str | None = None) -> tuple[str, ...]:
    renamed = (target if label == source else label for label in labels)
    return tuple(dict.fromkeys(label for label in renamed if label != drop))


def _relabel_everywhere(model: HygmModel, source: str, target: str) -> HygmModel:
    """Every parent, endpoint, disjoint type and label-valued profile setting naming node type `source` names `target`."""
    nodes = tuple(
        replace(
            t,
            parents=_relabel(t.parents, source, target, drop=t.label),
            disjoint_with=_relabel(t.disjoint_with, source, target, drop=t.label),
            relation_restrictions=tuple(
                replace(r, end_labels=_relabel(r.end_labels, source, target)) if r.end_labels else r
                for r in t.relation_restrictions
            ),
        )
        for t in model.node_types
    )
    relations = tuple(
        replace(
            r,
            start_labels=_relabel(r.start_labels, source, target),
            end_labels=_relabel(r.end_labels, source, target),
        )
        for r in model.relation_types
    )
    profiles = {
        name: {
            key: target if key in label_settings(name, "model") and value == source else value
            for key, value in settings.items()
        }
        for name, settings in model.profiles.items()
    }
    individuals = tuple(replace(i, type=target) if i.type == source else i for i in model.individuals)
    return replace(model, node_types=nodes, relation_types=relations, profiles=profiles, individuals=individuals)


def _add_node_type(model: HygmModel, op: AddNodeType) -> HygmModel:
    _free(model, "node", op.node_type.label)
    return replace(model, node_types=(*model.node_types, op.node_type))


def _remove_node_type(model: HygmModel, op: RemoveNodeType) -> HygmModel:
    _node(model, op.label)
    return replace(model, node_types=tuple(t for t in model.node_types if t.label != op.label))


def _rename_node_type(model: HygmModel, op: RenameNodeType) -> HygmModel:
    _node(model, op.label)
    if op.to == op.label:
        raise ModelError("invalid_operation", f"renaming {op.label!r} to itself")
    _free(model, "node", op.to)
    model = replace(
        model, node_types=tuple(replace(t, label=op.to) if t.label == op.label else t for t in model.node_types)
    )
    return _relabel_everywhere(model, op.label, op.to)


def _merge_node_types(model: HygmModel, op: MergeNodeTypes) -> HygmModel:
    _node(model, op.label)
    _node(model, op.into)
    if op.label == op.into:
        raise ModelError("invalid_operation", f"merging {op.label!r} into itself")
    model = replace(model, node_types=tuple(t for t in model.node_types if t.label != op.label))
    return _relabel_everywhere(model, op.label, op.into)


def _add_relation_type(model: HygmModel, op: AddRelationType) -> HygmModel:
    _free(model, "relation", op.relation_type.label)
    return replace(model, relation_types=(*model.relation_types, op.relation_type))


def _remove_relation_type(model: HygmModel, op: RemoveRelationType) -> HygmModel:
    _relation(model, op.label)
    return replace(model, relation_types=tuple(r for r in model.relation_types if r.label != op.label))


def _rename_relation_type(model: HygmModel, op: RenameRelationType) -> HygmModel:
    _relation(model, op.label)
    if op.to == op.label:
        raise ModelError("invalid_operation", f"renaming {op.label!r} to itself")
    _free(model, "relation", op.to)
    model = replace(
        model,
        relation_types=tuple(replace(r, label=op.to) if r.label == op.label else r for r in model.relation_types),
    )
    return _relation_relabeled(model, op.label, op.to)


def _relation_relabeled(model: HygmModel, source: str, target: str) -> HygmModel:
    """Every `inverse_of`, relation restriction and individual's relationship naming relation type `source` names `target`."""
    relations = tuple(replace(r, inverse_of=target) if r.inverse_of == source else r for r in model.relation_types)
    nodes = tuple(
        replace(
            t,
            relation_restrictions=tuple(
                replace(r, relation=target) if r.relation == source else r for r in t.relation_restrictions
            ),
        )
        for t in model.node_types
    )
    individuals = tuple(
        replace(i, relations={target if name == source else name: ids for name, ids in i.relations.items()})
        if source in i.relations
        else i
        for i in model.individuals
    )
    return replace(model, node_types=nodes, relation_types=relations, individuals=individuals)


def _union(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    # An empty side means "any label", so a union with one stays unconstrained.
    return tuple(dict.fromkeys(a + b)) if a and b else ()


def _merge_relation_types(model: HygmModel, op: MergeRelationTypes) -> HygmModel:
    gone, survivor = _relation(model, op.label), _relation(model, op.into)
    if op.label == op.into:
        raise ModelError("invalid_operation", f"merging {op.label!r} into itself")
    merged = replace(
        survivor,
        start_labels=_union(survivor.start_labels, gone.start_labels),
        end_labels=_union(survivor.end_labels, gone.end_labels),
    )
    model = replace(model, relation_types=tuple(r for r in model.relation_types if r.label != op.label))
    return _relation_relabeled(_with(model, merged), op.label, op.into)


def _add_property(model: HygmModel, op: AddProperty) -> HygmModel:
    owner = _owner(model, op.kind, op.label)
    if any(p.name == op.property.name for p in owner.properties):
        raise ModelError("duplicate_property", f"{op.label!r} already declares property {op.property.name!r}")
    return _with(model, replace(owner, properties=(*owner.properties, op.property)))


def _remove_property(model: HygmModel, op: RemoveProperty) -> HygmModel:
    owner = _owner(model, op.kind, op.label)
    _declared(owner, op.name)
    return _with(model, replace(owner, properties=tuple(p for p in owner.properties if p.name != op.name)))


def _rename_property(model: HygmModel, op: RenameProperty) -> HygmModel:
    owner = _owner(model, op.kind, op.label)
    _declared(owner, op.name)
    if op.to == op.name:
        raise ModelError("invalid_operation", f"renaming property {op.name!r} to itself")
    if any(p.name == op.to for p in owner.properties):
        raise ModelError("duplicate_property", f"{op.label!r} already declares property {op.to!r}")
    if isinstance(owner, RelationType):
        return _with(model, _renamed_property(owner, op.name, op.to))
    renamed_in = (op.label, *model.descendants(op.label))
    for label in renamed_in:
        node = _node(model, label)
        node = replace(node, key=tuple(op.to if name == op.name else name for name in node.key))
        model = _with(model, _renamed_property(node, op.name, op.to))
    individuals = tuple(
        replace(i, properties={op.to if k == op.name else k: v for k, v in i.properties.items()})
        if i.type in renamed_in and op.name in i.properties
        else i
        for i in model.individuals
    )
    return replace(model, individuals=individuals)


def _renamed_property(owner: Owner, name: str, to: str) -> Owner:
    return replace(owner, properties=tuple(replace(p, name=to) if p.name == name else p for p in owner.properties))


def _set_property(model: HygmModel, op: SetProperty) -> HygmModel:
    owner = _owner(model, op.kind, op.label)
    _declared(owner, op.property.name)
    properties = tuple(op.property if p.name == op.property.name else p for p in owner.properties)
    return _with(model, replace(owner, properties=properties))


def _set_description(model: HygmModel, op: SetDescription) -> HygmModel:
    return _with(model, replace(_owner(model, op.kind, op.label), description=op.description))


def _set_iri(model: HygmModel, op: SetIri) -> HygmModel:
    return _with(model, replace(_owner(model, op.kind, op.label), iri=op.iri))


def _set_parents(model: HygmModel, op: SetParents) -> HygmModel:
    return _with(model, replace(_node(model, op.label), parents=op.parents))


def _set_key(model: HygmModel, op: SetKey) -> HygmModel:
    return _with(model, replace(_node(model, op.label), key=op.key))


def _set_endpoints(model: HygmModel, op: SetEndpoints) -> HygmModel:
    relation = _relation(model, op.label)
    return _with(model, replace(relation, start_labels=op.start_labels, end_labels=op.end_labels))


def _profiles_with(profiles: Mapping[str, Mapping[str, Any]], name: str, settings: Any) -> dict[str, Any]:
    updated = {key: dict(value) for key, value in profiles.items() if key != name}
    if settings is not None:
        updated[name] = dict(settings)
    return updated


def _set_profile(model: HygmModel, op: SetProfile) -> HygmModel:
    if op.kind == "model":
        return replace(model, profiles=_profiles_with(model.profiles, op.profile, op.settings))
    if op.label is None:  # unreachable: SetProfile refuses a type-level profile without a label
        raise ModelError("invalid_format", "set_profile needs a label")
    owner = _owner(model, op.kind, op.label)
    if op.property_name is not None:
        prop = _declared(owner, op.property_name)
        prop = replace(prop, profiles=_profiles_with(prop.profiles, op.profile, op.settings))
        return _with(
            model, replace(owner, properties=tuple(prop if p.name == prop.name else p for p in owner.properties))
        )
    return _with(model, replace(owner, profiles=_profiles_with(owner.profiles, op.profile, op.settings)))


def _ordered(current: tuple[str, ...], wanted: tuple[str, ...], what: str) -> None:
    if sorted(current) != sorted(wanted):
        raise ModelError("invalid_operation", f"{what} order must name each of {list(current)} once")


def _order_node_types(model: HygmModel, op: OrderNodeTypes) -> HygmModel:
    _ordered(model.node_labels(), op.labels, "node type")
    nodes = {t.label: t for t in model.node_types}
    return replace(model, node_types=tuple(nodes[label] for label in op.labels))


def _order_relation_types(model: HygmModel, op: OrderRelationTypes) -> HygmModel:
    _ordered(model.relation_labels(), op.labels, "relation type")
    relations = {r.label: r for r in model.relation_types}
    return replace(model, relation_types=tuple(relations[label] for label in op.labels))


def _order_properties(model: HygmModel, op: OrderProperties) -> HygmModel:
    owner = _owner(model, op.kind, op.label)
    _ordered(tuple(p.name for p in owner.properties), op.names, f"{op.label!r} property")
    properties = {p.name: p for p in owner.properties}
    return _with(model, replace(owner, properties=tuple(properties[name] for name in op.names)))


_APPLY: dict[type, Callable[[HygmModel, Any], HygmModel]] = {
    AddNodeType: _add_node_type,
    RemoveNodeType: _remove_node_type,
    RenameNodeType: _rename_node_type,
    MergeNodeTypes: _merge_node_types,
    AddRelationType: _add_relation_type,
    RemoveRelationType: _remove_relation_type,
    RenameRelationType: _rename_relation_type,
    MergeRelationTypes: _merge_relation_types,
    AddProperty: _add_property,
    RemoveProperty: _remove_property,
    RenameProperty: _rename_property,
    SetProperty: _set_property,
    SetDescription: _set_description,
    SetIri: _set_iri,
    SetParents: _set_parents,
    SetKey: _set_key,
    SetEndpoints: _set_endpoints,
    SetProfile: _set_profile,
    OrderNodeTypes: _order_node_types,
    OrderRelationTypes: _order_relation_types,
    OrderProperties: _order_properties,
}
