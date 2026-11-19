"""The resolved form: a validated model with nothing left to work out, for readers in other languages.

A reader of a model file has to resolve imports, inheritance with facet
refinement and relation restrictions, and re-run validation, before it can
trust the file. The resolved form moves that work here: hygm validates the
model once and writes every fact a reader would otherwise derive
(docs/contract.md, 4.18). It is derived, never authored, and hygm doesn't
read it back.
"""

import json
from importlib.resources import files
from typing import Any

from .errors import ModelError
from .format import individual_to_mapping, ontology_to_mapping
from .types import HygmModel, NodeType, Property, RelationRestriction, RelationType
from .validation import validate_model

#: The `format` every resolved file names, so a reader can refuse any other file before reading on.
RESOLVED_FORMAT = "hygm-resolved"

#: The resolved form's own version; it changes independently of the model format's.
RESOLVED_VERSION = 1

#: The model format version whose meaning the resolved form carries. A
#: version-1 file reads into the same model as its version-2 equivalent.
RESOLVED_MODEL_VERSION = 2


def resolve_model(model: HygmModel) -> dict[str, Any]:
    """`model` in the resolved form: imports inlined, inheritance and endpoints worked out. JSON-safe.

    Key and list order are fixed, so writing the result with `json.dumps`
    is byte-stable for one model.

    Raises:
        ModelError: with the codes of every critical issue, if `model`
            fails validate_model(). Nothing is resolved from an invalid model.
    """
    result = validate_model(model)
    if not result.success:
        problems = "; ".join(issue.message for issue in result.critical_issues)
        raise ModelError(
            [issue.code or "invalid_format" for issue in result.critical_issues],
            f"Can't resolve an invalid model: {problems}",
        )
    out: dict[str, Any] = {
        "format": RESOLVED_FORMAT,
        "version": RESOLVED_VERSION,
        "model_version": RESOLVED_MODEL_VERSION,
    }
    if model.ontology is not None:
        out["ontology"] = ontology_to_mapping(model.ontology)
    out["node_types"] = [_node(model, node) for node in model.node_types]
    out["relation_types"] = [_relation(model, relation) for relation in model.relation_types]
    if model.individuals:
        out["individuals"] = [individual_to_mapping(i) for i in model.individuals]
    if model.profiles:
        out["profiles"] = _plain(model.profiles)
    return out


def resolved_to_json(model: HygmModel) -> str:
    """`model` resolved, as the text `hygm resolve` writes: two-space indented JSON with a final newline.

    Raises:
        ModelError: as resolve_model().
    """
    return json.dumps(resolve_model(model), indent=2, ensure_ascii=False) + "\n"


def resolved_schema(version: int = RESOLVED_VERSION) -> dict[str, Any]:
    """The JSON Schema for resolved-form `version`, as shipped in the package.

    Raises:
        ModelError: `unknown_version` for a version hygm doesn't write.
    """
    if version != RESOLVED_VERSION:
        raise ModelError("unknown_version", f"Unknown resolved form version {version}; known: {RESOLVED_VERSION}")
    return json.loads((files("hygm") / "schema" / f"resolved.v{version}.schema.json").read_text(encoding="utf-8"))


def _node(model: HygmModel, node: NodeType) -> dict[str, Any]:
    ancestors = model.ancestors(node.label)
    out: dict[str, Any] = {"label": node.label, "description": node.description}
    if node.parents:
        out["parents"] = list(node.parents)
    if ancestors:
        out["ancestors"] = list(ancestors)
    if node.abstract:
        out["abstract"] = True
    key = model.key_of(node.label)
    if key:
        out["key"] = list(key)
    properties = model.properties_of(node.label)
    if properties:
        out["properties"] = [_property(p, _introduced_by(model, node.label, p.name)) for p in properties]
    if node.iri:
        out["iri"] = node.iri
    if node.disjoint_with:
        out["disjoint_with"] = list(node.disjoint_with)
    _named(out, node)
    restrictions = [
        _restriction(model, restriction, owner)
        for owner in reversed((node.label, *ancestors))
        for restriction in _own_restrictions(model, owner)
    ]
    if restrictions:
        out["relation_restrictions"] = restrictions
    if node.profiles:
        out["profiles"] = _plain(node.profiles)
    return out


def _introduced_by(model: HygmModel, label: str, name: str) -> str:
    """The type that introduces property `name` to `label`: the farthest declarer among it and its ancestors.

    A subtype's redeclaration only narrows a property (4.15), so the
    property stays the one its topmost declarer introduced.
    """
    owners = (label, *model.ancestors(label))
    return next(
        owner
        for owner in reversed(owners)
        if any(p.name == name for p in getattr(model.node_type(owner), "properties", ()))
    )


def _own_restrictions(model: HygmModel, label: str) -> tuple[RelationRestriction, ...]:
    node = model.node_type(label)
    return node.relation_restrictions if node else ()


def _restriction(model: HygmModel, restriction: RelationRestriction, declared_by: str) -> dict[str, Any]:
    out: dict[str, Any] = {"relation": restriction.relation}
    if restriction.end_labels is not None:
        out["end_labels"] = list(restriction.end_labels)
        out["end_types"] = list(_expanded(model, restriction.end_labels))
    for key in ("min_count", "max_count"):
        if getattr(restriction, key) is not None:
            out[key] = getattr(restriction, key)
    out["declared_by"] = declared_by
    return out


def _relation(model: HygmModel, relation: RelationType) -> dict[str, Any]:
    out: dict[str, Any] = {"label": relation.label, "description": relation.description}
    if relation.start_labels:
        out["start_labels"] = list(relation.start_labels)
    if relation.end_labels:
        out["end_labels"] = list(relation.end_labels)
    out["start_types"] = list(model.endpoint_labels(relation, "start"))
    out["end_types"] = list(model.endpoint_labels(relation, "end"))
    if relation.properties:
        out["properties"] = [_property(p, relation.label) for p in relation.properties]
    if relation.iri:
        out["iri"] = relation.iri
    if relation.min_count:
        out["min_count"] = relation.min_count
    if relation.max_count is not None:
        out["max_count"] = relation.max_count
    if relation.inverse_of is not None:
        out["inverse_of"] = relation.inverse_of
    if relation.symmetric:
        out["symmetric"] = True
    _named(out, relation)
    if relation.profiles:
        out["profiles"] = _plain(relation.profiles)
    return out


def _property(prop: Property, declared_by: str) -> dict[str, Any]:
    """`prop` with its effective facets.

    `type` is always written. `max_count` is left out only at its default
    of 1; unbounded is written as null, since a reader can't tell a missing
    key from "no limit" otherwise.
    """
    out: dict[str, Any] = {"name": prop.name, "type": prop.type}
    if prop.description:
        out["description"] = prop.description
    if prop.required:
        out["required"] = True
    if prop.values is not None:
        out["values"] = list(prop.values)
    if prop.min_count is not None:
        out["min_count"] = prop.min_count
    if prop.max_count != 1:
        out["max_count"] = prop.max_count
    for key in ("default", "value", "minimum", "maximum", "unit", "iri"):
        value = getattr(prop, key)
        if value is not None:
            out[key] = list(value) if isinstance(value, tuple) else value
    _named(out, prop)
    if prop.profiles:
        out["profiles"] = _plain(prop.profiles)
    out["declared_by"] = declared_by
    return out


def _expanded(model: HygmModel, labels: tuple[str, ...]) -> tuple[str, ...]:
    expanded = dict.fromkeys(labels)
    for label in labels:
        expanded.update(dict.fromkeys(model.descendants(label)))
    return tuple(expanded)


def _named(out: dict[str, Any], named: NodeType | RelationType | Property) -> None:
    if named.synonyms:
        out["synonyms"] = list(named.synonyms)
    if named.names:
        out["names"] = dict(named.names)


def _plain(profiles: Any) -> dict[str, Any]:
    return {name: dict(settings) for name, settings in profiles.items()}
