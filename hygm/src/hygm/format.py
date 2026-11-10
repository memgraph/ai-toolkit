"""The model file: reading versions 1 and 2, writing the lowest one that fits (docs/contract.md, section 4).

Version 1 is every model written before the contract, including the ontology
versions stored in users' graphs: `entity_types`, a node-level `identity`, no
`version` key. Version 2 adds `node_types`, properties, `parents`, `key`,
`iri` and profiles. A reader refuses any key the version doesn't define, so a
typo is an error rather than silently dropped meaning.
"""

import json
from collections.abc import Callable
from dataclasses import replace
from importlib.resources import files
from typing import Any

from .errors import ModelError
from .profiles import REGISTERED_PROFILES, SETTINGS, has_implied_profiles, implied_profiles
from .types import HygmModel, NodeType, Property, RelationType
from .validation import validate_model

VERSIONS = (1, 2)

_V1_TOP = {"version", "entity_types", "relation_types"}
_V1_NODE = {"label", "description", "identity"}
_V1_RELATION = {"label", "description", "start_labels", "end_labels"}

_V2_TOP = {"version", "node_types", "relation_types", "profiles"}
_V2_NODE = {"label", "description", "parents", "properties", "key", "iri", "profiles"}
_V2_RELATION = {"label", "description", "start_labels", "end_labels", "properties", "iri", "profiles"}
_V2_PROPERTY = {"name", "type", "description", "required", "values", "iri", "profiles"}


def model_from_mapping(raw: Any, source: str, *, validate: bool = True) -> HygmModel:
    """A model from the mapping a model file holds, in either format version.

    Args:
        raw: The parsed mapping.
        source: What it was read from, for error messages ("file schema.yaml", "version 3").
        validate: False for a store of types that is not itself extracted
            against, like a pool of retired ones, whose relations may name
            types it doesn't hold. Each entry is still checked.

    Raises:
        ModelError: (a ValueError) if the mapping breaks the format, or
            `validate` and the model fails validate_model()'s hard gate.
    """
    where_from = f"Ontology {source}"
    if not isinstance(raw, dict):
        raise ModelError("invalid_format", f"{where_from} must be a mapping with an 'entity_types' list")
    version = raw.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ModelError("invalid_format", f"{where_from}: version must be an integer")
    if version not in VERSIONS:
        raise ModelError("unknown_version", f"{where_from}: unknown format version {version}; known: {VERSIONS}")

    reader = _Reader(where_from)
    model = reader.v1(raw) if version == 1 else reader.v2(raw)
    if not validate:
        return model
    result = validate_model(model)
    if not result.success:
        problems = "; ".join(issue.message for issue in result.critical_issues)
        codes = [issue.code or "invalid_format" for issue in result.critical_issues]
        raise ModelError(codes, f"{where_from} is not a valid model: {problems}")
    return model


def model_to_mapping(model: HygmModel) -> dict[str, Any]:
    """`model` as the mapping model_from_mapping() reads back: JSON- and YAML-safe.

    Writes version 1, exactly as before the contract, whenever the model uses
    nothing version 1 lacks, so stored versions stay byte-stable. Otherwise
    writes version 2, leaving out every default.
    """
    return _v2_mapping(model) if needs_version_2(model) else _v1_mapping(model)


def model_schema(version: int) -> dict[str, Any]:
    """The JSON Schema for a format `version`, as shipped in the package.

    It is the structural floor; docs/contract.md defines the rules a schema
    can't express, which model_from_mapping() also checks.

    Raises:
        ModelError: for an unknown version.
    """
    if version not in VERSIONS:
        raise ModelError("unknown_version", f"Unknown format version {version}; known: {VERSIONS}")
    return json.loads((files("hygm") / "schema" / f"model.v{version}.schema.json").read_text(encoding="utf-8"))


def needs_version_2(model: HygmModel) -> bool:
    """Whether `model` uses anything format version 1 can't express."""
    return bool(
        not has_implied_profiles(model)
        or any(t.parents or t.properties or t.key or t.iri for t in model.node_types)
        or any(r.properties or r.iri or r.profiles for r in model.relation_types)
    )


def node_type_to_mapping(node: NodeType) -> dict[str, Any]:
    """One node type as a version-2 `node_types` entry, defaults left out."""
    return _v2_node(node)


def relation_type_to_mapping(relation: RelationType) -> dict[str, Any]:
    """One relation type as a version-2 `relation_types` entry, defaults left out."""
    return _v2_relation(relation)


def property_to_mapping(prop: Property) -> dict[str, Any]:
    """One property as a version-2 `properties` entry, defaults left out."""
    return _v2_property(prop)


def node_type_from_mapping(raw: Any, source: str) -> NodeType:
    """A node type from a version-2 `node_types` entry, read as strictly as in a model file.

    Raises:
        ModelError: if the entry breaks the format.
    """
    return _Reader(source).node(raw, "")


def relation_type_from_mapping(raw: Any, source: str) -> RelationType:
    """A relation type from a version-2 `relation_types` entry, read as strictly as in a model file.

    Raises:
        ModelError: if the entry breaks the format.
    """
    return _Reader(source).relation(raw, "", _V2_RELATION)


def property_from_mapping(raw: Any, source: str) -> Property:
    """A property from a version-2 `properties` entry, read as strictly as in a model file.

    Raises:
        ModelError: if the entry breaks the format.
    """
    return _Reader(source).property(raw, "")


def check_profile(profile: str, settings: Any, level: str, source: str) -> None:
    """Check one profile's settings at `level` ("model", "node", "relation" or "property") as a model file would.

    `settings` None checks only the profile's name.

    Raises:
        ModelError: unknown_profile, unknown_key, invalid_profile_value or invalid_format.
    """
    _Reader(source).profiles({profile: {} if settings is None else settings}, "profiles", level)


def _v1_mapping(model: HygmModel) -> dict[str, Any]:
    return {
        "entity_types": [
            {"label": t.label, "description": t.description, "identity": t.identity} for t in model.node_types
        ],
        "relation_types": [
            {
                "label": r.label,
                "description": r.description,
                "start_labels": list(r.start_labels),
                "end_labels": list(r.end_labels),
            }
            for r in model.relation_types
        ],
    }


def _v2_mapping(model: HygmModel) -> dict[str, Any]:
    out: dict[str, Any] = {"version": 2, "node_types": [_v2_node(t) for t in model.node_types]}
    if model.relation_types:
        out["relation_types"] = [_v2_relation(r) for r in model.relation_types]
    if model.profiles:
        out["profiles"] = _plain(model.profiles)
    return out


def _v2_node(node: NodeType) -> dict[str, Any]:
    out: dict[str, Any] = {"label": node.label, "description": node.description}
    if node.parents:
        out["parents"] = list(node.parents)
    if node.properties:
        out["properties"] = [_v2_property(p) for p in node.properties]
    if node.key:
        out["key"] = list(node.key)
    if node.iri:
        out["iri"] = node.iri
    profiles = _plain(node.profiles)
    if node.identity != "chunk":
        profiles["extraction"] = {**profiles.get("extraction", {}), "identity": node.identity}
    if profiles:
        out["profiles"] = profiles
    return out


def _v2_relation(relation: RelationType) -> dict[str, Any]:
    out: dict[str, Any] = {"label": relation.label, "description": relation.description}
    if relation.start_labels:
        out["start_labels"] = list(relation.start_labels)
    if relation.end_labels:
        out["end_labels"] = list(relation.end_labels)
    if relation.properties:
        out["properties"] = [_v2_property(p) for p in relation.properties]
    if relation.iri:
        out["iri"] = relation.iri
    if relation.profiles:
        out["profiles"] = _plain(relation.profiles)
    return out


def _v2_property(prop: Property) -> dict[str, Any]:
    out: dict[str, Any] = {"name": prop.name}
    if prop.type != "string":
        out["type"] = prop.type
    if prop.description:
        out["description"] = prop.description
    if prop.required:
        out["required"] = True
    if prop.values is not None:
        out["values"] = list(prop.values)
    if prop.iri:
        out["iri"] = prop.iri
    if prop.profiles:
        out["profiles"] = _plain(prop.profiles)
    return out


def _plain(profiles: Any) -> dict[str, Any]:
    return {name: dict(settings) for name, settings in profiles.items()}


class _Reader:
    """Turns one parsed model mapping into a HygmModel, naming where each problem is."""

    def __init__(self, where_from: str) -> None:
        self.where_from = where_from

    def fail(self, code: str, where: str, problem: str) -> ModelError:
        return ModelError(code, f"{self.where_from}: {where} {problem}" if where else f"{self.where_from} {problem}")

    def v1(self, raw: dict) -> HygmModel:
        if "node_types" not in raw:
            self.listed(raw, "entity_types", required=True)
        self.known(raw, _V1_TOP, "", hint="version 2 keys need `version: 2`")
        nodes = self.listed(raw, "entity_types", required=True)
        relations = self.listed(raw, "relation_types")
        node_types = []
        for index, item in enumerate(nodes):
            where = f"entity_types[{index}]"
            label, description = self.labeled(item, where)
            self.known(item, _V1_NODE, where)
            node_types.append(self.build(NodeType, where, label, description, identity=item.get("identity", "chunk")))
        relation_types = [
            self.relation(item, f"relation_types[{index}]", _V1_RELATION) for index, item in enumerate(relations)
        ]
        model = HygmModel(node_types=tuple(node_types), relation_types=tuple(relation_types))
        top, nodes = implied_profiles(model)
        return HygmModel(
            node_types=tuple(replace(t, profiles=nodes[t.label]) for t in model.node_types),
            relation_types=model.relation_types,
            profiles=top,
        )

    def v2(self, raw: dict) -> HygmModel:
        if "entity_types" not in raw:
            self.listed(raw, "node_types", required=True)
        self.known(raw, _V2_TOP, "", hint="version 2 calls entity_types node_types")
        nodes = self.listed(raw, "node_types", required=True)
        relations = self.listed(raw, "relation_types")
        node_types = [self.node(item, f"node_types[{index}]") for index, item in enumerate(nodes)]
        relation_types = [
            self.relation(item, f"relation_types[{index}]", _V2_RELATION) for index, item in enumerate(relations)
        ]
        profiles = self.profiles(raw.get("profiles", {}), "profiles", "model")
        return HygmModel(node_types=tuple(node_types), relation_types=tuple(relation_types), profiles=profiles)

    def node(self, item: Any, where: str) -> NodeType:
        label, description = self.labeled(item, where)
        self.known(item, _V2_NODE, where)
        profiles = self.profiles(item.get("profiles", {}), f"{where} profiles", "node")
        identity = profiles.get("extraction", {}).get("identity", "chunk")
        rest = {name: settings for name, settings in profiles.items() if name != "extraction"}
        extraction = {k: v for k, v in profiles.get("extraction", {}).items() if k != "identity"}
        if extraction:
            rest["extraction"] = extraction
        return self.build(
            NodeType,
            where,
            label,
            description,
            identity=identity,
            parents=self.labels(item, "parents", where),
            properties=self.properties(item, where),
            key=self.labels(item, "key", where, what="list of property names"),
            iri=self.optional_string(item, "iri", where),
            profiles=rest,
        )

    def relation(self, item: Any, where: str, allowed: set[str]) -> RelationType:
        label, description = self.labeled(item, where)
        self.known(item, allowed, where)
        return self.build(
            RelationType,
            where,
            label,
            description,
            start_labels=self.labels(item, "start_labels", where),
            end_labels=self.labels(item, "end_labels", where),
            properties=self.properties(item, where),
            iri=self.optional_string(item, "iri", where),
            profiles=self.profiles(item.get("profiles", {}), f"{where} profiles", "relation"),
        )

    def properties(self, item: dict, where: str) -> tuple[Property, ...]:
        return tuple(
            self.property(prop, f"{where} properties[{index}]")
            for index, prop in enumerate(self.listed(item, "properties", where=where))
        )

    def property(self, prop: Any, at: str) -> Property:
        if not isinstance(prop, dict) or not isinstance(prop.get("name"), str):
            raise self.fail("invalid_format", at, "must be a mapping with a string 'name'")
        self.known(prop, _V2_PROPERTY, at)
        if not isinstance(prop.get("required", False), bool):
            raise self.fail("invalid_format", at, "required must be true or false")
        values = prop.get("values")
        if values is not None and (
            not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values)
        ):
            raise self.fail("invalid_format", at, "values must be a non-empty list of strings")
        return self.build(
            Property,
            at,
            prop["name"],
            type=prop.get("type", "string"),
            description=self.optional_string(prop, "description", at) or "",
            required=prop.get("required", False),
            values=tuple(values) if values is not None else None,
            iri=self.optional_string(prop, "iri", at),
            profiles=self.profiles(prop.get("profiles", {}), f"{at} profiles", "property"),
        )

    def profiles(self, value: Any, where: str, level: str) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
            raise self.fail("invalid_format", where, "must map profile names to mappings of settings")
        for name, settings in value.items():
            if name.startswith("x_"):
                continue
            if name not in REGISTERED_PROFILES:
                raise self.fail(
                    "unknown_profile",
                    where,
                    f"names unknown profile {name!r}; registered: {REGISTERED_PROFILES}, or prefix a private one x_",
                )
            allowed = SETTINGS.get((name, level), {})
            self.known(settings, set(allowed), f"{where} {name}")
            for key, valid in allowed.items():
                if key in settings and not valid(settings[key]):
                    raise self.fail("invalid_profile_value", f"{where} {name}", f"{key} can't be {settings[key]!r}")
        return {name: dict(settings) for name, settings in value.items()}

    def build(self, kind: Callable[..., Any], where: str, *args: Any, **kwargs: Any) -> Any:
        """Construct a type, prefixing a construction error with where it was read from."""
        try:
            return kind(*args, **kwargs)
        except ModelError as e:
            raise ModelError(e.codes, f"{self.where_from}: {where} {e}") from e

    def known(self, item: dict, allowed: set[str], where: str, hint: str = "") -> None:
        unknown = sorted(set(item) - allowed)
        if unknown:
            extra = f" ({hint})" if hint else ""
            raise self.fail("unknown_key", where, f"has unknown key {unknown[0]!r}{extra}".lstrip())

    def listed(self, item: dict, key: str, *, required: bool = False, where: str = "") -> list:
        value = item.get(key, None if required else [])
        if not isinstance(value, list):
            if required:
                raise self.fail("invalid_format", where, f"must be a mapping with an {key!r} list")
            raise self.fail("invalid_format", where, f"{key!r} must be a list")
        return value

    def labeled(self, item: Any, where: str) -> tuple[str, str]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not isinstance(item.get("description"), str)
        ):
            raise self.fail("invalid_format", where, "must be a mapping with string 'label' and 'description'")
        return item["label"], item["description"]

    def labels(self, item: dict, key: str, where: str, what: str = "list of labels") -> tuple[str, ...]:
        value = item.get(key, [])
        if not isinstance(value, list) or not all(isinstance(label, str) for label in value):
            raise self.fail("invalid_format", where, f"{key} must be a {what}")
        return tuple(value)

    def optional_string(self, item: dict, key: str, where: str) -> str | None:
        value = item.get(key)
        if value is not None and not isinstance(value, str):
            raise self.fail("invalid_format", where, f"{key} must be a string")
        return value
