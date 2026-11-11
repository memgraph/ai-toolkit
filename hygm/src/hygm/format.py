"""The model file: reading versions 1 and 2, writing the lowest one that fits (docs/contract.md, section 4).

Version 1 is every model written before the contract: `entity_types`, a
node-level `identity`, no `version` key. Version 2 adds `node_types`,
properties, `parents`, `key`, `iri` and profiles, and later, additively, the
`ontology` block and `imports` (4.11, 4.12). A reader refuses any key the
version doesn't define, so a typo is an error rather than silently dropped
meaning.
"""

import json
from collections.abc import Callable
from importlib.resources import files
from typing import Any

from .errors import ModelError
from .profiles import (
    IDENTITIES,
    REGISTERED_PROFILES,
    SETTINGS,
    Version1Profiles,
    has_version_1_profiles,
    no_implied_profiles,
    setting,
    with_version_1_profiles,
)
from .types import Conventions, HygmModel, Import, NodeType, Ontology, Property, RelationRestriction, RelationType
from .validation import validate_model

#: Resolves one `imports` path, as the importing file wrote it, to the
#: imported model with its own imports resolved. ManualStrategy supplies one
#: that reads files relative to the importing file.
ImportResolver = Callable[[str], HygmModel]

VERSIONS = (1, 2)

_V1_TOP = {"version", "entity_types", "relation_types"}
_V1_NODE = {"label", "description", "identity"}
_V1_RELATION = {"label", "description", "start_labels", "end_labels"}

_V2_TOP = {"version", "node_types", "relation_types", "profiles", "ontology", "imports"}
_ONTOLOGY = {
    "name",
    "description",
    "iri",
    "version_info",
    "scope",
    "competency_questions",
    "maintainers",
    "conventions",
}
_SCOPE = {"includes", "excludes"}
_CONVENTIONS = {"node_case", "property_case", "relation_case", "number"}
_IMPORT = {"path"}
_NAMED = {"synonyms", "names"}
_RESTRICTION = {"relation", "end_labels", "min_count", "max_count"}
_V2_NODE = {
    "label",
    "description",
    "parents",
    "properties",
    "key",
    "iri",
    "profiles",
    "abstract",
    "disjoint_with",
    "relation_restrictions",
} | _NAMED
_RELATION_FACETS = {"min_count", "max_count", "inverse_of", "symmetric"}
_V2_RELATION = (
    {"label", "description", "start_labels", "end_labels", "properties", "iri", "profiles"} | _NAMED | _RELATION_FACETS
)
_FACETS = {"min_count", "max_count", "default", "value", "minimum", "maximum", "unit"}
_V2_PROPERTY = {"name", "type", "description", "required", "values", "iri", "profiles"} | _NAMED | _FACETS


def model_from_mapping(
    raw: Any,
    source: str,
    *,
    validate: bool = True,
    version_1_profiles: Version1Profiles | None = None,
    resolver: ImportResolver | None = None,
) -> HygmModel:
    """A model from the mapping a model file holds, in either format version.

    Args:
        raw: The parsed mapping.
        source: What it was read from, for error messages ("file schema.yaml", "version 3").
        validate: False for a store of types that is not itself extracted
            against, like a pool of retired ones, whose relations may name
            types it doesn't hold. Each entry is still checked.
        version_1_profiles: The profiles a version-1 file implies, its owner's
            rule (docs/contract.md, 4.9). Applied to version-1 files only;
            without it, a version-1 file implies nothing beyond `identity`.
        resolver: Resolves each `imports` path to its model (4.12). Without
            one, a model with imports is refused: given a model as data,
            there is nothing to resolve its paths against.

    Raises:
        ModelError: (a ValueError) if the mapping breaks the format, or
            `validate` and the model fails validate_model()'s hard gate;
            `unknown_model` for imports and no resolver; `duplicate_label`
            when an import declares a label differently from the importing
            model or another import; whatever `resolver` raises.
    """
    where_from = f"Ontology {source}"
    if not isinstance(raw, dict):
        raise ModelError("invalid_format", f"{where_from} must be a mapping with an 'entity_types' list")
    version = raw.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ModelError("invalid_format", f"{where_from}: version must be an integer")
    if version not in VERSIONS:
        raise ModelError("unknown_version", f"{where_from}: unknown format version {version}; known: {VERSIONS}")

    reader = _Reader(where_from, resolver)
    model = reader.v1(raw) if version == 1 else reader.v2(raw)
    if version == 1 and version_1_profiles is not None:
        model = with_version_1_profiles(model, version_1_profiles)
    if not validate:
        return model
    result = validate_model(model)
    if not result.success:
        problems = "; ".join(issue.message for issue in result.critical_issues)
        codes = [issue.code or "invalid_format" for issue in result.critical_issues]
        raise ModelError(codes, f"{where_from} is not a valid model: {problems}")
    return model


def model_to_mapping(model: HygmModel, *, version_1_profiles: Version1Profiles | None = None) -> dict[str, Any]:
    """`model` as the mapping model_from_mapping() reads back: JSON- and YAML-safe.

    Writes version 1, exactly as before the contract, whenever the model uses
    nothing version 1 lacks and its profiles are exactly what
    `version_1_profiles` says a version-1 file implies, so stored versions
    stay byte-stable. Otherwise writes version 2, leaving out every default.
    Pass the same rule the reader was given.

    A model with imports is written as the importing file: its `imports`
    paths and its own types, never the imported ones inline.

    Raises:
        ModelError: `duplicate_label` if the model changed or dropped an
            imported type, which the importing file can't express: an
            importing model refines imported types by subtyping them (4.12).
    """
    return _v2_mapping(model) if needs_version_2(model, version_1_profiles) else _v1_mapping(model)


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


def needs_version_2(model: HygmModel, version_1_profiles: Version1Profiles | None = None) -> bool:
    """Whether `model` uses anything format version 1 can't express, given what a version-1 file implies."""
    return bool(
        model.uses_hierarchy_or_properties()
        or not has_version_1_profiles(model, version_1_profiles or no_implied_profiles)
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
            {"label": t.label, "description": t.description, "identity": _identity(t)} for t in model.node_types
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
    nodes, relations = _own_types(model)
    out: dict[str, Any] = {"version": 2}
    if model.ontology is not None:
        out["ontology"] = _ontology(model.ontology)
    if model.imports:
        out["imports"] = [{"path": i.path} for i in model.imports]
    out["node_types"] = [_v2_node(t, model) for t in nodes]
    if relations:
        out["relation_types"] = [_v2_relation(r) for r in relations]
    if model.profiles:
        out["profiles"] = _plain(model.profiles)
    return out


def _own_types(model: HygmModel) -> tuple[list[NodeType], list[RelationType]]:
    """The types the importing file itself declares; refuses a model that changed an imported one."""
    if not model.imports:
        return list(model.node_types), list(model.relation_types)
    imported_nodes = _imported(model.imports, "node_types")
    imported_relations = _imported(model.imports, "relation_types")
    for kind, imported, declared in (
        ("node type", imported_nodes, {t.label: t for t in model.node_types}),
        ("relation type", imported_relations, {r.label: r for r in model.relation_types}),
    ):
        for label, original in imported.items():
            if declared.get(label) != original:
                raise ModelError(
                    "duplicate_label",
                    f"The model changes or drops the imported {kind} {label!r}; an importing model can't "
                    "change an imported type, only subtype it",
                )
    return (
        [t for t in model.node_types if t.label not in imported_nodes],
        [r for r in model.relation_types if r.label not in imported_relations],
    )


def _imported(imports: tuple[Import, ...], attribute: str) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for item in imports:
        for declared in getattr(item.model, attribute):
            found.setdefault(declared.label, declared)
    return found


def _ontology(ontology: Ontology) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("name", "description", "iri", "version_info"):
        if getattr(ontology, key) is not None:
            out[key] = getattr(ontology, key)
    scope = {key: list(getattr(ontology, key)) for key in ("includes", "excludes") if getattr(ontology, key)}
    if scope:
        out["scope"] = scope
    for key in ("competency_questions", "maintainers"):
        if getattr(ontology, key):
            out[key] = list(getattr(ontology, key))
    if ontology.conventions is not None:
        out["conventions"] = {
            key: getattr(ontology.conventions, key)
            for key in ("node_case", "property_case", "relation_case", "number")
            if getattr(ontology.conventions, key) is not None
        }
    return out


def _merged(imports: tuple[Import, ...], own: tuple[Any, ...], attribute: str, kind: str, where_from: str) -> tuple:
    """Imported declarations, then `own`; a label both declare must be one declaration (a diamond).

    A label `own` declares twice is left for validate_model() to report.
    """
    merged: dict[str, Any] = {}
    for declared in (t for i in imports for t in getattr(i.model, attribute)):
        seen = merged.setdefault(declared.label, declared)
        if seen != declared:
            raise ModelError(
                "duplicate_label", f"{where_from}: {kind} {declared.label!r} is declared differently by two imports"
            )
    for declared in own:
        seen = merged.get(declared.label)
        if seen is not None and seen != declared:
            raise ModelError(
                "duplicate_label",
                f"{where_from}: {kind} {declared.label!r} is declared differently by the model and an import",
            )
    return (*merged.values(), *(t for t in own if merged.get(t.label) != t))


def _v2_node(node: NodeType, model: HygmModel | None = None) -> dict[str, Any]:
    """`node` as an entry; within `model`, a redeclared property leaves out what it inherits unchanged."""
    out: dict[str, Any] = {"label": node.label, "description": node.description}
    if node.parents:
        out["parents"] = list(node.parents)
    if node.properties:
        out["properties"] = [
            _v2_property(p, model.inherited(node.label, p.name) if model else None) for p in node.properties
        ]
    if node.key:
        out["key"] = list(node.key)
    if node.iri:
        out["iri"] = node.iri
    profiles = _plain(node.profiles)
    if profiles:
        out["profiles"] = profiles
    if node.abstract:
        out["abstract"] = True
    if node.disjoint_with:
        out["disjoint_with"] = list(node.disjoint_with)
    if node.relation_restrictions:
        out["relation_restrictions"] = [_restriction(r) for r in node.relation_restrictions]
    return _with_names(out, node)


def _restriction(restriction: RelationRestriction) -> dict[str, Any]:
    out: dict[str, Any] = {"relation": restriction.relation}
    if restriction.end_labels is not None:
        out["end_labels"] = list(restriction.end_labels)
    for key in ("min_count", "max_count"):
        if getattr(restriction, key) is not None:
            out[key] = getattr(restriction, key)
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
    if relation.min_count:
        out["min_count"] = relation.min_count
    if relation.max_count is not None:
        out["max_count"] = relation.max_count
    if relation.inverse_of is not None:
        out["inverse_of"] = relation.inverse_of
    if relation.symmetric:
        out["symmetric"] = True
    return _with_names(out, relation)


def _v2_property(prop: Property, inherited: Property | None = None) -> dict[str, Any]:
    """`prop` as an entry, defaults left out.

    A redeclaration (`inherited` given) also leaves out a `type` and
    `max_count` equal to the inherited ones, which the reader fills back in.
    """
    out: dict[str, Any] = {"name": prop.name}
    if prop.type != (inherited.type if inherited else "string"):
        out["type"] = prop.type
    if prop.description:
        out["description"] = prop.description
    if prop.min_count is not None:
        out["min_count"] = prop.min_count
    elif prop.required:
        out["required"] = True
    if prop.max_count != (inherited.max_count if inherited else 1):
        out["max_count"] = prop.max_count
    if prop.values is not None:
        out["values"] = list(prop.values)
    for key in ("default", "value", "minimum", "maximum", "unit"):
        value = getattr(prop, key)
        if value is not None:
            out[key] = list(value) if isinstance(value, tuple) else value
    if prop.iri:
        out["iri"] = prop.iri
    if prop.profiles:
        out["profiles"] = _plain(prop.profiles)
    return _with_names(out, prop)


def _with_names(out: dict[str, Any], named: NodeType | RelationType | Property) -> dict[str, Any]:
    if named.synonyms:
        out["synonyms"] = list(named.synonyms)
    if named.names:
        out["names"] = dict(named.names)
    return out


def _plain(profiles: Any) -> dict[str, Any]:
    return {name: dict(settings) for name, settings in profiles.items()}


class _Reader:
    """Turns one parsed model mapping into a HygmModel, naming where each problem is."""

    def __init__(self, where_from: str, resolver: ImportResolver | None = None) -> None:
        self.where_from = where_from
        self.resolver = resolver
        # The node types read so far, imported ones included, while reading a
        # model file's node types: what a redeclared property inherits from.
        self.known_types: tuple[NodeType, ...] | None = None

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
            identity = item.get("identity", "chunk")
            if identity not in IDENTITIES:
                raise self.fail(
                    "invalid_profile_value", where, f"identity must be one of {IDENTITIES}, got {identity!r}"
                )
            profiles = {"extraction": {"identity": identity}} if identity != "chunk" else {}
            node_types.append(self.build(NodeType, where, label, description, profiles=profiles))
        relation_types = [
            self.relation(item, f"relation_types[{index}]", _V1_RELATION) for index, item in enumerate(relations)
        ]
        return HygmModel(node_types=tuple(node_types), relation_types=tuple(relation_types))

    def v2(self, raw: dict) -> HygmModel:
        if "entity_types" not in raw:
            self.listed(raw, "node_types", required=True)
        self.known(raw, _V2_TOP, "", hint="version 2 calls entity_types node_types")
        nodes = self.listed(raw, "node_types", required=True)
        relations = self.listed(raw, "relation_types")
        imports = self.resolve(self.imports(raw))
        node_types = self.nodes(nodes, tuple(_imported(imports, "node_types").values()))
        relation_types = [
            self.relation(item, f"relation_types[{index}]", _V2_RELATION) for index, item in enumerate(relations)
        ]
        profiles = self.profiles(raw.get("profiles", {}), "profiles", "model")
        ontology = self.ontology(raw["ontology"]) if "ontology" in raw else None
        return HygmModel(
            node_types=_merged(imports, tuple(node_types), "node_types", "node label", self.where_from),
            relation_types=_merged(imports, tuple(relation_types), "relation_types", "relation type", self.where_from),
            profiles=profiles,
            ontology=ontology,
            imports=imports,
        )

    def resolve(self, paths: tuple[str, ...]) -> tuple[Import, ...]:
        """Each import with its model (4.12); with no resolver, a model that imports is `unknown_model`."""
        if paths and self.resolver is None:
            raise ModelError(
                "unknown_model",
                f"{self.where_from} imports {list(paths)}, and nothing resolves them: read it from a file, "
                "or pass a resolver",
            )
        return tuple(Import(path, self.resolver(path)) for path in paths) if self.resolver else ()

    def nodes(self, items: list, imported: tuple[NodeType, ...]) -> list[NodeType]:
        """The node types, read parents first so a redeclared property can fill in what it leaves unstated."""
        heads = []
        for index, item in enumerate(items):
            where = f"node_types[{index}]"
            label, _ = self.labeled(item, where)
            heads.append((label, self.labels(item, "parents", where)))
        built: dict[int, NodeType] = {}
        for index in _parents_first(heads):
            self.known_types = (*imported, *built.values())
            built[index] = self.node(items[index], f"node_types[{index}]")
        self.known_types = None
        return [built[index] for index in range(len(items))]

    def ontology(self, raw: Any) -> Ontology:
        where = "ontology"
        if not isinstance(raw, dict):
            raise self.fail("invalid_format", where, "must be a mapping")
        self.known(raw, _ONTOLOGY, where)
        scope = raw.get("scope", {})
        if not isinstance(scope, dict):
            raise self.fail("invalid_format", f"{where} scope", "must be a mapping")
        self.known(scope, _SCOPE, f"{where} scope")
        conventions = raw.get("conventions")
        if conventions is not None:
            if not isinstance(conventions, dict) or not all(isinstance(v, str) for v in conventions.values()):
                raise self.fail("invalid_format", f"{where} conventions", "must map convention names to strings")
            self.known(conventions, _CONVENTIONS, f"{where} conventions")
        return self.build(
            Ontology,
            where,
            name=self.optional_string(raw, "name", where),
            description=self.optional_string(raw, "description", where),
            iri=self.optional_string(raw, "iri", where),
            version_info=self.optional_string(raw, "version_info", where),
            includes=self.strings(scope, "includes", f"{where} scope"),
            excludes=self.strings(scope, "excludes", f"{where} scope"),
            competency_questions=self.strings(raw, "competency_questions", where),
            maintainers=self.strings(raw, "maintainers", where),
            conventions=self.build(Conventions, f"{where} conventions", **conventions)
            if conventions is not None
            else None,
        )

    def imports(self, raw: dict) -> tuple[str, ...]:
        paths = []
        for index, item in enumerate(self.listed(raw, "imports")):
            where = f"imports[{index}]"
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
                raise self.fail("invalid_format", where, "must be a mapping with a non-empty string 'path'")
            self.known(item, _IMPORT, where)
            paths.append(item["path"])
        return tuple(paths)

    def boolean(self, item: dict, key: str, where: str) -> bool:
        value = item.get(key, False)
        if not isinstance(value, bool):
            raise self.fail("invalid_format", where, f"{key} must be true or false")
        return value

    def names(self, item: dict, where: str) -> dict[str, str]:
        value = item.get("names", {})
        if not isinstance(value, dict) or not all(isinstance(name, str) for name in value.values()):
            raise self.fail("invalid_format", where, "names must map language tags to strings")
        return value

    def strings(self, item: dict, key: str, where: str) -> tuple[str, ...]:
        value = item.get(key, [])
        if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
            raise self.fail("invalid_format", where, f"{key} must be a list of strings")
        return tuple(value)

    def node(self, item: Any, where: str) -> NodeType:
        label, description = self.labeled(item, where)
        self.known(item, _V2_NODE, where)
        profiles = _without_default_identity(self.profiles(item.get("profiles", {}), f"{where} profiles", "node"))
        parents = self.labels(item, "parents", where)
        return self.build(
            NodeType,
            where,
            label,
            description,
            parents=parents,
            properties=self.properties(item, where, self.parents_view(label, parents)),
            key=self.labels(item, "key", where, what="list of property names"),
            iri=self.optional_string(item, "iri", where),
            profiles=profiles,
            abstract=self.boolean(item, "abstract", where),
            disjoint_with=self.labels(item, "disjoint_with", where),
            synonyms=self.strings(item, "synonyms", where),
            names=self.names(item, where),
            relation_restrictions=self.restrictions(item, where),
        )

    def restrictions(self, item: dict, where: str) -> tuple[RelationRestriction, ...]:
        found = []
        for index, raw in enumerate(self.listed(item, "relation_restrictions", where=where)):
            at = f"{where} relation_restrictions[{index}]"
            if not isinstance(raw, dict) or not isinstance(raw.get("relation"), str):
                raise self.fail("invalid_format", at, "must be a mapping with a string 'relation'")
            self.known(raw, _RESTRICTION, at)
            found.append(
                self.build(
                    RelationRestriction,
                    at,
                    raw["relation"],
                    end_labels=self.labels(raw, "end_labels", at) if "end_labels" in raw else None,
                    min_count=self.count(raw, "min_count", at, nullable=False),
                    max_count=self.count(raw, "max_count", at, nullable=False),
                )
            )
        return tuple(found)

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
            synonyms=self.strings(item, "synonyms", where),
            names=self.names(item, where),
            min_count=self.count(item, "min_count", where, nullable=False) or 0,
            max_count=self.count(item, "max_count", where, nullable=True),
            inverse_of=self.optional_string(item, "inverse_of", where),
            symmetric=self.boolean(item, "symmetric", where),
        )

    def parents_view(self, label: str, parents: tuple[str, ...]) -> tuple[str, HygmModel] | None:
        """The model a node type's redeclared properties inherit from, while reading a model file's node types."""
        if self.known_types is None:
            return None
        try:
            # First, so it shadows an imported or duplicate type of the same label.
            return label, HygmModel(node_types=(NodeType(label, parents=parents), *self.known_types))
        except ModelError:
            return None

    def properties(self, item: dict, where: str, owner: tuple[str, HygmModel] | None = None) -> tuple[Property, ...]:
        props = []
        for index, prop in enumerate(self.listed(item, "properties", where=where)):
            name = prop.get("name") if isinstance(prop, dict) else None
            inherited = owner[1].inherited(owner[0], name) if owner and isinstance(name, str) else None
            props.append(self.property(prop, f"{where} properties[{index}]", inherited))
        return tuple(props)

    def property(self, prop: Any, at: str, inherited: Property | None = None) -> Property:
        """One property; a redeclaration of `inherited` takes its unstated `type` and `max_count` from it (4.15)."""
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
        min_count = self.count(prop, "min_count", at, nullable=False)
        max_count = self.count(prop, "max_count", at, nullable=True)
        if "required" in prop and min_count is not None and prop["required"] != (min_count >= 1):
            raise self.fail("count_conflict", at, f"required is {prop['required']} but min_count is {min_count}")
        if inherited is not None and inherited.required and (prop.get("required") is False or min_count == 0):
            raise self.fail("property_conflict", at, f"makes {prop['name']!r} optional, which it inherits as required")
        return self.build(
            Property,
            at,
            prop["name"],
            type=prop.get("type", inherited.type if inherited else "string"),
            description=self.optional_string(prop, "description", at) or "",
            required=prop.get("required", False),
            values=tuple(values) if values is not None else None,
            iri=self.optional_string(prop, "iri", at),
            profiles=self.profiles(prop.get("profiles", {}), f"{at} profiles", "property"),
            synonyms=self.strings(prop, "synonyms", at),
            names=self.names(prop, at),
            min_count=min_count,
            max_count=max_count if "max_count" in prop else inherited.max_count if inherited else 1,
            default=prop.get("default"),
            value=prop.get("value"),
            minimum=prop.get("minimum"),
            maximum=prop.get("maximum"),
            unit=self.optional_string(prop, "unit", at),
        )

    def count(self, item: dict, key: str, where: str, *, nullable: bool) -> int | None:
        value = item.get(key)
        if value is None and (nullable or key not in item):
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            more = " or null" if nullable else ""
            raise self.fail("invalid_format", where, f"{key} must be a whole number{more}, got {value!r}")
        return value

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


def _parents_first(heads: list[tuple[str, tuple[str, ...]]]) -> list[int]:
    """Indexes of (label, parents) entries, each after the entries for its parents; a cycle is cut anywhere."""
    index_of: dict[str, int] = {}
    for index, (label, _) in enumerate(heads):
        index_of.setdefault(label, index)
    order: list[int] = []
    done: set[int] = set()

    def visit(index: int, path: frozenset[int]) -> None:
        if index in done or index in path:
            return
        for parent in heads[index][1]:
            if parent in index_of:
                visit(index_of[parent], path | {index})
        done.add(index)
        order.append(index)

    for index in range(len(heads)):
        visit(index, frozenset())
    return order


def _identity(node: NodeType) -> str:
    return setting(node.profiles, "extraction", "identity", "chunk")


def _without_default_identity(profiles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """`profiles` without an `extraction.identity` of `chunk`, the default, so both spellings read as one model."""
    extraction = profiles.get("extraction")
    if not extraction or extraction.get("identity") != "chunk":
        return profiles
    rest = {k: v for k, v in extraction.items() if k != "identity"}
    trimmed = {name: settings for name, settings in profiles.items() if name != "extraction"}
    if rest:
        trimmed["extraction"] = rest
    return trimmed
