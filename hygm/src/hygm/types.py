"""The structural graph model every strategy produces (docs/contract.md, section 4)."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .errors import ModelError
from .identifiers import is_valid_identifier

PropertyType = Literal["string", "integer", "float", "decimal", "boolean", "date", "datetime", "json"]

#: The contract's logical property types; each reader maps them to its own physical types.
PROPERTY_TYPES: tuple[PropertyType, ...] = (
    "string",
    "integer",
    "float",
    "decimal",
    "boolean",
    "date",
    "datetime",
    "json",
)

#: Profile settings by profile name: what one kind of consumer needs from a
#: type, kept out of the core (docs/contract.md, section 5).
Profiles = Mapping[str, Mapping[str, Any]]

#: Display names and translations by BCP 47 language tag (docs/contract.md, 4.13).
Names = Mapping[str, str]

# BCP 47's shape only (a primary language subtag, then subtags): hygm doesn't
# carry the IANA registry, so `xx-Fake` passes and `en_US` or `english!` don't.
_LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*$")


def _identifier(value: str, role: str) -> None:
    if not is_valid_identifier(value):
        raise ModelError(
            "invalid_identifier",
            f"Invalid {role} {value!r}: must be a valid identifier (letters, digits, "
            "underscore, not starting with a digit) to use directly in a Cypher query",
        )


def _tuple_of_labels(owner: object, attribute: str, what: str) -> None:
    value = getattr(owner, attribute)
    if isinstance(value, str):
        raise TypeError(f"{what}: {attribute} must be a tuple of labels, not a string")
    object.__setattr__(owner, attribute, tuple(value))


def _names_and_synonyms(owner: Any, what: str) -> None:
    _tuple_of_labels(owner, "synonyms", what)
    names = dict(owner.names)
    for tag, name in names.items():
        if not isinstance(tag, str) or not _LANGUAGE_TAG.match(tag) or not isinstance(name, str):
            raise ModelError(
                "invalid_format", f"{what}: names must map BCP 47 language tags to strings, got {tag!r}: {name!r}"
            )
    object.__setattr__(owner, "names", names)


#: The naming styles `Conventions` may ask for (docs/contract.md, 4.11).
CASES: tuple[str, ...] = ("upper_camel", "lower_camel", "snake", "upper_snake")

#: The grammatical numbers `Conventions.number` may ask for.
NUMBERS: tuple[str, ...] = ("singular", "plural")


@dataclass(frozen=True)
class Conventions:
    """The naming conventions a model follows, which lint checks (docs/contract.md, 4.11).

    Attributes:
        node_case: The case of node labels, one of CASES, or None for unstated.
        property_case: The case of property names, likewise.
        relation_case: The case of relation types, likewise.
        number: Whether type labels are `singular` or `plural`, or None.

    Raises:
        ModelError: invalid_format for a case or number outside CASES or NUMBERS.
    """

    node_case: str | None = None
    property_case: str | None = None
    relation_case: str | None = None
    number: str | None = None

    def __post_init__(self) -> None:
        for name in ("node_case", "property_case", "relation_case"):
            value = getattr(self, name)
            if value is not None and value not in CASES:
                raise ModelError("invalid_format", f"conventions {name} must be one of {CASES}, got {value!r}")
        if self.number is not None and self.number not in NUMBERS:
            raise ModelError("invalid_format", f"conventions number must be one of {NUMBERS}, got {self.number!r}")


@dataclass(frozen=True)
class Ontology:
    """What a model is for: its scope, the questions it answers, who keeps it (docs/contract.md, 4.11).

    Nothing here changes what the model means; it documents it, and lint reads
    `conventions`.

    Attributes:
        name: A short name.
        description: The domain the ontology covers.
        iri: The ontology's IRI (`owl:Ontology`).
        version_info: A free-form release name (`owl:versionInfo`), not the format version.
        includes: What is deliberately in scope (`scope.includes`).
        excludes: What is deliberately out of scope (`scope.excludes`).
        competency_questions: The questions the ontology must answer.
        maintainers: Who maintains it.
        conventions: The naming conventions, or None when unstated.
    """

    name: str | None = None
    description: str | None = None
    iri: str | None = None
    version_info: str | None = None
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    competency_questions: tuple[str, ...] = ()
    maintainers: tuple[str, ...] = ()
    conventions: Conventions | None = None

    def __post_init__(self) -> None:
        for name in ("includes", "excludes", "competency_questions", "maintainers"):
            _tuple_of_labels(self, name, "Ontology")


@dataclass(frozen=True)
class Property:
    """A property a node or relation type declares.

    Attributes:
        name: The property key, a valid Cypher identifier.
        type: One of PROPERTY_TYPES.
        description: What the property holds.
        required: Every instance has a non-null value.
        values: The allowed values; only on a string property.
        iri: The property's IRI when it came from, or must round-trip to, RDF/OWL.
        profiles: Profile settings for this property.
        synonyms: Other names for the same property (4.13).
        names: Display names and translations by language tag (4.13).
    """

    name: str
    type: PropertyType = "string"
    description: str = ""
    required: bool = False
    values: tuple[str, ...] | None = None
    iri: str | None = None
    profiles: Profiles = field(default_factory=dict, hash=False)
    synonyms: tuple[str, ...] = ()
    names: Names = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _identifier(self.name, "property name")
        _names_and_synonyms(self, f"Property {self.name!r}")
        if self.type not in PROPERTY_TYPES:
            raise ModelError(
                "unknown_property_type",
                f"Property {self.name!r}: type must be one of {PROPERTY_TYPES}, got {self.type!r}",
            )
        if self.values is not None:
            if self.type != "string":
                raise ModelError(
                    "values_not_string",
                    f"Property {self.name!r}: values are only allowed on a string property, not {self.type!r}",
                )
            object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class NodeType:
    """A node label the model declares.

    Attributes:
        label: The Memgraph label, a valid Cypher identifier.
        description: What the type covers, for people and for consumers that
            read descriptions.
        parents: Supertypes: every instance of this type is also one of each.
        properties: Properties declared here; inherited ones are not repeated.
        key: Property names whose values identify one entity across sources.
        iri: The type's IRI when it came from, or must round-trip to, RDF/OWL.
        profiles: Profile settings for this type, each owned by its consumer.
        abstract: The type has no direct instances; each belongs to a descendant (4.13).
        disjoint_with: Types no instance of this one also belongs to, and so
            none of their descendants either. Declaring it on one side is
            enough; HygmModel.disjoint() answers it both ways.
        synonyms: Other names for the same concept (4.13).
        names: Display names and translations by language tag (4.13).
    """

    label: str
    description: str = ""
    parents: tuple[str, ...] = ()
    properties: tuple[Property, ...] = ()
    key: tuple[str, ...] = ()
    iri: str | None = None
    profiles: Profiles = field(default_factory=dict, hash=False)
    abstract: bool = False
    disjoint_with: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    names: Names = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        what = f"Node type {self.label!r}"
        _tuple_of_labels(self, "parents", what)
        _tuple_of_labels(self, "key", what)
        _tuple_of_labels(self, "disjoint_with", what)
        _names_and_synonyms(self, what)
        object.__setattr__(self, "properties", tuple(self.properties))


@dataclass(frozen=True)
class RelationType:
    """A relationship type and the node labels it may connect.

    Attributes:
        label: The Memgraph relationship type, a valid Cypher identifier.
        description: What the relationship means.
        start_labels: Labels the relationship may start at, each admitting its
            descendants. Empty means any declared label -- OWL's rdfs:domain.
        end_labels: Labels the relationship may end at, likewise. Empty means
            any declared label -- OWL's rdfs:range.
        properties: Properties a relationship of this type carries.
        iri: The type's IRI when it came from, or must round-trip to, RDF/OWL.
        profiles: Profile settings for this type.
        synonyms: Other names for the same relationship (4.13).
        names: Display names and translations by language tag (4.13).
    """

    label: str
    description: str = ""
    start_labels: tuple[str, ...] = ()
    end_labels: tuple[str, ...] = ()
    properties: tuple[Property, ...] = ()
    iri: str | None = None
    profiles: Profiles = field(default_factory=dict, hash=False)
    synonyms: tuple[str, ...] = ()
    names: Names = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        what = f"Relation type {self.label!r}"
        _tuple_of_labels(self, "start_labels", what)
        _tuple_of_labels(self, "end_labels", what)
        _names_and_synonyms(self, what)
        object.__setattr__(self, "properties", tuple(self.properties))

    @property
    def constrained(self) -> bool:
        """Whether either endpoint is restricted beyond "any declared label"."""
        return bool(self.start_labels or self.end_labels)


@dataclass(frozen=True)
class Import:
    """One model another builds on (docs/contract.md, 4.12).

    Attributes:
        path: The imported file, as the importing file names it: relative to the importing file.
        model: The imported model, its own imports already resolved. Its node
            and relation types are part of the importing model.
    """

    path: str
    model: "HygmModel"


@dataclass(frozen=True)
class HygmModel:
    """A graph model: node types and the relation types between them.

    The hierarchy queries (ancestors, descendants, properties_of, key_of)
    assume a model that passed validate_model(); on a parent cycle they stop
    rather than loop.

    Attributes:
        node_types: Every node type, imported ones included, imported first.
        relation_types: Every relation type, likewise.
        profiles: Model-wide profile settings.
        ontology: What the model is for (4.11), or None.
        imports: The models this one builds on (4.12). Their types are in
            `node_types` and `relation_types` too, and `imported_labels()`
            says which, so a writer writes them back as imports, not inline.
    """

    node_types: tuple[NodeType, ...]
    relation_types: tuple[RelationType, ...] = ()
    profiles: Profiles = field(default_factory=dict, hash=False)
    ontology: Ontology | None = None
    imports: tuple[Import, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_types", tuple(self.node_types))
        object.__setattr__(self, "relation_types", tuple(self.relation_types))
        object.__setattr__(self, "imports", tuple(self.imports))

    def imported_labels(self) -> tuple[frozenset[str], frozenset[str]]:
        """The node labels and the relation types that come from `imports`, transitively."""
        nodes = frozenset(label for i in self.imports for label in i.model.node_labels())
        relations = frozenset(label for i in self.imports for label in i.model.relation_labels())
        return nodes, relations

    def node_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.node_types)

    def relation_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.relation_types)

    def node_type(self, label: str) -> NodeType | None:
        return next((t for t in self.node_types if t.label == label), None)

    def ancestors(self, label: str) -> tuple[str, ...]:
        """Every supertype of `label`, nearest first. Parents that aren't declared are skipped."""
        found: list[str] = []
        frontier = [label]
        while frontier:
            node = self.node_type(frontier.pop(0))
            for parent in node.parents if node else ():
                if parent not in found and parent != label and self.node_type(parent):
                    found.append(parent)
                    frontier.append(parent)
        return tuple(found)

    def disjoint(self, a: str, b: str) -> bool:
        """Whether no instance is both an `a` and a `b` (4.13).

        Symmetric, and inherited: if `RedWine` and `WhiteWine` are disjoint,
        so is every descendant of one with every descendant of the other.
        """
        a_side = (a, *self.ancestors(a))
        b_side = (b, *self.ancestors(b))
        for x in a_side:
            for y in b_side:
                x_node, y_node = self.node_type(x), self.node_type(y)
                if (x_node and y in x_node.disjoint_with) or (y_node and x in y_node.disjoint_with):
                    return True
        return False

    def descendants(self, label: str) -> tuple[str, ...]:
        """Every subtype of `label`, in declaration order."""
        return tuple(t.label for t in self.node_types if t.label != label and label in self.ancestors(t.label))

    def is_a(self, label: str, other: str) -> bool:
        """Whether every `label` instance is an `other` instance: the same type or a descendant."""
        return label == other or other in self.ancestors(label)

    def properties_of(self, label: str) -> tuple[Property, ...]:
        """The properties a `label` instance has, inherited ones first.

        A redeclared property takes the nearest description and is required if
        any declaration says so (docs/contract.md, 4.7).
        """
        effective: dict[str, Property] = {}
        for owner in (*reversed(self.ancestors(label)), label):
            node = self.node_type(owner)
            for prop in node.properties if node else ():
                inherited = effective.get(prop.name)
                if inherited is None:
                    effective[prop.name] = prop
                else:
                    effective[prop.name] = replace(prop, required=inherited.required or prop.required)
        return tuple(effective.values())

    def key_of(self, label: str) -> tuple[str, ...]:
        """The key identifying a `label` instance: its own, else its nearest ancestor's; () if none."""
        for owner in (label, *self.ancestors(label)):
            node = self.node_type(owner)
            if node and node.key:
                return node.key
        return ()

    def endpoint_labels(self, relation: RelationType, side: Literal["start", "end"]) -> tuple[str, ...]:
        """The labels `relation` may connect at `side`: each named label and its descendants, or every declared label.

        For a consumer that can't express an unconstrained endpoint. With no
        `parents` anywhere it is the raw field.
        """
        labels = relation.start_labels if side == "start" else relation.end_labels
        if not labels:
            return self.node_labels()
        expanded = dict.fromkeys(labels)
        for label in labels:
            expanded.update(dict.fromkeys(self.descendants(label)))
        return tuple(expanded)

    def uses_hierarchy_or_properties(self) -> tuple[str, ...]:
        """The keys beyond labels, descriptions, endpoints and profiles the model uses.

        For readers that rebuild types from those alone and so must refuse
        the rest rather than drop it: `parents`, `properties`, `key`, then
        the keys of docs/contract.md 4.11 onwards.
        """
        used = {
            "parents": any(t.parents for t in self.node_types),
            "properties": any(t.properties for t in (*self.node_types, *self.relation_types)),
            "key": any(t.key for t in self.node_types),
            "ontology": self.ontology is not None,
            "imports": bool(self.imports),
            "abstract": any(t.abstract for t in self.node_types),
            "disjoint_with": any(t.disjoint_with for t in self.node_types),
            "synonyms": any(t.synonyms for t in self._named()),
            "names": any(t.names for t in self._named()),
        }
        return tuple(name for name, present in used.items() if present)

    def _named(self) -> tuple[NodeType | RelationType | Property, ...]:
        """Everything that may carry `synonyms` and `names`: types and their properties."""
        types = (*self.node_types, *self.relation_types)
        return (*types, *(p for t in types for p in t.properties))
