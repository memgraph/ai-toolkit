"""The structural graph model every strategy produces (docs/contract.md, section 4)."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .errors import ModelError
from .identifiers import is_valid_identifier

Identity = Literal["global", "chunk", "span"]

#: How a node type's mentions become nodes (#346, #361):
#:   global -- one node per normalized text across every chunk (named things that recur)
#:   chunk  -- one node per normalized text per chunk (generic nouns)
#:   span   -- one node per mention (values: every "3" in a session is a different fact)
IDENTITIES: tuple[Identity, ...] = ("global", "chunk", "span")

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
    """

    name: str
    type: PropertyType = "string"
    description: str = ""
    required: bool = False
    values: tuple[str, ...] | None = None
    iri: str | None = None
    profiles: Profiles = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _identifier(self.name, "property name")
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
        description: What the type covers. Extractors that read descriptions
            (GLiNER2's entity specs, LightRAG's prompt) steer on it.
        identity: How mentions of this type are merged into nodes; see
            IDENTITIES. The `extraction` profile's setting, kept as a field
            because every extractor reads it.
        parents: Supertypes: every instance of this type is also one of each.
        properties: Properties declared here; inherited ones are not repeated.
        key: Property names whose values identify one entity across sources.
        iri: The type's IRI when it came from, or must round-trip to, RDF/OWL.
        profiles: Other profile settings for this type.
    """

    label: str
    description: str = ""
    identity: Identity = "chunk"
    parents: tuple[str, ...] = ()
    properties: tuple[Property, ...] = ()
    key: tuple[str, ...] = ()
    iri: str | None = None
    profiles: Profiles = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        if self.identity not in IDENTITIES:
            raise ModelError(
                "invalid_profile_value",
                f"Node type {self.label!r}: identity must be one of {IDENTITIES}, got {self.identity!r}",
            )
        what = f"Node type {self.label!r}"
        _tuple_of_labels(self, "parents", what)
        _tuple_of_labels(self, "key", what)
        object.__setattr__(self, "properties", tuple(self.properties))


@dataclass(frozen=True)
class RelationType:
    """A relationship type and the node labels it may connect.

    Attributes:
        label: The Memgraph relationship type, a valid Cypher identifier.
        description: Documentation only for GLiNER2, whose joint compiler drops
            relation descriptions; the label is its only steering surface (#360).
        start_labels: Labels the relationship may start at, each admitting its
            descendants. Empty means any declared label -- OWL's rdfs:domain,
            sql2graph's start_node_labels.
        end_labels: Labels the relationship may end at, likewise. Empty means
            any declared label -- OWL's rdfs:range.
        properties: Properties a relationship of this type carries.
        iri: The type's IRI when it came from, or must round-trip to, RDF/OWL.
        profiles: Profile settings for this type.
    """

    label: str
    description: str = ""
    start_labels: tuple[str, ...] = ()
    end_labels: tuple[str, ...] = ()
    properties: tuple[Property, ...] = ()
    iri: str | None = None
    profiles: Profiles = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        what = f"Relation type {self.label!r}"
        _tuple_of_labels(self, "start_labels", what)
        _tuple_of_labels(self, "end_labels", what)
        object.__setattr__(self, "properties", tuple(self.properties))

    @property
    def constrained(self) -> bool:
        """Whether either endpoint is restricted beyond "any declared label"."""
        return bool(self.start_labels or self.end_labels)


@dataclass(frozen=True)
class HygmModel:
    """A graph model: node types and the relation types between them.

    The hierarchy queries (ancestors, descendants, properties_of, key_of)
    assume a model that passed validate_model(); on a parent cycle they stop
    rather than loop.
    """

    node_types: tuple[NodeType, ...]
    relation_types: tuple[RelationType, ...] = ()
    profiles: Profiles = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_types", tuple(self.node_types))
        object.__setattr__(self, "relation_types", tuple(self.relation_types))

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

        Extractors that cannot express an unconstrained endpoint (GLiNER2's
        JointSchema rejects an empty head/tail, #345) consume this rather than
        the raw field. With no `parents` anywhere it is the raw field.
        """
        labels = relation.start_labels if side == "start" else relation.end_labels
        if not labels:
            return self.node_labels()
        expanded = dict.fromkeys(labels)
        for label in labels:
            expanded.update(dict.fromkeys(self.descendants(label)))
        return tuple(expanded)

    def uses_hierarchy_or_properties(self) -> tuple[str, ...]:
        """Which of `parents`, `properties` and `key` any type declares, for readers that refuse them."""
        used = {
            "parents": any(t.parents for t in self.node_types),
            "properties": any(t.properties for t in (*self.node_types, *self.relation_types)),
            "key": any(t.key for t in self.node_types),
        }
        return tuple(name for name, present in used.items() if present)
