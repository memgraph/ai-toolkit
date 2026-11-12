"""The structural graph model every strategy produces (docs/contract.md, section 4)."""

import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from .errors import ModelError
from .facets import FacetConflictError, check_facets, is_multi_valued, merge, refine
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


def _count(owner: object, key: str, what: str, *, nullable: bool) -> None:
    count = getattr(owner, key)
    if count is None and nullable:
        return
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ModelError("invalid_format", f"{what}: {key} must be a whole number, got {count!r}")


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
    """A property a node or relation type declares, with its facets (docs/contract.md, 4.5, 4.14).

    `required` is shorthand for `min_count: 1`, and the two are kept in step:
    a `min_count` of 1 or more makes the property required, and `min_count`
    is stored only above 1 (None otherwise), so each fact has one spelling.
    A property is multi-valued, a list, when `max_count` is None or above 1;
    `values`, the bounds and each list element's type apply per element.

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
        min_count: The fewest values an instance has, when above 1; see `least_count`.
        max_count: The most values an instance has; None for no limit, 0 to
            forbid the property on a subtype (4.15).
        default: Filled in when an instance is created without a value; not a constraint.
        value: The one value every instance has.
        minimum: The inclusive lower bound, on an ordered type (facets.ORDERED_TYPES).
        maximum: The inclusive upper bound, likewise.
        unit: The unit of a numeric value; documentation.

    Raises:
        ModelError: `count_conflict` for `required` with a `min_count` of 0 or
            counts the wrong way round; `invalid_value` for a bound, default or
            value the facets don't allow; `invalid_format` for a negative or
            non-integer count; and the codes below for the type and values.
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
    min_count: int | None = None
    max_count: int | None = 1
    default: Any = field(default=None, hash=False)
    value: Any = field(default=None, hash=False)
    minimum: Any = field(default=None, hash=False)
    maximum: Any = field(default=None, hash=False)
    unit: str | None = None

    @property
    def least_count(self) -> int:
        """The fewest values an instance has: `min_count`, else 1 if required, else 0."""
        return self.min_count if self.min_count is not None else int(self.required)

    @property
    def multi_valued(self) -> bool:
        """Whether a value is a list: `max_count` is None or above 1."""
        return is_multi_valued(self.max_count)

    def __post_init__(self) -> None:
        _identifier(self.name, "property name")
        _names_and_synonyms(self, f"Property {self.name!r}")
        self._normalize_counts()
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
        for key in ("default", "value"):
            if self.multi_valued and isinstance(getattr(self, key), list):
                object.__setattr__(self, key, tuple(getattr(self, key)))
        check_facets(self)

    def _normalize_counts(self) -> None:
        for key in ("min_count", "max_count"):
            count = getattr(self, key)
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                raise ModelError(
                    "invalid_format", f"Property {self.name!r}: {key} must be a whole number, got {count!r}"
                )
        if self.min_count is None:
            return
        if self.required and self.min_count == 0:
            raise ModelError("count_conflict", f"Property {self.name!r}: required, but min_count is 0")
        object.__setattr__(self, "required", self.min_count >= 1)
        object.__setattr__(self, "min_count", self.min_count if self.min_count > 1 else None)


@dataclass(frozen=True)
class RelationRestriction:
    """How a node type narrows a relation for its own instances (docs/contract.md, 4.16).

    Attributes:
        relation: The relation type narrowed; the node type must be able to start it.
        end_labels: Where the relation may end from this type's instances,
            each a descendant of (or one of) the end labels it narrows; None
            keeps the inherited ones.
        min_count: The fewest such relationships an instance starts; None keeps the inherited count.
        max_count: The most; None keeps the inherited count. A restriction
            only narrows, so it can't lift a limit.

    Raises:
        ModelError: `invalid_identifier` for a bad relation type or label,
            `invalid_format` for a negative count, `count_conflict` for
            `min_count` above `max_count`.
    """

    relation: str
    end_labels: tuple[str, ...] | None = None
    min_count: int | None = None
    max_count: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.relation, "relation type")
        what = f"Restriction of {self.relation!r}"
        if self.end_labels is not None:
            _tuple_of_labels(self, "end_labels", what)
        _count(self, "min_count", what, nullable=True)
        _count(self, "max_count", what, nullable=True)
        if self.min_count is not None and self.max_count is not None and self.min_count > self.max_count:
            raise ModelError(
                "count_conflict", f"{what}: min_count {self.min_count} is above max_count {self.max_count}"
            )


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
        relation_restrictions: Relations narrowed for this type's instances (4.16).
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
    relation_restrictions: tuple[RelationRestriction, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        what = f"Node type {self.label!r}"
        _tuple_of_labels(self, "parents", what)
        _tuple_of_labels(self, "key", what)
        _tuple_of_labels(self, "disjoint_with", what)
        _names_and_synonyms(self, what)
        object.__setattr__(self, "relation_restrictions", tuple(self.relation_restrictions))

    def restriction(self, relation: str) -> RelationRestriction | None:
        """This type's own restriction of `relation`, if it declares one."""
        return next((r for r in self.relation_restrictions if r.relation == relation), None)
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
        min_count: The fewest relationships of this type a start node has (4.16).
        max_count: The most; None for no limit. 1 is a single-valued slot.
        inverse_of: The relation type every one of these implies backwards:
            `(a)-[:R]->(b)` implies `(b)-[:S]->(a)`. Declared on one side;
            HygmModel.inverse() answers it from either.
        symmetric: `(a)-[:R]->(b)` implies `(b)-[:R]->(a)`.

    Raises:
        ModelError: `invalid_format` for a negative count, `count_conflict`
            for `min_count` above `max_count`.
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
    min_count: int = 0
    max_count: int | None = None
    inverse_of: str | None = None
    symmetric: bool = False

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        what = f"Relation type {self.label!r}"
        _tuple_of_labels(self, "start_labels", what)
        _tuple_of_labels(self, "end_labels", what)
        _names_and_synonyms(self, what)
        _count(self, "min_count", what, nullable=False)
        _count(self, "max_count", what, nullable=True)
        if self.max_count is not None and self.min_count > self.max_count:
            raise ModelError(
                "count_conflict", f"{what}: min_count {self.min_count} is above max_count {self.max_count}"
            )
        if self.inverse_of is not None:
            _identifier(self.inverse_of, "relation type")
        object.__setattr__(self, "properties", tuple(self.properties))

    @property
    def constrained(self) -> bool:
        """Whether either endpoint is restricted beyond "any declared label"."""
        return bool(self.start_labels or self.end_labels)


@dataclass(frozen=True)
class Individual:
    """An instance the ontology carries: reference data or an example (docs/contract.md, 4.17).

    Attributes:
        id: Stands for the individual in other individuals' `relations`; a
            valid identifier, unique among the model's individuals.
        type: The node type it is an instance of, and so of its ancestors.
        iri: Its IRI (`owl:NamedIndividual`), if it has one.
        properties: Values by property name, as JSON values: a list for a
            multi-valued property, ISO 8601 strings for dates.
        relations: Target individual ids by relation type. Inverse and
            symmetric relationships are implied, not repeated.

    Raises:
        ModelError: `invalid_identifier` for an id, property name or relation
            type that isn't one.
    """

    id: str
    type: str
    iri: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict, hash=False)
    relations: Mapping[str, tuple[str, ...]] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _identifier(self.id, "individual id")
        _identifier(self.type, "node label")
        for name in self.properties:
            _identifier(name, "property name")
        relations = {}
        for name, targets in self.relations.items():
            _identifier(name, "relation type")
            if isinstance(targets, str):
                raise TypeError(f"Individual {self.id!r}: relation {name!r} must list target ids, not a string")
            relations[name] = tuple(targets)
        object.__setattr__(self, "properties", dict(self.properties))
        object.__setattr__(self, "relations", relations)


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
        imports: The models this one builds on (4.12). Their types and
            individuals are in `node_types`, `relation_types` and
            `individuals` too, and `imported_labels()` says which types, so a
            writer writes them back as imports, not inline.
        individuals: The instances the ontology carries (4.17), imported ones first.
    """

    node_types: tuple[NodeType, ...]
    relation_types: tuple[RelationType, ...] = ()
    profiles: Profiles = field(default_factory=dict, hash=False)
    ontology: Ontology | None = None
    imports: tuple[Import, ...] = ()
    individuals: tuple[Individual, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_types", tuple(self.node_types))
        object.__setattr__(self, "relation_types", tuple(self.relation_types))
        object.__setattr__(self, "imports", tuple(self.imports))
        object.__setattr__(self, "individuals", tuple(self.individuals))

    def individual(self, id: str) -> Individual | None:
        return next((i for i in self.individuals if i.id == id), None)

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

    def relation_type(self, label: str) -> RelationType | None:
        return next((r for r in self.relation_types if r.label == label), None)

    def inverse(self, label: str) -> str | None:
        """The relation type `label` implies backwards (4.16), or None.

        That is its `inverse_of`, else the one declaring itself `label`'s
        inverse, else `label` itself when symmetric.
        """
        relation = self.relation_type(label)
        if relation is None:
            return None
        if relation.inverse_of is not None:
            return relation.inverse_of
        declaring = next((r.label for r in self.relation_types if r.inverse_of == label), None)
        return declaring or (label if relation.symmetric else None)

    def relation_for(self, label: str, relation: str) -> RelationType | None:
        """`relation` as instances of node type `label` have it: narrowed by every restriction they inherit (4.16).

        Restrictions apply from the topmost ancestor down; one from each of
        two parents combine to what both allow. The result keeps the
        relation's own start labels. None if `relation` isn't declared or
        `label`'s instances can't start it.
        """
        declared = self.relation_type(relation)
        if declared is None or self.node_type(label) is None or not self.may_start(label, declared):
            return None
        return self._restricted(label, declared, frozenset())

    def may_start(self, label: str, relation: RelationType) -> bool:
        """Whether a `label` instance may start `relation`: it is one of its start labels, or a descendant."""
        return not relation.start_labels or any(self.is_a(label, start) for start in relation.start_labels)

    def _restricted(self, label: str, relation: RelationType, visiting: frozenset[str]) -> RelationType:
        node = self.node_type(label)
        effective = relation
        inherited = [
            self._restricted(parent, relation, visiting | {label})
            for parent in (node.parents if node else ())
            if parent not in visiting and parent != label and self.node_type(parent) is not None
        ]
        for one in inherited:
            effective = self.narrowed(effective, one.end_labels, one.min_count, one.max_count)
        own = node.restriction(relation.label) if node else None
        if own is not None:
            effective = self.narrowed(effective, own.end_labels, own.min_count, own.max_count)
        return effective

    def narrowed(
        self, relation: RelationType, end_labels: tuple[str, ...] | None, min_count: int | None, max_count: int | None
    ) -> RelationType:
        """`relation` narrowed to what it and the given end labels and counts both allow.

        End labels narrow to the labels one side names that the other admits
        (each named label and its descendants); where nothing both allow, the
        given ones stand. Counts take the higher minimum and lower maximum,
        which may cross. validate_model() reports both.
        """
        ends = relation.end_labels
        if end_labels is not None and end_labels != relation.end_labels:
            if not relation.end_labels:
                ends = end_labels
            else:
                # Empty would read as "any label", so with nothing both allow, the narrower side stands.
                ends = (
                    tuple(
                        dict.fromkeys(
                            [e for e in end_labels if any(self.is_a(e, x) for x in relation.end_labels)]
                            + [x for x in relation.end_labels if any(self.is_a(x, e) for e in end_labels)]
                        )
                    )
                    or end_labels
                )
        least = max(relation.min_count, min_count or 0)
        limits = [m for m in (relation.max_count, max_count) if m is not None]
        most = min(limits) if limits else None
        return _unchecked(relation, end_labels=ends, min_count=least, max_count=most)

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
        """The properties a `label` instance has, with their effective facets; inherited ones first.

        A redeclaration narrows what `label` inherits, and takes what it
        leaves unstated from it (facets.refine(), docs/contract.md 4.15).
        Where it contradicts the inherited facets, it is as declared, for
        validate_model() to report. One inherited from several
        parents has the narrowest facets they all allow; where none exist, the
        first parent's, for validate_model() to report as `property_conflict`.
        """
        return tuple(self._effective(label, frozenset()).values())

    def inherited(self, label: str, name: str) -> Property | None:
        """Property `name` as `label`'s parents give it, before `label`'s own redeclaration; None if they don't."""
        return self._from_parents(label, frozenset()).get(name)

    def _effective(self, label: str, visiting: frozenset[str]) -> dict[str, Property]:
        effective = self._from_parents(label, visiting)
        node = self.node_type(label)
        for prop in node.properties if node else ():
            inherited = effective.get(prop.name)
            try:
                effective[prop.name] = refine(inherited, prop) if inherited else prop
            except ModelError:
                effective[prop.name] = prop
        return effective

    def _from_parents(self, label: str, visiting: frozenset[str]) -> dict[str, Property]:
        node = self.node_type(label)
        merged: dict[str, Property] = {}
        for parent in node.parents if node else ():
            if parent in visiting or parent == label or self.node_type(parent) is None:
                continue
            for prop in self._effective(parent, visiting | {label}).values():
                seen = merged.get(prop.name)
                if seen is None or seen == prop:
                    merged.setdefault(prop.name, prop)
                    continue
                # With no facets both allow, the first parent's stand, for validate_model() to report.
                with suppress(FacetConflictError):
                    merged[prop.name] = merge(seen, prop)
        return merged

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
            "individuals": bool(self.individuals),
            "abstract": any(t.abstract for t in self.node_types),
            "relation_restrictions": any(t.relation_restrictions for t in self.node_types),
            "relation_counts": any(r.min_count or r.max_count is not None for r in self.relation_types),
            "inverse_of": any(r.inverse_of for r in self.relation_types),
            "symmetric": any(r.symmetric for r in self.relation_types),
            "disjoint_with": any(t.disjoint_with for t in self.node_types),
            "synonyms": any(t.synonyms for t in self._named()),
            "names": any(t.names for t in self._named()),
        }
        return tuple(name for name, present in used.items() if present)

    def _named(self) -> tuple[NodeType | RelationType | Property, ...]:
        """Everything that may carry `synonyms` and `names`: types and their properties."""
        types = (*self.node_types, *self.relation_types)
        return (*types, *(p for t in types for p in t.properties))


def _unchecked(relation: RelationType, **changes: Any) -> RelationType:
    """`relation` with `changes`, skipping __post_init__'s checks.

    A narrowed relation's counts may cross; validate_model() reports that,
    rather than a query raising.
    """
    narrowed = object.__new__(RelationType)
    for spec in fields(RelationType):
        object.__setattr__(narrowed, spec.name, changes.get(spec.name, getattr(relation, spec.name)))
    return narrowed
