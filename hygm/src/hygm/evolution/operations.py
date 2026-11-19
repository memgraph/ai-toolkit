"""The operations a model changes by, and their JSON form (docs/contract.md, section 11).

Each operation is a frozen dataclass naming what it changes; `apply` gives
it meaning. Its JSON form is a mapping with an ``op`` key and one key per
field, embedded types written as version-2 model entries. Reading is as
strict as reading a model file: an unknown ``op`` or key is refused.
"""

from collections.abc import Callable, Mapping
from dataclasses import MISSING, Field, dataclass, field, fields
from types import UnionType
from typing import Any, ClassVar, Literal, get_args, get_origin

from ..errors import ModelError
from ..format import (
    ImportResolver,
    check_profile,
    imports_from_mapping,
    imports_to_mapping,
    individual_from_mapping,
    individual_to_mapping,
    node_type_from_mapping,
    node_type_to_mapping,
    ontology_from_mapping,
    ontology_to_mapping,
    property_from_mapping,
    property_to_mapping,
    relation_type_from_mapping,
    relation_type_to_mapping,
    restriction_from_mapping,
    restriction_to_mapping,
)
from ..types import (
    Import,
    Individual,
    NodeType,
    Ontology,
    Property,
    RelationRestriction,
    RelationType,
    _count,
    _identifier,
    _names,
)

Kind = Literal["node", "relation"]
ProfileLevel = Literal["model", "node", "relation"]

_KINDS: tuple[str, ...] = ("node", "relation")


def _tuple(owner: object, attribute: str) -> None:
    value = getattr(owner, attribute)
    if isinstance(value, str):
        raise TypeError(f"{type(owner).__name__}: {attribute} must be a tuple, not a string")
    object.__setattr__(owner, attribute, tuple(value))


def _boolean(owner: object, attribute: str) -> None:
    if not isinstance(getattr(owner, attribute), bool):
        raise ModelError("invalid_format", f"{type(owner).__name__}: {attribute} must be true or false")


def _kind(kind: str, allowed: tuple[str, ...] = _KINDS) -> None:
    if kind not in allowed:
        raise ModelError("invalid_format", f"kind must be one of {allowed}, got {kind!r}")


@dataclass(frozen=True)
class AddNodeType:
    """Declare `node_type`. Refused if its label is taken."""

    op: ClassVar[str] = "add_node_type"
    node_type: NodeType


@dataclass(frozen=True)
class RemoveNodeType:
    """Drop the node type `label`. References to it are left for later operations, or validation, to settle."""

    op: ClassVar[str] = "remove_node_type"
    label: str

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")


@dataclass(frozen=True)
class RenameNodeType:
    """Rename node type `label` to `to`, and every parent, endpoint and profile setting naming it."""

    op: ClassVar[str] = "rename_node_type"
    label: str
    to: str

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        _identifier(self.to, "node label")


@dataclass(frozen=True)
class MergeNodeTypes:
    """Fold node type `label` into `into`: `label` is dropped and everything naming it names `into`.

    The survivor keeps its own definition, as in derivation.
    """

    op: ClassVar[str] = "merge_node_types"
    label: str
    into: str

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        _identifier(self.into, "node label")


@dataclass(frozen=True)
class AddRelationType:
    """Declare `relation_type`. Refused if its label is taken."""

    op: ClassVar[str] = "add_relation_type"
    relation_type: RelationType


@dataclass(frozen=True)
class RemoveRelationType:
    """Drop the relation type `label`."""

    op: ClassVar[str] = "remove_relation_type"
    label: str

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")


@dataclass(frozen=True)
class RenameRelationType:
    """Rename relation type `label` to `to`."""

    op: ClassVar[str] = "rename_relation_type"
    label: str
    to: str

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        _identifier(self.to, "relation type")


@dataclass(frozen=True)
class MergeRelationTypes:
    """Fold relation type `label` into `into`, whose endpoints become the union of both.

    An empty side means any declared label, so a union with one stays empty.
    """

    op: ClassVar[str] = "merge_relation_types"
    label: str
    into: str

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        _identifier(self.into, "relation type")


@dataclass(frozen=True)
class AddProperty:
    """Declare `property` on the `kind` type `label`. Refused if the type already declares its name."""

    op: ClassVar[str] = "add_property"
    kind: Kind
    label: str
    property: Property

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")


@dataclass(frozen=True)
class RemoveProperty:
    """Drop property `name` from the `kind` type `label`. A key naming it is left for validation."""

    op: ClassVar[str] = "remove_property"
    kind: Kind
    label: str
    name: str

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")
        _identifier(self.name, "property name")


@dataclass(frozen=True)
class RenameProperty:
    """Rename property `name` of the `kind` type `label` to `to`.

    On a node type the rename also reaches every descendant that redeclares
    the property, and every key among them naming it.
    """

    op: ClassVar[str] = "rename_property"
    kind: Kind
    label: str
    name: str
    to: str

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")
        _identifier(self.name, "property name")
        _identifier(self.to, "property name")


@dataclass(frozen=True)
class SetProperty:
    """Replace the declaration of `property.name` on the `kind` type `label`, in place."""

    op: ClassVar[str] = "set_property"
    kind: Kind
    label: str
    property: Property

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")


@dataclass(frozen=True)
class SetDescription:
    """Set the description of the `kind` type `label`."""

    op: ClassVar[str] = "set_description"
    kind: Kind
    label: str
    description: str

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")


@dataclass(frozen=True)
class SetIri:
    """Set, or with None clear, the IRI of the `kind` type `label`."""

    op: ClassVar[str] = "set_iri"
    kind: Kind
    label: str
    iri: str | None

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")


@dataclass(frozen=True)
class SetParents:
    """Set the supertypes of node type `label`."""

    op: ClassVar[str] = "set_parents"
    label: str
    parents: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        _tuple(self, "parents")


@dataclass(frozen=True)
class SetKey:
    """Set the key of node type `label`; () clears it."""

    op: ClassVar[str] = "set_key"
    label: str
    key: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        _tuple(self, "key")


@dataclass(frozen=True)
class SetEndpoints:
    """Set the labels relation type `label` may start and end at; () means any declared label."""

    op: ClassVar[str] = "set_endpoints"
    label: str
    start_labels: tuple[str, ...]
    end_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        _tuple(self, "start_labels")
        _tuple(self, "end_labels")


@dataclass(frozen=True)
class SetProfile:
    """Set one profile's settings, or with `settings` None remove the profile, at one place in the model.

    `kind` "model" is the top level. "node" or "relation" is the type
    `label`, or with `property_name` that type's property. A node type's
    ``extraction`` settings include ``identity``, as in a model file.
    Settings are checked as a model file's are.
    """

    op: ClassVar[str] = "set_profile"
    profile: str
    settings: Mapping[str, Any] | None = field(hash=False)
    kind: ProfileLevel = "model"
    label: str | None = None
    property_name: str | None = None

    def __post_init__(self) -> None:
        _kind(self.kind, ("model", *_KINDS))
        if self.kind == "model" and (self.label is not None or self.property_name is not None):
            raise ModelError("invalid_format", "set_profile: a model-level profile takes no label or property")
        if self.kind != "model":
            if self.label is None:
                raise ModelError("invalid_format", f"set_profile: a {self.kind}-level profile needs a label")
            _identifier(self.label, f"{self.kind} label")
        if self.property_name is not None:
            _identifier(self.property_name, "property name")
        if self.settings is not None:
            object.__setattr__(self, "settings", dict(self.settings))
        check_profile(self.profile, self.settings, self.level, f"Operation {self.op}")

    @property
    def level(self) -> str:
        """Where the profile sits, as the profile registry names levels: model, node, relation or property."""
        return "property" if self.property_name is not None else self.kind


@dataclass(frozen=True)
class OrderNodeTypes:
    """Put the node types in `labels` order; `labels` must name each declared type once."""

    op: ClassVar[str] = "order_node_types"
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _tuple(self, "labels")


@dataclass(frozen=True)
class OrderRelationTypes:
    """Put the relation types in `labels` order; `labels` must name each declared type once."""

    op: ClassVar[str] = "order_relation_types"
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _tuple(self, "labels")


@dataclass(frozen=True)
class OrderProperties:
    """Put the properties the `kind` type `label` declares in `names` order; each must be named once."""

    op: ClassVar[str] = "order_properties"
    kind: Kind
    label: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")
        _tuple(self, "names")


@dataclass(frozen=True)
class SetOntology:
    """Set, or with None clear, the model's `ontology` block (4.11)."""

    op: ClassVar[str] = "set_ontology"
    ontology: Ontology | None


@dataclass(frozen=True)
class SetImports:
    """Set the models this one builds on (4.12).

    Only the record changes. The types and individuals an import declares
    come and go through their own operations, and apply() refuses a result
    that lacks or changes one an import declares.
    """

    op: ClassVar[str] = "set_imports"
    imports: tuple[Import, ...]

    def __post_init__(self) -> None:
        _tuple(self, "imports")


@dataclass(frozen=True)
class SetAbstract:
    """Set whether node type `label` is abstract (4.13)."""

    op: ClassVar[str] = "set_abstract"
    label: str
    abstract: bool

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        _boolean(self, "abstract")


@dataclass(frozen=True)
class SetDisjointWith:
    """Set the types node type `label` is declared disjoint with (4.13)."""

    op: ClassVar[str] = "set_disjoint_with"
    label: str
    disjoint_with: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        _tuple(self, "disjoint_with")


@dataclass(frozen=True)
class SetSynonyms:
    """Set the synonyms of the `kind` type `label` (4.13)."""

    op: ClassVar[str] = "set_synonyms"
    kind: Kind
    label: str
    synonyms: tuple[str, ...]

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")
        _tuple(self, "synonyms")


@dataclass(frozen=True)
class SetNames:
    """Set the display names, by language tag, of the `kind` type `label` (4.13)."""

    op: ClassVar[str] = "set_names"
    kind: Kind
    label: str
    names: Mapping[str, str] = field(hash=False)

    def __post_init__(self) -> None:
        _kind(self.kind)
        _identifier(self.label, f"{self.kind} label")
        _names(self, f"Operation {self.op}")


@dataclass(frozen=True)
class SetRelationRestrictions:
    """Set how node type `label` narrows the relations it starts (4.16)."""

    op: ClassVar[str] = "set_relation_restrictions"
    label: str
    relation_restrictions: tuple[RelationRestriction, ...]

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        _tuple(self, "relation_restrictions")


@dataclass(frozen=True)
class SetRelationCounts:
    """Set how many relationships of type `label` a start node has (4.16); `max_count` None is no limit.

    Both counts at once, so the pair is never seen half-changed and out of order.
    """

    op: ClassVar[str] = "set_relation_counts"
    label: str
    min_count: int
    max_count: int | None

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        _count(self, "min_count", f"Operation {self.op}", nullable=False)
        _count(self, "max_count", f"Operation {self.op}", nullable=True)
        if self.max_count is not None and self.min_count > self.max_count:
            raise ModelError(
                "count_conflict", f"Operation {self.op}: min_count {self.min_count} is above max_count {self.max_count}"
            )


@dataclass(frozen=True)
class SetInverse:
    """Set, or with None clear, the relation type `label` is the inverse of (4.16)."""

    op: ClassVar[str] = "set_inverse"
    label: str
    inverse_of: str | None

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        if self.inverse_of is not None:
            _identifier(self.inverse_of, "relation type")


@dataclass(frozen=True)
class SetSymmetric:
    """Set whether relation type `label` is symmetric (4.16)."""

    op: ClassVar[str] = "set_symmetric"
    label: str
    symmetric: bool

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        _boolean(self, "symmetric")


@dataclass(frozen=True)
class AddIndividual:
    """Add `individual` (4.17). Refused if its id is taken."""

    op: ClassVar[str] = "add_individual"
    individual: Individual


@dataclass(frozen=True)
class RemoveIndividual:
    """Drop the individual `id`. Relations naming it are left for validation."""

    op: ClassVar[str] = "remove_individual"
    id: str

    def __post_init__(self) -> None:
        _identifier(self.id, "individual id")


@dataclass(frozen=True)
class SetIndividual:
    """Replace the individual with `individual.id`, in place."""

    op: ClassVar[str] = "set_individual"
    individual: Individual


@dataclass(frozen=True)
class OrderIndividuals:
    """Put the individuals in `ids` order; `ids` must name each individual once."""

    op: ClassVar[str] = "order_individuals"
    ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _tuple(self, "ids")


Operation = (
    AddNodeType
    | RemoveNodeType
    | RenameNodeType
    | MergeNodeTypes
    | AddRelationType
    | RemoveRelationType
    | RenameRelationType
    | MergeRelationTypes
    | AddProperty
    | RemoveProperty
    | RenameProperty
    | SetProperty
    | SetDescription
    | SetIri
    | SetParents
    | SetKey
    | SetEndpoints
    | SetProfile
    | OrderNodeTypes
    | OrderRelationTypes
    | OrderProperties
    | SetOntology
    | SetImports
    | SetAbstract
    | SetDisjointWith
    | SetSynonyms
    | SetNames
    | SetRelationRestrictions
    | SetRelationCounts
    | SetInverse
    | SetSymmetric
    | AddIndividual
    | RemoveIndividual
    | SetIndividual
    | OrderIndividuals
)

OPERATIONS: dict[str, type[Operation]] = {
    kind.op: kind
    for kind in (
        AddNodeType,
        RemoveNodeType,
        RenameNodeType,
        MergeNodeTypes,
        AddRelationType,
        RemoveRelationType,
        RenameRelationType,
        MergeRelationTypes,
        AddProperty,
        RemoveProperty,
        RenameProperty,
        SetProperty,
        SetDescription,
        SetIri,
        SetParents,
        SetKey,
        SetEndpoints,
        SetProfile,
        OrderNodeTypes,
        OrderRelationTypes,
        OrderProperties,
        SetOntology,
        SetImports,
        SetAbstract,
        SetDisjointWith,
        SetSynonyms,
        SetNames,
        SetRelationRestrictions,
        SetRelationCounts,
        SetInverse,
        SetSymmetric,
        AddIndividual,
        RemoveIndividual,
        SetIndividual,
        OrderIndividuals,
    )
}

#: Fields holding a model file entry, written and read as one: (to_mapping, from_mapping).
_EMBEDDED: dict[str, tuple[Callable[[Any], Any], Callable[[Any, str], Any]]] = {
    "node_type": (node_type_to_mapping, node_type_from_mapping),
    "relation_type": (relation_type_to_mapping, relation_type_from_mapping),
    "property": (property_to_mapping, property_from_mapping),
    "individual": (individual_to_mapping, individual_from_mapping),
    "ontology": (ontology_to_mapping, ontology_from_mapping),
}


def operation_to_mapping(operation: Operation) -> dict[str, Any]:
    """`operation` as the JSON-safe mapping operation_from_mapping() reads back.

    Every field is written, except an optional one left at None. Imports
    are written as their paths, as in a model file.
    """
    out: dict[str, Any] = {"op": operation.op}
    for spec in fields(operation):
        value = getattr(operation, spec.name)
        if value is None:
            if spec.default is not None:
                out[spec.name] = None
            continue
        if spec.name in _EMBEDDED:
            out[spec.name] = _EMBEDDED[spec.name][0](value)
        elif spec.name == "relation_restrictions":
            out[spec.name] = [restriction_to_mapping(r) for r in value]
        elif spec.name == "imports":
            out[spec.name] = imports_to_mapping(value)
        elif isinstance(value, tuple):
            out[spec.name] = list(value)
        elif isinstance(value, Mapping):
            out[spec.name] = dict(value)
        else:
            out[spec.name] = value
    return out


def operation_from_mapping(raw: Any, source: str = "operation", resolver: ImportResolver | None = None) -> Operation:
    """An operation from its JSON form.

    Args:
        raw: The parsed mapping.
        source: What it was read from, for error messages.
        resolver: Resolves each path of a `set_imports` operation to its
            model, as a model file reader's does (4.12).

    Raises:
        ModelError: invalid_format for a missing key or a value of the wrong
            JSON type, unknown_operation for an ``op`` hygm doesn't define,
            unknown_key for a key the operation doesn't define,
            unknown_model for imports and no resolver, or the code of an
            embedded type, property, restriction, individual, ontology block
            or profile that breaks the format.
    """
    where = f"Operation {source}"
    if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
        raise ModelError("invalid_format", f"{where} must be a mapping with a string 'op'")
    kind = OPERATIONS.get(raw["op"])
    if kind is None:
        raise ModelError("unknown_operation", f"{where}: unknown op {raw['op']!r}; known: {tuple(OPERATIONS)}")
    specs = {spec.name: spec for spec in fields(kind)}
    unknown = sorted(set(raw) - set(specs) - {"op"})
    if unknown:
        raise ModelError("unknown_key", f"{where}: {raw['op']} has unknown key {unknown[0]!r}")
    values: dict[str, Any] = {}
    for name, spec in specs.items():
        if name not in raw:
            if spec.default is not MISSING:
                continue
            raise ModelError("invalid_format", f"{where}: {raw['op']} needs {name!r}")
        values[name] = _field(raw[name], spec, where, resolver)
    try:
        return kind(**values)
    except ModelError as e:
        raise ModelError(e.codes, f"{where}: {e}") from e


def _field(value: Any, spec: Field, where: str, resolver: ImportResolver | None) -> Any:
    """One field's value from its JSON form, checked against the field's annotation."""
    name, annotation = spec.name, spec.type
    nullable = isinstance(annotation, UnionType) and type(None) in get_args(annotation)
    if value is None and nullable:
        return None
    if name in _EMBEDDED:
        return _EMBEDDED[name][1](value, f"{where} {name}")
    if name == "relation_restrictions":
        if not isinstance(value, list):
            raise ModelError("invalid_format", f"{where}: {name} must be a list")
        return tuple(restriction_from_mapping(item, f"{where} {name}[{i}]") for i, item in enumerate(value))
    if name == "imports":
        return imports_from_mapping(value, where, resolver)
    expected = next(a for a in get_args(annotation) if a is not type(None)) if nullable else annotation
    if expected == tuple[str, ...]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ModelError("invalid_format", f"{where}: {name} must be a list of strings")
        return tuple(value)
    if expected is bool:
        if not isinstance(value, bool):
            raise ModelError("invalid_format", f"{where}: {name} must be true or false")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModelError("invalid_format", f"{where}: {name} must be a whole number")
        return value
    if get_origin(expected) is Mapping:
        if not isinstance(value, dict):
            raise ModelError("invalid_format", f"{where}: {name} must be a mapping")
        return value
    if not isinstance(value, str):
        raise ModelError("invalid_format", f"{where}: {name} must be a string")
    return value
