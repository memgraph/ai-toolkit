"""The operations a model changes by, and their JSON form (docs/contract.md, section 11).

Each operation is a frozen dataclass naming what it changes; `apply` gives
it meaning. Its JSON form is a mapping with an ``op`` key and one key per
field, embedded types written as version-2 model entries. Reading is as
strict as reading a model file: an unknown ``op`` or key is refused.
"""

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, ClassVar, Literal

from ..errors import ModelError
from ..format import (
    check_profile,
    node_type_from_mapping,
    node_type_to_mapping,
    property_from_mapping,
    property_to_mapping,
    relation_type_from_mapping,
    relation_type_to_mapping,
)
from ..types import NodeType, Property, RelationType, _identifier

Kind = Literal["node", "relation"]
ProfileLevel = Literal["model", "node", "relation"]

_KINDS: tuple[str, ...] = ("node", "relation")


def _labels(owner: object, attribute: str) -> None:
    value = getattr(owner, attribute)
    if isinstance(value, str):
        raise TypeError(f"{type(owner).__name__}: {attribute} must be a tuple, not a string")
    object.__setattr__(owner, attribute, tuple(value))


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
        _labels(self, "parents")


@dataclass(frozen=True)
class SetKey:
    """Set the key of node type `label`; () clears it."""

    op: ClassVar[str] = "set_key"
    label: str
    key: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.label, "node label")
        _labels(self, "key")


@dataclass(frozen=True)
class SetEndpoints:
    """Set the labels relation type `label` may start and end at; () means any declared label."""

    op: ClassVar[str] = "set_endpoints"
    label: str
    start_labels: tuple[str, ...]
    end_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.label, "relation type")
        _labels(self, "start_labels")
        _labels(self, "end_labels")


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
        _labels(self, "labels")


@dataclass(frozen=True)
class OrderRelationTypes:
    """Put the relation types in `labels` order; `labels` must name each declared type once."""

    op: ClassVar[str] = "order_relation_types"
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _labels(self, "labels")


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
        _labels(self, "names")


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
    )
}

_EMBEDDED = {
    "node_type": (node_type_to_mapping, node_type_from_mapping),
    "relation_type": (relation_type_to_mapping, relation_type_from_mapping),
    "property": (property_to_mapping, property_from_mapping),
}
_LISTS = {"parents", "key", "start_labels", "end_labels", "labels", "names"}


def operation_to_mapping(operation: Operation) -> dict[str, Any]:
    """`operation` as the JSON-safe mapping operation_from_mapping() reads back.

    Every field is written, except an optional one left at None.
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
        elif spec.name in _LISTS:
            out[spec.name] = list(value)
        elif spec.name == "settings":
            out[spec.name] = dict(value)
        else:
            out[spec.name] = value
    return out


def operation_from_mapping(raw: Any, source: str = "operation") -> Operation:
    """An operation from its JSON form.

    Args:
        raw: The parsed mapping.
        source: What it was read from, for error messages.

    Raises:
        ModelError: invalid_format for a missing key or a value of the wrong
            JSON type, unknown_operation for an ``op`` hygm doesn't define,
            unknown_key for a key the operation doesn't define, or the code of
            an embedded type, property or profile that breaks the format.
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
        values[name] = _field(
            raw[name], name, nullable=spec.default is None or name in {"iri", "settings"}, where=where
        )
    try:
        return kind(**values)
    except ModelError as e:
        raise ModelError(e.codes, f"{where}: {e}") from e


def _field(value: Any, name: str, *, nullable: bool, where: str) -> Any:
    if value is None and nullable:
        return None
    if name in _EMBEDDED:
        return _EMBEDDED[name][1](value, f"{where} {name}")
    if name in _LISTS:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ModelError("invalid_format", f"{where}: {name} must be a list of strings")
        return tuple(value)
    if name == "settings":
        if not isinstance(value, dict):
            raise ModelError("invalid_format", f"{where}: settings must be a mapping or null")
        return value
    if not isinstance(value, str):
        raise ModelError("invalid_format", f"{where}: {name} must be a string")
    return value
