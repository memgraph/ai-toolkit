"""The mapping file's parts: sources and the bindings of model types to them (docs/contract.md, 6.1).

A mapping only names things. Types and properties name the model's, which
check_mapping() confirms; tables, columns, JSON paths and a graph source's own
labels name the source's, which only a consumer that reads the source can
confirm. No part holds connection details.
"""

from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["table", "graph"]

#: The source kinds a mapping may declare. `document` is reserved (contract, 12.3) and refused until specified.
SOURCE_KINDS: tuple[SourceKind, ...] = ("table", "graph")

BindingKind = Literal["table", "graph", "join"]


@dataclass(frozen=True)
class Source:
    """A named source the bindings read from.

    Attributes:
        name: What bindings call it; unique within the mapping. How to reach
            it is each consumer's own configuration.
        kind: `table` (relational rows) or `graph` (a native property graph).
    """

    name: str
    kind: SourceKind


@dataclass(frozen=True)
class TableRef:
    """A table in a `table` source.

    Attributes:
        name: The table's name, as the source spells it.
        schema: The schema (namespace) holding it; None for the source's default.
    """

    name: str
    schema: str | None = None


@dataclass(frozen=True)
class GraphRef:
    """Where a `graph` source keeps a type's nodes or relationships.

    Attributes:
        label: The source's own label (or relationship type) for them, when it
            differs from the model's; None means the model's. It names what to
            read, never a label to write: a bound node's labels are its type
            and that type's ancestors (contract, 4.7).
    """

    label: str | None = None


@dataclass(frozen=True)
class PropertyBinding:
    """A model property read from a source column.

    Attributes:
        name: The model property, declared or inherited by the bound type.
        column: The column holding the value.
        path: A JSON path (`$.points`) into the column's value, for a column
            holding JSON. The column's type is the source's, so a path may
            feed a property of any model type.
    """

    name: str
    column: str
    path: str | None = None


@dataclass(frozen=True)
class NodeBinding:
    """One model node type bound to one source; exactly one of `table` and `graph` is set.

    Attributes:
        type: The model node type the bound nodes are instances of.
        source: The source's name.
        table: The table whose rows are the nodes, in a `table` source.
        graph: The source's nodes of a label, in a `graph` source.
        id: The columns identifying a row of `table`; empty for `graph`.
        properties: The bound properties; only on a `table` binding.
    """

    type: str
    source: str
    table: TableRef | None = None
    graph: GraphRef | None = None
    id: tuple[str, ...] = ()
    properties: tuple[PropertyBinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", tuple(self.id))
        object.__setattr__(self, "properties", tuple(self.properties))

    @property
    def kind(self) -> Literal["table", "graph"]:
        """Which binding this is."""
        return "table" if self.table is not None else "graph"


@dataclass(frozen=True)
class ColumnEndpoint:
    """One end of a `table` relation binding: columns holding a reference to an endpoint node.

    Attributes:
        type: The endpoint's model node type, admitted by the relation type.
        columns: The relation table's columns that match, in order, the `id`
            of the endpoint type's table binding in the same source, or its
            `key` when it has none there (check_mapping()).
    """

    type: str
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))


@dataclass(frozen=True)
class JoinEndpoint:
    """One end of a `join`: a model property on an endpoint type.

    Attributes:
        type: The endpoint's model node type, admitted by the relation type.
        property: A property the type declares or inherits.
    """

    type: str
    property: str


@dataclass(frozen=True)
class Join:
    """A relationship with no backing rows: one exists wherever the two endpoint properties are equal.

    Attributes:
        from_: The start node's side (`from` in the file).
        to: The end node's side.
    """

    from_: JoinEndpoint
    to: JoinEndpoint


@dataclass(frozen=True)
class RelationBinding:
    """One model relation type bound to its data; exactly one of `table`, `graph` and `join` is set.

    Attributes:
        type: The model relation type.
        source: The source's name; None for a `join`, which reads no source.
        table: The table whose rows are the relationships, in a `table` source.
        from_: The start node's columns in `table` (`from` in the file).
        to: The end node's columns in `table`.
        graph: The source's relationships of a type, in a `graph` source.
        join: Equal properties on the two endpoints.
        properties: The bound relation properties; only on a `table` binding.
    """

    type: str
    source: str | None = None
    table: TableRef | None = None
    from_: ColumnEndpoint | None = None
    to: ColumnEndpoint | None = None
    graph: GraphRef | None = None
    join: Join | None = None
    properties: tuple[PropertyBinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "properties", tuple(self.properties))

    @property
    def kind(self) -> BindingKind:
        """Which binding this is."""
        if self.table is not None:
            return "table"
        return "graph" if self.graph is not None else "join"


@dataclass(frozen=True)
class Mapping:
    """A mapping file: where a model's types come from (docs/contract.md, section 6).

    Attributes:
        model: The model file's path as written, relative to the mapping file.
        sources: The sources the bindings name.
        nodes: The node bindings; a type may have several.
        relations: The relation bindings; a type may have several.
    """

    model: str
    sources: tuple[Source, ...] = ()
    nodes: tuple[NodeBinding, ...] = ()
    relations: tuple[RelationBinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "relations", tuple(self.relations))

    def source(self, name: str) -> Source | None:
        """The source called `name`, or None."""
        return next((s for s in self.sources if s.name == name), None)
