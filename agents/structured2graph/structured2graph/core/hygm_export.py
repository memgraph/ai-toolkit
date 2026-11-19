"""sql2graph's GraphModel as a hygm model and mapping (hygm/docs/contract.md, sections 4 and 6).

The first step of moving sql2graph onto hygm: sql2graph keeps its own
GraphModel and output, and also writes the same model in hygm's files, so other
tools in the Memgraph ecosystem can read what it decided.

- Each GraphNode becomes a node type named by its first label, and its
  NodeSource a `table` node binding whose `id` is the source's `id_field`.
  When that column is bound to a property, the property is the type's `key`
  and is `required`.
- Each GraphRelationship becomes a relation type from its start node's type to
  its end node's, and its RelationshipSource a `table` relation binding over
  the table that holds the rows: the foreign key's own table, or the
  many-to-many join table. A foreign key that references a column other than
  the endpoint's `id` becomes a `join` when both columns are bound properties.
- Property `types` become logical types; an enum linked to a property (by a
  `data_type` constraint naming it, or by its source's `migrated_from`)
  becomes its `values`; `existence` constraints become `required`.
- Statistics (`count`, `filling_factor`, `examples`) describe a dataset, not a
  model, and stay in sql2graph.

Whatever hygm can't carry is never dropped silently: it is reported as a Gap,
written beside the files by write_hygm_files().
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hygm import (
    ColumnEndpoint,
    HygmModel,
    Join,
    JoinEndpoint,
    Mapping,
    ModelError,
    NodeBinding,
    NodeType,
    Property,
    PropertyBinding,
    PropertyType,
    RelationBinding,
    RelationType,
    Source,
    TableRef,
    check_mapping,
    is_valid_identifier,
    mapping_to_mapping,
    model_to_mapping,
    validate_model,
)

logger = logging.getLogger(__name__)

#: The source a mapping names when the caller doesn't: the one relational database sql2graph reads.
DEFAULT_SOURCE = "sql"

#: sql2graph property type names (Memgraph's schema-info names and data_type
#: constraint types), lowercased, to hygm's logical types.
_LOGICAL_TYPES: dict[str, PropertyType] = {
    "string": "string",
    "str": "string",
    "integer": "integer",
    "int": "integer",
    "float": "float",
    "double": "float",
    "decimal": "decimal",
    "boolean": "boolean",
    "bool": "boolean",
    "date": "date",
    "localdatetime": "datetime",
    "local date time": "datetime",
    "local_date_time": "datetime",
    "datetime": "datetime",
    "zoneddatetime": "datetime",
    "zoned date time": "datetime",
    "zoned_date_time": "datetime",
    "map": "json",
    "json": "json",
}

#: A type entry that only says some values were missing: the property is nullable, not of another type.
_NULL_TYPE = "null"

#: Relationship source types whose rows are read from a table.
_TABLE_RELATION_SOURCES = {"table", "view", "many_to_many"}


@dataclass(frozen=True)
class Gap:
    """Something in sql2graph's GraphModel that the hygm files don't carry.

    Attributes:
        element: What it belongs to, e.g. "node Customer" or "relationship PLACED".
        kind: A short, stable name for the kind of gap, to group them by.
        detail: What was not carried and why.
    """

    element: str
    kind: str
    detail: str


class HygmExport(NamedTuple):
    """A GraphModel in hygm's terms: the model, its mapping, and what neither could carry."""

    model: HygmModel
    mapping: Mapping
    gaps: tuple[Gap, ...]


class HygmFiles(NamedTuple):
    """The files write_hygm_files() wrote."""

    model: Path
    mapping: Path
    gaps: Path


def graph_model_to_hygm(
    graph_model: Any, *, source: str = DEFAULT_SOURCE, model_path: str = "model.yaml"
) -> HygmExport:
    """Convert sql2graph's GraphModel into a hygm model and a mapping onto one `table` source.

    Args:
        graph_model: sql2graph's GraphModel (core.hygm.models.graph_models).
        source: The name the mapping gives the source database.
        model_path: The model file's path as the mapping refers to it,
            relative to the mapping file.

    Returns:
        The model, the mapping, and the gaps: everything in `graph_model`
        that hygm can't carry.

    Raises:
        ModelError: if the result fails validate_model() or check_mapping(),
            which would be a bug in this conversion.
    """
    return _Converter(graph_model, source, model_path).convert()


def write_hygm_files(graph_model: Any, beside: str | Path, *, source: str = DEFAULT_SOURCE) -> HygmFiles:
    """Write `graph_model` as hygm files next to `beside`, sql2graph's own output file.

    For `out/shop.json` it writes `out/shop.model.yaml`, `out/shop.mapping.yaml`
    (whose `model` names the model file) and `out/shop.hygm-gaps.yaml`, which
    lists what the hygm files don't carry. Each gap is also logged at INFO.

    Returns:
        The paths written.

    Raises:
        ModelError: as graph_model_to_hygm() does.
        OSError: if a file can't be written.
    """
    beside = Path(beside)
    stem = beside.stem
    paths = HygmFiles(
        model=beside.with_name(f"{stem}.model.yaml"),
        mapping=beside.with_name(f"{stem}.mapping.yaml"),
        gaps=beside.with_name(f"{stem}.hygm-gaps.yaml"),
    )
    export = graph_model_to_hygm(graph_model, source=source, model_path=paths.model.name)
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    _dump(model_to_mapping(export.model), paths.model)
    _dump(mapping_to_mapping(export.mapping), paths.mapping)
    _dump({"gaps": [_gap_entry(gap) for gap in export.gaps]}, paths.gaps)
    for gap in export.gaps:
        logger.info("hygm export: %s: %s (%s)", gap.element, gap.detail, gap.kind)
    return paths


def _dump(content: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _gap_entry(gap: Gap) -> dict[str, str]:
    return {"element": gap.element, "kind": gap.kind, "detail": gap.detail}


def _split_ref(ref: str) -> tuple[str, str]:
    """Split a `table.column` reference; a bare name is a column."""
    if "." in ref:
        table, column = ref.split(".", 1)
        return table, column
    return "", ref


@dataclass
class _NodeDraft:
    """One node type as the conversion gathers it, possibly from several GraphNodes."""

    label: str
    properties: dict[str, Property] = field(default_factory=dict)
    bindings: list[NodeBinding] = field(default_factory=list)
    columns: dict[str, str] = field(default_factory=dict)  # column -> property, over every table binding
    keys: list[tuple[str, ...] | None] = field(default_factory=list)  # one per table binding


@dataclass
class _RelationDraft:
    label: str
    pairs: list[tuple[str, str]]  # the (start, end) types sql2graph connects with it
    properties: dict[str, Property] = field(default_factory=dict)
    bindings: list[RelationBinding] = field(default_factory=list)


class _Converter:
    def __init__(self, graph_model: Any, source: str, model_path: str) -> None:
        self.graph_model = graph_model
        self.source = source
        self.model_path = model_path
        self.gaps: list[Gap] = []
        self.nodes: dict[str, _NodeDraft] = {}
        self.relations: dict[str, _RelationDraft] = {}
        self.required: dict[str, set[str]] = {}
        self.relation_required: dict[str, set[str]] = {}
        self.values: dict[tuple[str, str], tuple[str, ...]] = {}
        self.typed: dict[tuple[str, str], PropertyType] = {}

    def gap(self, element: str, kind: str, detail: str) -> None:
        self.gaps.append(Gap(element, kind, detail))

    def convert(self) -> HygmExport:
        for node in self.graph_model.nodes:
            self.node(node)
        self.node_constraints()
        self.enums()
        self.node_indexes()
        for edge in self.graph_model.edges:
            self.edge(edge)
        self.edge_constraints()

        model = HygmModel(
            node_types=tuple(self.node_type(draft) for draft in self.nodes.values()),
            relation_types=tuple(self.relation_type(draft) for draft in self.relations.values()),
        )
        mapping = Mapping(
            model=self.model_path,
            sources=(Source(self.source, "table"),),
            nodes=tuple(b for draft in self.nodes.values() for b in draft.bindings),
            relations=tuple(b for draft in self.relations.values() for b in draft.bindings),
        )
        result = validate_model(model)
        if not result.success:
            raise ModelError(
                [issue.code or "invalid_format" for issue in result.critical_issues],
                "sql2graph's model converts to an invalid hygm model: "
                + "; ".join(issue.message for issue in result.critical_issues),
            )
        check_mapping(mapping, model)
        return HygmExport(model, mapping, tuple(self.gaps))

    # Nodes

    def node(self, node: Any) -> None:
        if not node.labels:
            self.gap("node (no label)", "unlabeled_node", "a node with no label can't be a node type; dropped")
            return
        label = node.labels[0]
        element = f"node {label}"
        if not is_valid_identifier(label):
            self.gap(element, "invalid_identifier", f"label {label!r} isn't a valid identifier; node dropped")
            return
        if len(node.labels) > 1:
            self.gap(
                element,
                "extra_labels",
                f"labels {node.labels[1:]} aren't carried: a hygm node carries its type and its ancestors' labels",
            )
        draft = self.nodes.setdefault(label, _NodeDraft(label))
        for prop in node.properties:
            self.declare(draft.properties, prop, f"{element} property {prop.key}")

        source = node.source
        if source is None:
            self.gap(element, "no_source", "no NodeSource, so the type has no binding")
            return
        if source.type not in ("table", "view"):
            self.gap(element, "source_type", f"NodeSource type {source.type!r} has no hygm source kind; not bound")
            return
        if source.type == "view":
            self.gap(element, "view_source", f"view {source.name!r} is bound as a table; that it is a view is lost")
        id_column = _split_ref(str(source.mapping.get("id_field") or ""))[1]
        if not id_column:
            self.gap(element, "no_id", f"table {source.name!r} has no id_field, so it can't be bound")
            return

        bound: list[PropertyBinding] = []
        for prop in node.properties:
            column = self.column_of(prop, source.name, f"{element} property {prop.key}")
            if column is not None and prop.key in draft.properties:
                bound.append(PropertyBinding(prop.key, column))
                draft.columns.setdefault(column, prop.key)
        key_property = next((b.name for b in bound if b.column == id_column), None)
        draft.keys.append((key_property,) if key_property else None)
        draft.bindings.append(
            NodeBinding(
                type=label,
                source=self.source,
                table=TableRef(source.name),
                id=(id_column,),
                properties=tuple(bound),
            )
        )

    def column_of(self, prop: Any, table: str, element: str) -> str | None:
        """The column of `table` that `prop` reads, or None (with a gap) when it reads no column of it."""
        if prop.source is None or not prop.source.field:
            return prop.key
        if prop.source.transformation:
            self.gap(
                element,
                "transformation",
                f"transformation {prop.source.transformation!r} has no place in a hygm binding; not bound",
            )
            return None
        prefix, column = _split_ref(prop.source.field)
        if prefix == "llm":
            self.gap(
                element, "unsourced_property", "declared, but sourced from the LLM rather than a column; not bound"
            )
            return None
        if prefix and prefix != table:
            self.gap(
                element,
                "foreign_column",
                f"reads {prop.source.field!r}, a column of another table than {table!r}; not bound",
            )
            return None
        return column

    def declare(self, properties: dict[str, Property], prop: Any, element: str) -> None:
        if not is_valid_identifier(prop.key):
            self.gap(element, "invalid_identifier", f"property name {prop.key!r} isn't a valid identifier; dropped")
            return
        declared = Property(prop.key, self.logical_type(prop.types, element))
        existing = properties.get(prop.key)
        if existing is None:
            properties[prop.key] = declared
        elif existing.type != declared.type:
            self.gap(
                element,
                "type_conflict",
                f"declared as {existing.type} and as {declared.type} by two sql2graph elements; kept {existing.type}",
            )

    def logical_type(self, types: list[dict[str, Any]], element: str) -> PropertyType:
        names = {str(t.get("type", "")).strip().lower() for t in types or ()} - {_NULL_TYPE, ""}
        unknown = sorted(name for name in names if name not in _LOGICAL_TYPES)
        if unknown:
            self.gap(element, "property_type", f"type(s) {unknown} have no hygm logical type; declared as string")
            return "string"
        logical = {_LOGICAL_TYPES[name] for name in names}
        if len(logical) > 1:
            self.gap(element, "mixed_types", f"holds values of several types {sorted(names)}; declared as string")
            return "string"
        return logical.pop() if logical else "string"

    def node_type(self, draft: _NodeDraft) -> NodeType:
        keys = set(draft.keys)
        key: tuple[str, ...] = ()
        if len(keys) == 1 and None not in keys:
            key = next(iter(keys)) or ()
        elif len(draft.keys) > 1 and keys != {None}:
            self.gap(
                f"node {draft.label}",
                "no_common_key",
                "bound from several tables whose ids are different properties; no key declared",
            )
        required = self.required.get(draft.label, set()) | set(key)
        properties = []
        for prop in draft.properties.values():
            where = (draft.label, prop.name)
            if where in self.values and self.typed.get(where, prop.type) != "string":
                self.gap(
                    f"node {draft.label} property {prop.name}",
                    "enum_type",
                    f"enum values on a {self.typed.get(where, prop.type)} property; declared as string to carry them",
                )
            properties.append(
                Property(
                    prop.name,
                    "string" if where in self.values else self.typed.get(where, prop.type),
                    required=prop.name in required,
                    values=self.values.get(where),
                )
            )
        return NodeType(label=draft.label, properties=tuple(properties), key=key)

    def key_of(self, label: str) -> tuple[str, ...]:
        draft = self.nodes[label]
        keys = set(draft.keys)
        return (next(iter(keys)) or ()) if len(keys) == 1 else ()

    def node_constraints(self) -> None:
        for constraint in self.graph_model.node_constraints:
            label = (constraint.labels or [""])[0]
            element = f"constraint {constraint.type} on {label}({', '.join(constraint.properties)})"
            draft = self.nodes.get(label)
            if draft is None or any(p not in draft.properties for p in constraint.properties):
                self.gap(element, "unknown_target", "names a type or property the model doesn't declare; dropped")
                continue
            if constraint.type == "existence":
                self.required.setdefault(label, set()).update(constraint.properties)
            elif constraint.type == "unique":
                if tuple(constraint.properties) != self.key_of(label):
                    self.gap(
                        element,
                        "unique_not_key",
                        "a type has one key, from its table's id; another uniqueness constraint isn't carried",
                    )
            elif constraint.type == "data_type":
                self.data_type(label, constraint, element)
            else:
                self.gap(element, "constraint_type", f"constraint type {constraint.type!r} isn't carried")

    def data_type(self, label: str, constraint: Any, element: str) -> None:
        data_type = (constraint.data_type or "").strip()
        enum = self.enum_named(data_type)
        for prop in constraint.properties:
            if enum is not None:
                self.values[(label, prop)] = tuple(str(v) for v in enum.values)
            elif data_type.lower() in _LOGICAL_TYPES:
                self.typed[(label, prop)] = _LOGICAL_TYPES[data_type.lower()]
            else:
                self.gap(element, "property_type", f"data type {data_type!r} has no hygm logical type")

    def enum_named(self, data_type: str) -> Any:
        name = data_type.split("::", 1)[1] if data_type.lower().startswith("enum::") else data_type
        return next((e for e in self.graph_model.enums if e.name == name), None)

    def enums(self) -> None:
        """Link each enum to the properties it constrains; one linked to none is a gap."""
        linked = {id(e) for e in (self.enum_named(c.data_type or "") for c in self.graph_model.node_constraints) if e}
        for enum in self.graph_model.enums:
            origin = (enum.source.migrated_from or "") if enum.source else ""
            targets = [
                (node.labels[0], prop.key)
                for node in self.graph_model.nodes
                if node.labels and node.labels[0] in self.nodes
                for prop in node.properties
                if origin and prop.source and prop.source.field and f".{origin}".endswith(f".{prop.source.field}")
            ]
            for target in targets:
                self.values[target] = tuple(str(v) for v in enum.values)
            if not targets and id(enum) not in linked:
                self.gap(
                    f"enum {enum.name}",
                    "unlinked_enum",
                    "no data_type constraint or migrated_from links it to a property; its values aren't carried",
                )

    def node_indexes(self) -> None:
        for index in self.graph_model.node_indexes:
            label = (index.labels or [""])[0]
            if label in self.nodes and index.properties and tuple(index.properties) == self.key_of(label):
                continue
            detail = "a hygm model has no indexes; a plan derives them from keys, so only a key's index is carried"
            undeclared = [
                p for p in index.properties if label not in self.nodes or p not in self.nodes[label].properties
            ]
            if undeclared:
                detail += f" (and {undeclared} aren't declared properties of {label!r})"
            self.gap(f"index on {label}({', '.join(index.properties)})", "index", detail)

    # Relations

    def edge(self, edge: Any) -> None:
        label = edge.edge_type
        element = f"relationship {label}"
        if not is_valid_identifier(label):
            self.gap(element, "invalid_identifier", f"type {label!r} isn't a valid identifier; dropped")
            return
        if edge.directionality != "directed":
            self.gap(element, "directionality", f"{edge.directionality!r}: hygm relation types are directed")
        for side, labels in (("start", edge.start_node_labels), ("end", edge.end_node_labels)):
            if len(labels) > 1:
                self.gap(element, "endpoint_labels", f"{side} labels {labels[1:]} beyond the first aren't carried")
        start = (edge.start_node_labels or [""])[0]
        end = (edge.end_node_labels or [""])[0]
        if start not in self.nodes or end not in self.nodes:
            missing = [label for label in (start, end) if label not in self.nodes]
            self.gap(element, "undeclared_endpoint", f"endpoint label(s) {missing} aren't node types; dropped")
            return

        draft = self.relations.get(label)
        if draft is None:
            draft = self.relations[label] = _RelationDraft(label, [(start, end)])
        elif (start, end) not in draft.pairs:
            draft.pairs.append((start, end))
        for prop in edge.properties:
            self.declare(draft.properties, prop, f"{element} property {prop.key}")
        binding = self.relation_binding(edge, start, end, element)
        if binding is None:
            return
        draft.bindings.append(binding)
        if binding.join is not None and edge.properties:
            self.gap(
                element,
                "join_properties",
                f"properties {[p.key for p in edge.properties]} are declared, but a join reads no rows to bind them from",
            )

    def relation_binding(self, edge: Any, start: str, end: str, element: str) -> RelationBinding | None:
        source = edge.source
        if source is None:
            self.gap(element, "no_source", "no RelationshipSource, so the type has no binding")
            return None
        mapping = source.mapping or {}
        start_table, start_col = _split_ref(str(mapping.get("start_node", "")))
        end_table, end_col = _split_ref(str(mapping.get("end_node", "")))
        if source.type == "derived":
            return self.join(edge.edge_type, start, start_table, start_col, end, end_table, end_col, element)
        if source.type not in _TABLE_RELATION_SOURCES:
            self.gap(element, "source_type", f"RelationshipSource type {source.type!r} has no hygm binding")
            return None
        if source.type == "view":
            self.gap(element, "view_source", "a view is bound as a table; that it is a view is lost")

        from_table, from_id = self.node_table(start)
        to_table, to_id = self.node_table(end)
        unbound = [label for label, table in ((start, from_table), (end, to_table)) if not table]
        if unbound:
            self.gap(
                element, "unbound_endpoint", f"endpoint(s) {unbound} have no table binding to reference; not bound"
            )
            return None
        join_table = mapping.get("join_table") or (source.name if source.type == "many_to_many" else "")
        if join_table:
            from_col = mapping.get("join_from_column") or start_col
            to_col = mapping.get("join_to_column") or end_col
            for referenced, id_column, endpoint in (
                (mapping.get("from_column"), from_id, start),
                (mapping.get("to_column"), to_id, end),
            ):
                if referenced and referenced != id_column:
                    self.gap(
                        element,
                        "fk_target_not_id",
                        f"join table {join_table!r} references {endpoint}'s column {referenced!r}, not its id "
                        f"{id_column!r}; not bound",
                    )
                    return None
            return self.table_binding(edge, join_table, (start, from_col), (end, to_col), element)
        if start_table == from_table and start_col != from_id:
            # The foreign key is on the start node's table and points at the end node.
            if end_col != to_id:
                return self.join(
                    edge.edge_type, start, start_table, start_col, end, end_table, end_col, element, "fk_target_not_id"
                )
            return self.table_binding(edge, from_table, (start, from_id), (end, start_col), element)
        if end_table == to_table and end_col != to_id:
            # The foreign key is on the end node's table and points back at the start node.
            if start_col != from_id:
                return self.join(
                    edge.edge_type, start, start_table, start_col, end, end_table, end_col, element, "fk_target_not_id"
                )
            return self.table_binding(edge, to_table, (start, end_col), (end, to_id), element)
        self.gap(
            element,
            "no_backing_rows",
            f"{mapping.get('start_node')!r} -> {mapping.get('end_node')!r} is neither a foreign key "
            "nor a join table; not bound",
        )
        return None

    def node_table(self, label: str) -> tuple[str, str]:
        """The table and id column of `label`'s first table binding, or ("", "")."""
        binding = next((b for b in self.nodes[label].bindings if b.table is not None), None)
        return (binding.table.name, binding.id[0]) if binding and binding.table else ("", "")

    def table_binding(
        self, edge: Any, table: str, start: tuple[str, str], end: tuple[str, str], element: str
    ) -> RelationBinding:
        draft = self.relations[edge.edge_type]
        bound = []
        for prop in edge.properties:
            column = self.column_of(prop, table, f"{element} property {prop.key}")
            if column is not None and prop.key in draft.properties:
                bound.append(PropertyBinding(prop.key, column))
        return RelationBinding(
            type=edge.edge_type,
            source=self.source,
            table=TableRef(table),
            from_=ColumnEndpoint(start[0], (start[1],)),
            to=ColumnEndpoint(end[0], (end[1],)),
            properties=tuple(bound),
        )

    def join(
        self,
        label: str,
        start: str,
        start_table: str,
        start_col: str,
        end: str,
        end_table: str,
        end_col: str,
        element: str,
        kind: str = "derived_source",
    ) -> RelationBinding | None:
        """A join on equal properties when both columns are bound on their endpoints, else None with a gap."""
        from_prop = self.property_of(start, start_table, start_col)
        to_prop = self.property_of(end, end_table, end_col)
        if from_prop is None or to_prop is None:
            self.gap(
                element,
                kind,
                f"{start_table}.{start_col} -> {end_table}.{end_col} isn't a foreign key onto the endpoint's id, "
                "and isn't equality on bound properties a join could express; not bound",
            )
            return None
        return RelationBinding(type=label, join=Join(JoinEndpoint(start, from_prop), JoinEndpoint(end, to_prop)))

    def property_of(self, label: str, table: str, column: str) -> str | None:
        draft = self.nodes[label]
        tables = {b.table.name for b in draft.bindings if b.table is not None}
        if table and table not in tables:
            return None
        return draft.columns.get(column)

    def edge_constraints(self) -> None:
        for constraint in self.graph_model.edge_constraints:
            label = constraint.edge_type or ""
            element = f"constraint {constraint.type} on [{label}]({', '.join(constraint.properties)})"
            draft = self.relations.get(label)
            if draft is None or any(p not in draft.properties for p in constraint.properties):
                self.gap(element, "unknown_target", "names a type or property the model doesn't declare; dropped")
            elif constraint.type == "existence":
                self.relation_required.setdefault(label, set()).update(constraint.properties)
            else:
                self.gap(element, "relation_constraint", f"hygm relation types have no {constraint.type} constraints")
        for index in self.graph_model.edge_indexes:
            self.gap(
                f"index on [{index.edge_type}]({', '.join(index.properties)})",
                "index",
                "a hygm model has no indexes, and relation types have no keys to derive one from",
            )

    def relation_type(self, draft: _RelationDraft) -> RelationType:
        starts = tuple(dict.fromkeys(start for start, _ in draft.pairs))
        ends = tuple(dict.fromkeys(end for _, end in draft.pairs))
        if len(starts) * len(ends) > len(draft.pairs):
            self.gap(
                f"relationship {draft.label}",
                "endpoint_pairs",
                f"connects only {draft.pairs}, but start_labels {list(starts)} and end_labels {list(ends)} "
                "admit every start with every end (each binding still names its own pair)",
            )
        return RelationType(
            label=draft.label,
            start_labels=starts,
            end_labels=ends,
            properties=tuple(self.relation_property(draft, p) for p in draft.properties.values()),
        )

    def relation_property(self, draft: _RelationDraft, prop: Property) -> Property:
        return Property(prop.name, prop.type, required=prop.name in self.relation_required.get(draft.label, set()))
