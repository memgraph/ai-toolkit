"""The mapping file: reading, writing and loading it with its model (docs/contract.md, section 6).

A reader refuses any key format version 1 doesn't define, at every level, so a
typo is an error rather than silently dropped meaning. Reading checks the
shape; check_mapping() then checks the mapping against its model.
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from ..errors import ModelError
from ..strategies.manual import ManualStrategy
from ..types import HygmModel
from .types import (
    SOURCE_KINDS,
    ColumnEndpoint,
    GraphRef,
    Join,
    JoinEndpoint,
    Mapping,
    NodeBinding,
    PropertyBinding,
    RelationBinding,
    Source,
    TableRef,
)
from .validation import check_mapping

VERSIONS = (1,)

_TOP = {"version", "model", "sources", "nodes", "relations"}
_SOURCE = {"name", "kind"}
_NODE = {"type", "source", "table", "graph", "id", "properties"}
_RELATION = {"type", "source", "table", "from", "to", "graph", "join", "properties"}
_TABLE = {"schema", "name"}
_GRAPH = {"label"}
_PROPERTY = {"name", "column", "path"}
_COLUMN_ENDPOINT = {"type", "columns"}
_JOIN = {"from", "to"}
_JOIN_ENDPOINT = {"type", "property"}


def mapping_from_mapping(raw: Any, source: str, model: HygmModel) -> Mapping:
    """A mapping from the parsed contents of a mapping file, checked against `model`.

    Args:
        raw: The parsed file.
        source: What it was read from, for error messages ("file shop.mapping.yaml").
        model: The model its `model` key names, already loaded.

    Raises:
        ModelError: (a ValueError) if the file breaks the format or the
            mapping breaks check_mapping()'s rules against `model`.
    """
    reader = _Reader(f"Mapping {source}")
    reader.header(raw)
    mapping = reader.mapping(raw)
    try:
        check_mapping(mapping, model)
    except ModelError as e:
        raise ModelError(e.codes, f"Mapping {source} is not a valid mapping of its model: {e}") from e
    return mapping


def mapping_to_mapping(mapping: Mapping) -> dict[str, Any]:
    """`mapping` as the contents mapping_from_mapping() reads back: JSON- and YAML-safe, every default left out."""
    out: dict[str, Any] = {"version": 1, "model": mapping.model}
    if mapping.sources:
        out["sources"] = [{"name": s.name, "kind": s.kind} for s in mapping.sources]
    if mapping.nodes:
        out["nodes"] = [_node(b) for b in mapping.nodes]
    if mapping.relations:
        out["relations"] = [_relation(b) for b in mapping.relations]
    return out


def mapping_schema(version: int = 1) -> dict[str, Any]:
    """The JSON Schema for mapping format `version`, as shipped in the package.

    It is the structural floor; docs/contract.md defines the rules a schema
    can't express, which check_mapping() also checks.

    Raises:
        ModelError: for an unknown version.
    """
    if version not in VERSIONS:
        raise ModelError("unknown_version", f"Unknown mapping format version {version}; known: {VERSIONS}")
    return json.loads((files("hygm") / "schema" / f"mapping.v{version}.schema.json").read_text(encoding="utf-8"))


def load_mapping(path: str | Path) -> tuple[HygmModel, Mapping]:
    """Read the mapping file at `path` and the model it names, and check one against the other.

    `model` is resolved relative to the mapping file and loaded with
    ManualStrategy.

    Returns:
        The model and the mapping.

    Raises:
        ModelError: (a ValueError) `invalid_format` if the file can't be read
            or parsed, `unknown_model` if its model can't be, or any code
            mapping_from_mapping() raises.
    """
    resolved = Path(path)
    try:
        raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except OSError as e:
        raise ModelError("invalid_format", f"Could not read mapping file {resolved}: {e}") from e
    except yaml.YAMLError as e:
        raise ModelError("invalid_format", f"Invalid YAML in mapping file {resolved}: {e}") from e

    where_from = f"file {resolved}"
    model_path = resolved.parent / _Reader(f"Mapping {where_from}").header(raw)
    try:
        model = ManualStrategy().create_model(model_path)
    except ModelError as e:
        raise ModelError("unknown_model", f"Mapping {where_from}: its model can't be read: {e}") from e
    return model, mapping_from_mapping(raw, where_from, model)


def _node(binding: NodeBinding) -> dict[str, Any]:
    out: dict[str, Any] = {"type": binding.type, "source": binding.source}
    if binding.table is not None:
        out["table"] = _table(binding.table)
    if binding.graph is not None:
        out["graph"] = _graph(binding.graph)
    if binding.id:
        out["id"] = list(binding.id)
    if binding.properties:
        out["properties"] = [_property(p) for p in binding.properties]
    return out


def _relation(binding: RelationBinding) -> dict[str, Any]:
    out: dict[str, Any] = {"type": binding.type}
    if binding.source is not None:
        out["source"] = binding.source
    if binding.table is not None:
        out["table"] = _table(binding.table)
    if binding.from_ is not None:
        out["from"] = {"type": binding.from_.type, "columns": list(binding.from_.columns)}
    if binding.to is not None:
        out["to"] = {"type": binding.to.type, "columns": list(binding.to.columns)}
    if binding.graph is not None:
        out["graph"] = _graph(binding.graph)
    if binding.join is not None:
        out["join"] = {
            "from": {"type": binding.join.from_.type, "property": binding.join.from_.property},
            "to": {"type": binding.join.to.type, "property": binding.join.to.property},
        }
    if binding.properties:
        out["properties"] = [_property(p) for p in binding.properties]
    return out


def _table(table: TableRef) -> dict[str, Any]:
    return {"schema": table.schema, "name": table.name} if table.schema is not None else {"name": table.name}


def _graph(graph: GraphRef) -> dict[str, Any]:
    return {"label": graph.label} if graph.label is not None else {}


def _property(binding: PropertyBinding) -> dict[str, Any]:
    out = {"name": binding.name, "column": binding.column}
    if binding.path is not None:
        out["path"] = binding.path
    return out


class _Reader:
    """Turns one parsed mapping file into a Mapping, naming where each problem is."""

    def __init__(self, where_from: str) -> None:
        self.where_from = where_from

    def fail(self, code: str, where: str, problem: str) -> ModelError:
        return ModelError(code, f"{self.where_from}: {where} {problem}" if where else f"{self.where_from} {problem}")

    def header(self, raw: Any) -> str:
        """Check what must be right before the model is read: a mapping, its version and its `model` path."""
        if not isinstance(raw, dict):
            raise self.fail("invalid_format", "", "must be a mapping with 'version' and 'model'")
        if "version" not in raw:
            raise self.fail("invalid_format", "", "needs 'version: 1'")
        version = raw["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise self.fail("invalid_format", "", "version must be an integer")
        if version not in VERSIONS:
            raise self.fail("unknown_version", "", f"has unknown format version {version}; known: {VERSIONS}")
        self.known(raw, _TOP, "")
        return self.string(raw, "model", "")

    def mapping(self, raw: dict) -> Mapping:
        return Mapping(
            model=self.string(raw, "model", ""),
            sources=tuple(self.source(item, f"sources[{i}]") for i, item in enumerate(self.listed(raw, "sources"))),
            nodes=tuple(self.node(item, f"nodes[{i}]") for i, item in enumerate(self.listed(raw, "nodes"))),
            relations=tuple(
                self.relation(item, f"relations[{i}]") for i, item in enumerate(self.listed(raw, "relations"))
            ),
        )

    def source(self, item: Any, where: str) -> Source:
        self.object(item, _SOURCE, where)
        name = self.string(item, "name", where)
        kind = item.get("kind")
        if kind not in SOURCE_KINDS:
            reserved = " (document is reserved)" if kind == "document" else ""
            raise self.fail("invalid_format", where, f"kind must be one of {SOURCE_KINDS}{reserved}, got {kind!r}")
        return Source(name=name, kind=kind)

    def node(self, item: Any, where: str) -> NodeBinding:
        self.object(item, _NODE, where)
        return NodeBinding(
            type=self.string(item, "type", where),
            source=self.string(item, "source", where),
            table=self.table(item, where),
            graph=self.graph(item, where),
            id=self.strings(item, "id", where),
            properties=self.properties(item, where),
        )

    def relation(self, item: Any, where: str) -> RelationBinding:
        self.object(item, _RELATION, where)
        join = item.get("join")
        if join is not None:
            self.object(join, _JOIN, f"{where} join")
            join = Join(
                from_=self.join_endpoint(join.get("from"), f"{where} join.from"),
                to=self.join_endpoint(join.get("to"), f"{where} join.to"),
            )
        return RelationBinding(
            type=self.string(item, "type", where),
            source=self.string(item, "source", where) if "source" in item else None,
            table=self.table(item, where),
            from_=self.column_endpoint(item.get("from"), f"{where} from"),
            to=self.column_endpoint(item.get("to"), f"{where} to"),
            graph=self.graph(item, where),
            join=join,
            properties=self.properties(item, where),
        )

    def table(self, item: dict, where: str) -> TableRef | None:
        table = item.get("table")
        if table is None:
            return None
        at = f"{where} table"
        self.object(table, _TABLE, at)
        return TableRef(
            name=self.string(table, "name", at),
            schema=self.string(table, "schema", at) if "schema" in table else None,
        )

    def graph(self, item: dict, where: str) -> GraphRef | None:
        graph = item.get("graph")
        if graph is None:
            return None
        at = f"{where} graph"
        self.object(graph, _GRAPH, at)
        return GraphRef(label=self.string(graph, "label", at) if "label" in graph else None)

    def properties(self, item: dict, where: str) -> tuple[PropertyBinding, ...]:
        found = []
        for index, prop in enumerate(self.listed(item, "properties", where)):
            at = f"{where} properties[{index}]"
            self.object(prop, _PROPERTY, at)
            path = self.string(prop, "path", at) if "path" in prop else None
            if path is not None and not path.startswith("$"):
                raise self.fail("invalid_format", at, f"path must be a JSON path starting with '$', got {path!r}")
            found.append(
                PropertyBinding(name=self.string(prop, "name", at), column=self.string(prop, "column", at), path=path)
            )
        return tuple(found)

    def column_endpoint(self, item: Any, where: str) -> ColumnEndpoint | None:
        if item is None:
            return None
        self.object(item, _COLUMN_ENDPOINT, where)
        if "columns" not in item:
            raise self.fail("invalid_format", where, "needs 'columns'")
        return ColumnEndpoint(type=self.string(item, "type", where), columns=self.strings(item, "columns", where))

    def join_endpoint(self, item: Any, where: str) -> JoinEndpoint:
        self.object(item, _JOIN_ENDPOINT, where)
        return JoinEndpoint(type=self.string(item, "type", where), property=self.string(item, "property", where))

    def object(self, item: Any, allowed: set[str], where: str) -> None:
        if not isinstance(item, dict):
            raise self.fail("invalid_format", where, "must be a mapping")
        self.known(item, allowed, where)

    def known(self, item: dict, allowed: set[str], where: str) -> None:
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise self.fail("unknown_key", where, f"has unknown key {unknown[0]!r}".lstrip())

    def listed(self, item: dict, key: str, where: str = "") -> list:
        value = item.get(key, [])
        if not isinstance(value, list):
            raise self.fail("invalid_format", where, f"{key!r} must be a list")
        return value

    def string(self, item: dict, key: str, where: str) -> str:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise self.fail("invalid_format", where, f"needs {key!r}, a non-empty string")
        return value

    def strings(self, item: dict, key: str, where: str) -> tuple[str, ...]:
        value = item.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise self.fail("invalid_format", where, f"{key} must be a list of non-empty strings")
        return tuple(value)
