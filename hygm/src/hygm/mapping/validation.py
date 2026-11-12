"""The rules a mapping obeys against its model, beyond its format (docs/contract.md, 6.2).

Every problem found is reported, each with its contract error code, so a
mapping is fixed in one pass rather than one refusal at a time.
"""

from collections import Counter

from ..errors import ModelError
from ..types import HygmModel, RelationType
from .types import ColumnEndpoint, JoinEndpoint, Mapping, NodeBinding, PropertyBinding, RelationBinding

Problem = tuple[str, str]


def check_mapping(mapping: Mapping, model: HygmModel) -> None:
    """Refuse `mapping` unless it obeys the contract's mapping rules against `model`.

    The rules (contract, 6.2), and where the contract is silent, hygm's reading of it:

    - Source names are unique (`duplicate_source`).
    - Each binding names a declared type (`unknown_type`) and a declared
      source (`unknown_source`). A `join` names no source.
    - Each binding sets exactly one of `table`, `graph` and `join`, allowed
      for its source's kind: `table` in a `table` source, `graph` in a
      `graph` source, `join` in none (`binding_kind`). Only a `table`
      binding has `id`, `from`/`to` or bound properties (`binding_kind`), and
      a `table` binding has all of them it needs (`invalid_format`).
    - Bound properties are declared or inherited by the type (or declared by
      the relation type), each bound once per binding (`undeclared_property`,
      `duplicate_property`). A property the type forbids, by refining it to
      `max_count: 0` (4.15), counts as undeclared; a multi-valued one may be
      bound, its column holding the list. A source column's type is the source's, so a
      binding can't be checked for type compatibility here, and a JSON `path`
      may feed a property of any type; a consumer that reads the source checks both.
    - A node type bound in more than one source has a `key`, and each of its
      `table` bindings binds every key property, so equal keys merge into
      one entity (`unbound_key`). Sources count per exact type, not per
      ancestor.
    - Every relation binding's endpoint types are admitted by the relation
      type, its descendants included, and the end type by the start type's
      relation restrictions (4.16) as well (`endpoint_not_admitted`); a
      `join`'s properties are declared or inherited by its endpoint types,
      and not forbidden (`undeclared_property`).
    - A `table` relation's `from`/`to` columns reference an endpoint node:
      by the `id` of each `table` binding in the same source of that type or
      a descendant, whose column counts must all match; or, when there is
      none in that source, by the endpoint type's `key`, whose length must
      match (`endpoint_columns`); with neither, it can't reference one
      (`unbound_key`).

    Raises:
        ModelError: carrying every problem's code, if any rule fails.
    """
    problems = _Checker(mapping, model).problems()
    if problems:
        raise ModelError([code for code, _ in problems], "; ".join(message for _, message in problems))


class _Checker:
    def __init__(self, mapping: Mapping, model: HygmModel) -> None:
        self.mapping = mapping
        self.model = model
        self.found: list[Problem] = []

    def problems(self) -> list[Problem]:
        for name, count in Counter(s.name for s in self.mapping.sources).items():
            if count > 1:
                self.add("duplicate_source", f"source {name!r} is declared {count} times")
        for index, binding in enumerate(self.mapping.nodes):
            self.node(binding, f"nodes[{index}]")
        self.keys()
        for index, binding in enumerate(self.mapping.relations):
            self.relation(binding, f"relations[{index}]")
        return self.found

    def add(self, code: str, message: str) -> None:
        self.found.append((code, message))

    def node(self, binding: NodeBinding, where: str) -> None:
        declared = self.model.node_type(binding.type) is not None
        if not declared:
            self.add("unknown_type", f"{where} binds node type {binding.type!r}, which the model doesn't declare")
        if (binding.table is None) == (binding.graph is None):
            self.add("binding_kind", f"{where} must set exactly one of 'table' and 'graph'")
            return
        self.source_allows(binding.source, binding.kind, where)
        if binding.table is not None:
            if not binding.id:
                self.add("invalid_format", f"{where} binds a table, so it needs the 'id' columns identifying a row")
        elif binding.id or binding.properties:
            self.add("binding_kind", f"{where} binds a graph label, which takes no 'id' or 'properties'")
            return
        if declared:
            self.bound_properties(binding.properties, self.allowed(binding.type), f"node type {binding.type!r}", where)

    def keys(self) -> None:
        sources: dict[str, set[str]] = {}
        for binding in self.mapping.nodes:
            if self.model.node_type(binding.type) is not None:
                sources.setdefault(binding.type, set()).add(binding.source)
        for label, bound_in in sources.items():
            if len(bound_in) < 2:
                continue
            key = self.model.key_of(label)
            if not key:
                self.add(
                    "unbound_key",
                    f"node type {label!r} is bound in {len(bound_in)} sources ({', '.join(sorted(bound_in))}) "
                    "but has no key to merge them by",
                )
                continue
            for index, binding in enumerate(self.mapping.nodes):
                missing = [k for k in key if k not in {p.name for p in binding.properties}]
                if binding.type == label and binding.table is not None and missing:
                    self.add(
                        "unbound_key",
                        f"nodes[{index}] binds {label!r}, which is bound in several sources, "
                        f"without its key property {missing[0]!r}",
                    )

    def relation(self, binding: RelationBinding, where: str) -> None:
        relation = next((r for r in self.model.relation_types if r.label == binding.type), None)
        if relation is None:
            self.add("unknown_type", f"{where} binds relation type {binding.type!r}, which the model doesn't declare")
        kinds = [binding.table is not None, binding.graph is not None, binding.join is not None]
        if sum(kinds) != 1:
            self.add("binding_kind", f"{where} must set exactly one of 'table', 'graph' and 'join'")
            return
        if binding.join is not None:
            if binding.source is not None or binding.from_ or binding.to or binding.properties:
                self.add("binding_kind", f"{where} is a join, which takes no 'source', 'from', 'to' or 'properties'")
                return
            self.join_side(binding.join.from_, relation, "start", f"{where} join.from")
            self.join_side(
                binding.join.to, self.restricted(relation, binding.join.from_.type), "end", f"{where} join.to"
            )
            return
        if binding.source is None:
            self.add("invalid_format", f"{where} binds a {binding.kind}, so it needs a 'source'")
            return
        self.source_allows(binding.source, binding.kind, where)
        if binding.graph is not None:
            if binding.from_ or binding.to or binding.properties:
                self.add("binding_kind", f"{where} binds a graph, which takes no 'from', 'to' or 'properties'")
            return
        if binding.from_ is None or binding.to is None:
            self.add("invalid_format", f"{where} binds a table, so it needs 'from' and 'to'")
            return
        self.column_side(binding.from_, relation, "start", binding.source, f"{where} from")
        to_relation = self.restricted(relation, binding.from_.type)
        self.column_side(binding.to, to_relation, "end", binding.source, f"{where} to")
        if relation is not None:
            allowed = {p.name: p.max_count == 0 for p in relation.properties}
            self.bound_properties(binding.properties, allowed, f"relation type {relation.label!r}", where)

    def allowed(self, label: str) -> dict[str, bool]:
        """The properties a `label` binding may name, each with whether the type forbids it (`max_count: 0`)."""
        return {p.name: p.max_count == 0 for p in self.model.properties_of(label)}

    def restricted(self, relation: RelationType | None, start: str) -> RelationType | None:
        """`relation` as a `start` instance has it, its restrictions applied; as declared if `start` can't start it."""
        if relation is None or self.model.node_type(start) is None:
            return relation
        return self.model.relation_for(start, relation.label) or relation

    def source_allows(self, name: str, kind: str, where: str) -> None:
        source = self.mapping.source(name)
        if source is None:
            self.add("unknown_source", f"{where} names source {name!r}, which the mapping doesn't declare")
        elif source.kind != kind:
            self.add("binding_kind", f"{where} binds a {kind}, which a {source.kind} source {name!r} can't hold")

    def bound_properties(
        self, properties: tuple[PropertyBinding, ...], allowed: dict[str, bool], owner: str, where: str
    ) -> None:
        for name, count in Counter(p.name for p in properties).items():
            if count > 1:
                self.add("duplicate_property", f"{where} binds property {name!r} {count} times")
            if name not in allowed:
                self.add("undeclared_property", f"{where} binds property {name!r}, which {owner} doesn't declare")
            elif allowed[name]:
                self.add("undeclared_property", f"{where} binds property {name!r}, which {owner} forbids (max_count 0)")

    def endpoint(self, label: str, relation: RelationType | None, side: str, where: str) -> bool:
        """Whether `label` is a declared type the relation admits at `side`, reporting why not."""
        if self.model.node_type(label) is None:
            self.add("unknown_type", f"{where} names node type {label!r}, which the model doesn't declare")
            return False
        if relation is None:
            return True
        if label not in self.model.endpoint_labels(relation, "start" if side == "start" else "end"):
            declared = self.model.relation_type(relation.label)
            restricted = " for its start type" if declared is not None and declared != relation else ""
            self.add(
                "endpoint_not_admitted",
                f"{where} names {label!r}, which relation type {relation.label!r} doesn't admit at its {side}"
                f"{restricted}",
            )
        return True

    def join_side(self, endpoint: JoinEndpoint, relation: RelationType | None, side: str, where: str) -> None:
        if not self.endpoint(endpoint.type, relation, side, where):
            return
        forbidden = self.allowed(endpoint.type).get(endpoint.property)
        if forbidden is None:
            self.add(
                "undeclared_property",
                f"{where} joins on {endpoint.property!r}, which node type {endpoint.type!r} doesn't declare",
            )
        elif forbidden:
            self.add(
                "undeclared_property",
                f"{where} joins on {endpoint.property!r}, which node type {endpoint.type!r} forbids (max_count 0)",
            )

    def column_side(
        self, endpoint: ColumnEndpoint, relation: RelationType | None, side: str, source: str, where: str
    ) -> None:
        if not endpoint.columns:
            self.add("invalid_format", f"{where} needs at least one column")
            return
        if not self.endpoint(endpoint.type, relation, side, where):
            return
        count = len(endpoint.columns)
        targets = [
            b
            for b in self.mapping.nodes
            if b.source == source
            and b.table is not None
            and self.model.node_type(b.type) is not None
            and self.model.is_a(b.type, endpoint.type)
        ]
        if targets:
            for target in targets:
                if len(target.id) != count:
                    self.add(
                        "endpoint_columns",
                        f"{where} has {count} column(s), but the {target.type!r} binding of table "
                        f"{target.table.name if target.table else ''!r} in source {source!r} "
                        f"has an id of {len(target.id)}",
                    )
            return
        key = self.model.key_of(endpoint.type)
        if not key:
            self.add(
                "unbound_key",
                f"{where} references {endpoint.type!r}, which has no table binding in source {source!r} "
                "and no key to reference it by",
            )
        elif len(key) != count:
            self.add(
                "endpoint_columns",
                f"{where} has {count} column(s), but {endpoint.type!r} is referenced by its key of {len(key)}",
            )
