"""Seeding a model's individuals into Memgraph, on request (docs/contract.md, 4.17 and 7.1).

Individuals are data, not schema, so the plan never seeds them by itself:
a caller asks with seed_statements(), and apply_seed() runs them. Each
individual is one `MERGE` on its type's label, keyed by the type's `key`
(own or inherited) or, for a type without one, by an `id` property holding
the individual's id, so seeding twice leaves one copy. Its relationships
are `MERGE`d between those nodes afterwards. What the file states is what is
written: an inverse or symmetric relationship the model implies is not
written a second time, the other way round.

Values are written the way the plan's constraints expect: a `values`
property through `ToEnum` (elementwise for a list), a `date` through
`date()`, a `datetime` through `datetime()` (with an offset) or
`localDateTime()`, and an integer given for a `float` or `decimal` property
as a float. A property's fixed `value`, and its `default` where the
individual sets none, are written too.

Security boundary: every value, and every enum name, travels as a Cypher
parameter, never in the query text. Labels, relationship types and property
names can't be parameterized, so they are interpolated, which is safe only
because hygm's types refuse anything but a valid Cypher identifier when they
are constructed (docs/contract.md, 4.10); they are backtick-quoted as well, so
an identifier that is also a keyword (`Order`) still parses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import ModelError
from .memgraph import _name, _require_valid, enum_for, labels_for

if TYPE_CHECKING:
    from ..types import HygmModel, Individual, Property

#: Runs one Cypher statement with its parameters and returns its rows.
RunWithParameters = Callable[[str, Mapping[str, Any]], Iterable[Mapping[str, Any]]]


@dataclass(frozen=True)
class SeedStatement:
    """One parameterized statement of a seed.

    Attributes:
        cypher: The statement, with every value a `$parameter`.
        parameters: The values, by parameter name.
    """

    cypher: str
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)


def seed_statements(model: HygmModel) -> tuple[SeedStatement, ...]:
    """The statements that seed `model`'s individuals: their nodes, in order, then their relationships.

    Idempotent: applying them twice leaves one node per individual and one
    relationship per stated pair.

    Raises:
        ModelError: if `model` fails validate_model(), its individuals
            included; or `invalid_value` if a key-less type's individual sets
            an `id` property other than its own id, which would key it twice.
    """
    _require_valid(model)
    nodes = tuple(_node(model, individual) for individual in model.individuals)
    by_id = {individual.id: individual for individual in model.individuals}
    relationships = tuple(
        _relationship(model, individual, name, by_id[target])
        for individual in model.individuals
        for name, targets in individual.relations.items()
        for target in targets
    )
    return (*nodes, *relationships)


def apply_seed(statements: Iterable[SeedStatement], run: RunWithParameters) -> None:
    """Run each seed statement in order.

    Args:
        statements: What seed_statements() returned.
        run: Runs one statement with its parameters, e.g.
            `lambda q, p: list(session.run(q, p))`. hygm needs no client.
    """
    for statement in statements:
        list(run(statement.cypher, statement.parameters))


def _key(model: HygmModel, individual: Individual) -> dict[str, Any]:
    """The property values `individual` is merged by: its type's key, else `id`."""
    key = model.key_of(individual.type)
    if key:
        return {name: individual.properties[name] for name in key}
    if individual.properties.get("id", individual.id) != individual.id:
        raise ModelError(
            "invalid_value",
            f"Individual {individual.id!r}: its type has no key, so it is seeded by an id property, "
            f"but it sets id to {individual.properties['id']!r}",
        )
    return {"id": individual.id}


def _match(model: HygmModel, individual: Individual, variable: str, prefix: str) -> tuple[str, dict[str, Any]]:
    key = _key(model, individual)
    parameters = {f"{prefix}{index}": value for index, value in enumerate(key.values())}
    pattern = ", ".join(f"{_name(name)}: ${prefix}{index}" for index, name in enumerate(key))
    return f"({variable}:{_name(individual.type)} {{{pattern}}})", parameters


def _node(model: HygmModel, individual: Individual) -> SeedStatement:
    pattern, parameters = _match(model, individual, "n", "k")
    declared = {p.name: p for p in model.properties_of(individual.type)}
    values = dict(individual.properties)
    for prop in declared.values():
        if prop.value is not None:
            values[prop.name] = prop.value
        elif prop.default is not None:
            values.setdefault(prop.name, prop.default)
    assignments = []
    for index, (name, value) in enumerate(values.items()):
        parameters[f"v{index}"] = _plain(declared[name], value)
        expression = _expression(model, individual.type, declared[name], f"$v{index}", parameters, index)
        assignments.append(f"n.{_name(name)} = {expression}")
    labels = "".join(f":{_name(label)}" for label in labels_for(model, individual.type)[1:])
    sets = [f"n{labels}"] if labels else []
    sets += assignments
    cypher = f"MERGE {pattern}" + (f" SET {', '.join(sets)}" if sets else "")
    return SeedStatement(cypher, parameters)


def _plain(prop: Property, value: Any) -> Any:
    """`value` as the parameter Memgraph's typed constraint takes: a float for a float, lists as lists."""
    if isinstance(value, list | tuple):
        return [_plain(prop, element) for element in value]
    if prop.type in ("float", "decimal") and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _expression(
    model: HygmModel, label: str, prop: Property, parameter: str, parameters: dict[str, Any], index: int
) -> str:
    """The Cypher expression that writes `parameter` the way `prop`'s constraints expect."""
    convert = _conversion(model, label, prop, parameters, index)
    if convert is None:
        return parameter
    if prop.multi_valued:
        return f"[x IN {parameter} | {convert('x')}]"
    return convert(parameter)


def _conversion(
    model: HygmModel, label: str, prop: Property, parameters: dict[str, Any], index: int
) -> Callable[[str], str] | None:
    enum = enum_for(model, label, prop.name)
    if enum is not None:
        parameters[f"e{index}"] = enum
        return lambda value: f"ToEnum($e{index}, {value})"
    if prop.type == "date":
        return lambda value: f"date({value})"
    if prop.type == "datetime":
        # A zoned value stays zoned; one without an offset is local, as Memgraph's two types say.
        return lambda value: (
            f"CASE WHEN {value} =~ '.*(Z|[+-][0-9]{{2}}:?[0-9]{{2}})$' "
            f"THEN datetime({value}) ELSE localDateTime({value}) END"
        )
    return None


def _relationship(model: HygmModel, start: Individual, name: str, end: Individual) -> SeedStatement:
    a, a_parameters = _match(model, start, "a", "a")
    b, b_parameters = _match(model, end, "b", "b")
    return SeedStatement(f"MATCH {a}, {b} MERGE (a)-[:{_name(name)}]->(b)", {**a_parameters, **b_parameters})
