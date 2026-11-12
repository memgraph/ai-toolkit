"""Property facets: what a value must be, narrowing them in a subtype, and merging two (docs/contract.md, 4.14, 4.15).

Values are JSON values: `date` and `datetime` ones are ISO 8601 strings,
compared as the dates and times they name. A multi-valued property's value
is a list, and `values`, the bounds, `default` and `value` apply to each
element. Its `default` and `value` are whole values, so lists too.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from .errors import ModelError

if TYPE_CHECKING:
    from .types import Property

#: The property types that take `minimum` and `maximum`.
ORDERED_TYPES: tuple[str, ...] = ("integer", "float", "decimal", "date", "datetime")


def is_multi_valued(max_count: int | None) -> bool:
    """Whether a property with this `max_count` holds a list: no limit, or more than one value."""
    return max_count is None or max_count > 1


def _parsed(prop_type: str, value: Any) -> Any:
    """`value` as something comparable for `prop_type`; raises ValueError if it isn't one of its values."""
    if prop_type == "string":
        if not isinstance(value, str):
            raise ValueError("not a string")
        return value
    if prop_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("not an integer")
        return value
    if prop_type in ("float", "decimal"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("not a number")
        return value
    if prop_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError("not true or false")
        return value
    if prop_type == "date":
        if not isinstance(value, str):
            raise ValueError("not an ISO 8601 date string")
        return date.fromisoformat(value)
    if prop_type == "datetime":
        if not isinstance(value, str):
            raise ValueError("not an ISO 8601 date-time string")
        # Python before 3.11 doesn't read the `Z` suffix ISO 8601 allows.
        return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    return value  # json: any JSON value


def _below(a: Any, b: Any) -> bool:
    """Whether `a` < `b`; raises ValueError for values that can't be ordered, like naive and zoned times."""
    try:
        return a < b
    except TypeError as e:
        raise ValueError(f"{a!r} and {b!r} can't be compared") from e


def element_problem(prop: Property, value: Any) -> str | None:
    """Why `value` can't be one value (one element, for a multi-valued property) of `prop`, or None."""
    try:
        parsed = _parsed(prop.type, value)
        if prop.values is not None and value not in prop.values:
            return f"{value!r} is not one of {list(prop.values)}"
        if prop.minimum is not None and _below(parsed, _parsed(prop.type, prop.minimum)):
            return f"{value!r} is below the minimum {prop.minimum!r}"
        if prop.maximum is not None and _below(_parsed(prop.type, prop.maximum), parsed):
            return f"{value!r} is above the maximum {prop.maximum!r}"
    except ValueError as e:
        return f"{value!r} is not a valid {prop.type} value: {e}"
    return None


def value_problem(prop: Property, value: Any) -> tuple[str, str] | None:
    """Why `value` can't be a whole value of `prop`, as (code, reason), or None.

    A multi-valued property takes a list whose length is within the counts
    (`count_conflict` otherwise); a single-valued one takes one value.
    """
    if not is_multi_valued(prop.max_count):
        problem = element_problem(prop, value)
        return ("invalid_value", problem) if problem else None
    if not isinstance(value, list | tuple):
        return "invalid_value", f"{value!r} is not a list, and the property is multi-valued"
    if len(value) < prop.least_count or (prop.max_count is not None and len(value) > prop.max_count):
        most = "any number" if prop.max_count is None else prop.max_count
        return "count_conflict", f"{len(value)} values, outside {prop.least_count} to {most}"
    for element in value:
        problem = element_problem(prop, element)
        if problem:
            return "invalid_value", problem
    return None


def check_facets(prop: Property) -> None:
    """Refuse a property whose facets contradict each other.

    Raises:
        ModelError: `count_conflict` for `min_count` above `max_count`;
            `invalid_value` for bounds on a type without an order, bounds the
            wrong way round, or a `default` or `value` the facets don't allow.
    """
    where = f"Property {prop.name!r}"
    if prop.max_count is not None and prop.least_count > prop.max_count:
        raise ModelError("count_conflict", f"{where}: min_count {prop.least_count} is above max_count {prop.max_count}")
    bounds = {"minimum": prop.minimum, "maximum": prop.maximum}
    for key, bound in bounds.items():
        if bound is None:
            continue
        if prop.type not in ORDERED_TYPES:
            raise ModelError("invalid_value", f"{where}: {key} is only allowed on {ORDERED_TYPES}, not {prop.type!r}")
        try:
            _parsed(prop.type, bound)
        except ValueError as e:
            raise ModelError("invalid_value", f"{where}: {key} {bound!r} is not a valid {prop.type}: {e}") from e
    if prop.minimum is not None and prop.maximum is not None:
        try:
            wrong_way = _below(_parsed(prop.type, prop.maximum), _parsed(prop.type, prop.minimum))
        except ValueError as e:
            raise ModelError("invalid_value", f"{where}: {e}") from e
        if wrong_way:
            raise ModelError("invalid_value", f"{where}: minimum {prop.minimum!r} is above maximum {prop.maximum!r}")
    for key in ("default", "value"):
        value = getattr(prop, key)
        if value is None:
            continue
        problem = value_problem(prop, value)
        if problem:
            raise ModelError("invalid_value", f"{where}: {key} {problem[1]}")


class FacetConflictError(ValueError):
    """Two declarations of one property that can't both hold (docs/contract.md, 4.15): `property_conflict`."""


def refine(inherited: Property, declared: Property) -> Property:
    """`declared`, redeclaring `inherited` in a subtype, with what it leaves unstated inherited (4.15).

    A redeclaration states what it narrows. `values`, `minimum`, `maximum`,
    `value`, `default` and `unit` it leaves as None, a `min_count` of 0 (not
    required), and an empty `description`, `synonyms` or `names` come from
    `inherited`. Its `type`, `max_count`, `iri` and profiles are its own: a model file's reader fills an unstated
    `type` and `max_count` from the inherited property, since their defaults
    aren't "unstated".

    Raises:
        ModelError: when the inherited facets contradict the declared ones,
            such as an inherited `value` outside the declared `values`.
    """
    least = declared.least_count or inherited.least_count
    return replace(
        declared,
        values=declared.values if declared.values is not None else inherited.values,
        required=least >= 1,
        min_count=least,
        minimum=declared.minimum if declared.minimum is not None else inherited.minimum,
        maximum=declared.maximum if declared.maximum is not None else inherited.maximum,
        value=declared.value if declared.value is not None else inherited.value,
        default=declared.default if declared.default is not None else inherited.default,
        unit=declared.unit if declared.unit is not None else inherited.unit,
        description=declared.description or inherited.description,
        synonyms=declared.synonyms or inherited.synonyms,
        names=declared.names or inherited.names,
    )


def narrowing_problem(inherited: Property, declared: Property) -> str | None:
    """Why `declared`, redeclaring `inherited` in a subtype, doesn't only narrow it, or None (4.15).

    Its effective facets (refine()) may name fewer `values`, raise the
    counts' minimum, lower their maximum (keeping a multi-valued property
    multi-valued, or to 0 to forbid it), tighten the bounds, and fix a
    `value` the inherited facets allow; an inherited `value` stays. It may
    change `default`, `description`, `synonyms`, `names`, `unit`, `iri` and
    profiles. It may not change the type or widen anything.
    """
    if declared.type != inherited.type:
        return f"changes the type from {inherited.type!r} to {declared.type!r}"
    if 0 < declared.least_count < inherited.least_count:
        return f"lowers min_count from {inherited.least_count} to {declared.least_count}"
    try:
        refined = refine(inherited, declared)
    except ModelError as e:
        return f"contradicts the inherited facets: {e}"
    if inherited.values is not None and not set(refined.values or ()) <= set(inherited.values):
        return f"allows values beyond the inherited {list(inherited.values)}"
    if inherited.max_count is not None and (refined.max_count is None or refined.max_count > inherited.max_count):
        return f"raises max_count above {inherited.max_count}"
    # A subtype's instances are its parent's, so a list stays a list: narrowing
    # one to max_count 1 would make the same value a list or not by type.
    if is_multi_valued(inherited.max_count) and refined.max_count == 1:
        return "makes a multi-valued property single-valued; narrow max_count to 2 or more, or 0 to forbid it"
    try:
        if inherited.minimum is not None and _below(
            _parsed(refined.type, refined.minimum), _parsed(inherited.type, inherited.minimum)
        ):
            return f"lowers the minimum below {inherited.minimum!r}"
        if inherited.maximum is not None and _below(
            _parsed(inherited.type, inherited.maximum), _parsed(refined.type, refined.maximum)
        ):
            return f"raises the maximum above {inherited.maximum!r}"
    except ValueError as e:
        return str(e)
    if inherited.value is not None and refined.value != inherited.value:
        return f"changes the fixed value {inherited.value!r}"
    return None


def merge(a: Property, b: Property) -> Property:
    """The narrowest facets both `a` and `b` allow: a property inherited from two parents (4.7, 4.15).

    Documentation (`description`, `unit`, `synonyms`, `names`, `iri`,
    profiles) and a `default` come from `a`, else `b`.

    Raises:
        FacetConflictError: when no value satisfies both.
    """
    if a.type != b.type:
        raise FacetConflictError(f"types {a.type!r} and {b.type!r} differ")
    values = a.values if b.values is None else b.values if a.values is None else None
    if a.values is not None and b.values is not None:
        values = tuple(v for v in a.values if v in b.values)
        if not values:
            raise FacetConflictError(f"no value is in both {list(a.values)} and {list(b.values)}")
    most = [m for m in (a.max_count, b.max_count) if m is not None]
    if is_multi_valued(a.max_count) != is_multi_valued(b.max_count) and 0 not in most:
        raise FacetConflictError("one declaration is multi-valued and the other single-valued")
    if a.value is not None and b.value is not None and a.value != b.value:
        raise FacetConflictError(f"fixed values {a.value!r} and {b.value!r} differ")
    try:
        minimum = _tighter(a.type, a.minimum, b.minimum, lower=False)
        maximum = _tighter(a.type, a.maximum, b.maximum, lower=True)
        return replace(
            a,
            values=values,
            min_count=max(a.least_count, b.least_count),
            required=max(a.least_count, b.least_count) >= 1,
            max_count=min(most) if most else None,
            minimum=minimum,
            maximum=maximum,
            value=a.value if a.value is not None else b.value,
            default=a.default if a.default is not None else b.default,
            unit=a.unit or b.unit,
        )
    except (ModelError, ValueError) as e:
        raise FacetConflictError(str(e)) from e


def _tighter(prop_type: str, a: Any, b: Any, *, lower: bool) -> Any:
    """Of two bounds, the one that admits less: the lower maximum or the higher minimum."""
    if a is None or b is None:
        return b if a is None else a
    b_first = _below(_parsed(prop_type, b), _parsed(prop_type, a))
    return (b if b_first else a) if lower else (a if b_first else b)
