"""Cypher identifier validation.

Cypher parameterizes values but not labels, relationship types, property keys
or variable names, so any of those built from configuration or extracted data
is interpolated into query text. Restricting them to this pattern is what keeps
that interpolation from breaking or injecting into a query.
"""

import re

CYPHER_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_identifier(value: str) -> bool:
    """Whether `value` is safe to interpolate into Cypher as an identifier."""
    return isinstance(value, str) and CYPHER_IDENTIFIER_PATTERN.match(value) is not None


def require_valid_identifier(value: str, role: str) -> str:
    """Return `value` unchanged if it is a valid Cypher identifier.

    Args:
        value: The candidate label, relationship type or property key.
        role: What `value` is, for the error message (e.g. "workspace").

    Raises:
        ValueError: if `value` is not a valid identifier.
    """
    if not is_valid_identifier(value):
        raise ValueError(
            f"Invalid {role} {value!r}: must be a valid identifier (letters, digits, "
            "underscore, not starting with a digit) to use directly in a Cypher query"
        )
    return value
