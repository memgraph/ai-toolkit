"""Profiles: what one kind of consumer needs from a model, kept out of the core (docs/contract.md, section 5).

Three are registered, all for reading conversations into a graph:

- ``extraction`` (per node type): ``identity``, how mentions merge, and
  ``value``, extracted as a claim about something else rather than as an entity.
- ``conversation`` (per model): ``speaker``, the person the conversation is
  with, and ``other_person``, what a speaker mention that is someone else is
  re-typed to (#358).
- ``learning`` (per node type): ``protected``, never merged or retired by
  derivation, and ``catch_all``, measured by the adoption gate.

A version-1 file implies the profiles that were hardcoded before profiles
existed (``implied_profiles``), so it means what it always meant.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from .identifiers import is_valid_identifier
from .types import IDENTITIES, HygmModel
from .validation import (
    PERSON_LABEL,
    USER_LABEL,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

#: Version-1 conventions: one node per mention, since every "3" in a session is a different fact.
VALUE_LABELS: tuple[str, ...] = ("Duration", "Quantity", "Money", "Date", "TimeWindow")

#: Version-1 conventions: the fixed core derivation never merges or retires (#353).
CORE_LABELS: tuple[str, ...] = (USER_LABEL, PERSON_LABEL, *VALUE_LABELS)

#: Version-1 conventions: the types that collect what has no better type. The
#: adoption gate measures the share of mentions landing here.
CATCH_ALL_LABELS: tuple[str, ...] = ("Topic", "Artifact")

REGISTERED_PROFILES = ("extraction", "conversation", "learning")


def _boolean(value: Any) -> bool:
    return isinstance(value, bool)


def _label(value: Any) -> bool:
    return is_valid_identifier(value)


#: The settings each registered profile defines at each level, with what makes a value valid.
SETTINGS: dict[tuple[str, str], dict[str, Callable[[Any], bool]]] = {
    ("extraction", "node"): {"identity": lambda value: value in IDENTITIES, "value": _boolean},
    ("conversation", "model"): {"speaker": _label, "other_person": _label},
    ("learning", "node"): {"protected": _boolean, "catch_all": _boolean},
}


def label_settings(profile: str, level: str) -> tuple[str, ...]:
    """The settings of `profile` at `level` whose value names a node type; renames and merges carry them."""
    return tuple(key for key, valid in SETTINGS.get((profile, level), {}).items() if valid is _label)


def implied_profiles(model: HygmModel) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """The profiles a version-1 file implies for `model`'s labels: (model-level, per node label)."""
    labels = model.node_labels()
    nodes: dict[str, dict[str, dict[str, Any]]] = {}
    for label in labels:
        profiles: dict[str, dict[str, Any]] = {}
        if label in VALUE_LABELS:
            profiles["extraction"] = {"value": True}
        learning = {}
        if label in CORE_LABELS:
            learning["protected"] = True
        if label in CATCH_ALL_LABELS:
            learning["catch_all"] = True
        if learning:
            profiles["learning"] = learning
        nodes[label] = profiles
    top = {"conversation": {"speaker": USER_LABEL, "other_person": PERSON_LABEL}} if USER_LABEL in labels else {}
    return top, nodes


def with_implied_profiles(model: HygmModel) -> HygmModel:
    """`model` with the profiles a version-1 file implies added, its own settings winning.

    For a model built in code that means what a version-1 file of the same
    types would: values, the core, the catch-alls and the conversation roles.
    """
    top, nodes = implied_profiles(model)
    return replace(
        model,
        node_types=tuple(replace(t, profiles=_merged(nodes[t.label], t.profiles)) for t in model.node_types),
        profiles=_merged(top, model.profiles),
    )


def _merged(base: Mapping[str, Mapping[str, Any]], own: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = _plain(base)
    for name, settings in own.items():
        merged[name] = {**merged.get(name, {}), **settings}
    return merged


def has_implied_profiles(model: HygmModel) -> bool:
    """Whether `model`'s profiles are exactly the ones a version-1 file would imply."""
    top, nodes = implied_profiles(model)
    return _plain(model.profiles) == top and all(_plain(t.profiles) == nodes[t.label] for t in model.node_types)


def _setting(profiles: Mapping[str, Mapping[str, Any]], profile: str, key: str) -> Any:
    return profiles.get(profile, {}).get(key)


def value_labels(model: HygmModel) -> tuple[str, ...]:
    """Node types the `extraction` profile marks as values."""
    return tuple(t.label for t in model.node_types if _setting(t.profiles, "extraction", "value"))


def protected_labels(model: HygmModel) -> tuple[str, ...]:
    """Node types the `learning` profile protects from merging and retirement."""
    return tuple(t.label for t in model.node_types if _setting(t.profiles, "learning", "protected"))


def catch_all_labels(model: HygmModel) -> tuple[str, ...]:
    """Node types the `learning` profile marks as catch-alls."""
    return tuple(t.label for t in model.node_types if _setting(t.profiles, "learning", "catch_all"))


def conversation(model: HygmModel) -> tuple[str, str] | None:
    """The model's (speaker, other_person) labels, or None without a `conversation` profile."""
    settings = model.profiles.get("conversation")
    if not settings:
        return None
    return settings.get("speaker", USER_LABEL), settings.get("other_person", PERSON_LABEL)


def validate_for_conversation(model: HygmModel) -> ValidationResult:
    """The `conversation` profile's rule: every endpoint admitting the speaker admits the other person.

    A speaker mention that turns out to be someone else is re-typed to the
    other person (#358), and that re-typed edge must still conform. A model
    without the profile passes. Consumers that read conversations apply this
    beside validate_model().
    """
    result = ValidationResult("conversation")
    roles = conversation(model)
    if roles is None:
        return result
    speaker, other_person = roles
    for relation in model.relation_types:
        for side in ("start", "end"):
            admitted = model.endpoint_labels(relation, side)
            if speaker in admitted and other_person not in admitted:
                result.add_issue(
                    ValidationIssue(
                        ValidationSeverity.CRITICAL,
                        ValidationCategory.STRUCTURE,
                        f"Relation type {relation.label!r}: {side}_labels includes {speaker!r} but not {other_person!r}",
                        recommendation=(
                            f"A {speaker} mention that is not the speaker is re-typed to {other_person}, "
                            "so every endpoint that accepts one must accept the other"
                        ),
                        code="profile_rule",
                    )
                )
    return result


def _plain(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: dict(settings) for name, settings in profiles.items()}
