"""Profiles: what one kind of consumer needs from a model, kept out of the core (docs/contract.md, section 5).

hygm owns the mechanism and the contract data only: the registry of profile
names, each with its owner and the shape of its settings at each level, so
any reader checks every registered profile's settings without its owner
installed. What a setting means, and the rules a profile adds, belong to the
owner (docs/contract.md, 5.6).
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from .identifiers import is_valid_identifier
from .types import HygmModel

#: Profile settings by profile name, as a model, a type or a property holds them.
Profiles = Mapping[str, Mapping[str, Any]]

#: The profiles a version-1 file implies: (model profiles, profiles per node label).
ImpliedProfiles = tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]

#: An owner's rule for what a version-1 file implies beyond its node-level
#: `identity` (docs/contract.md, 4.9). The reader adds it; the writer keeps a
#: model in version 1 when its profiles are exactly it.
Version1Profiles = Callable[[HygmModel], ImpliedProfiles]

#: `extraction.identity`'s values: the one setting the version-1 format itself
#: carries, as a node type's `identity`.
IDENTITIES: tuple[str, ...] = ("global", "chunk", "span")


def _boolean(value: Any) -> bool:
    return isinstance(value, bool)


def _label(value: Any) -> bool:
    return is_valid_identifier(value)


def _identity(value: Any) -> bool:
    return value in IDENTITIES


#: The registry (docs/contract.md, 5.2 and 5.6): each registered profile's owner.
OWNERS: Mapping[str, str] = {
    "extraction": "unstructured2graph",
    "conversation": "unstructured2graph",
    "learning": "sessions-graph",
}

REGISTERED_PROFILES: tuple[str, ...] = tuple(OWNERS)

#: The shape of each registered profile's settings at each level, with what makes a value valid.
SETTINGS: dict[tuple[str, str], dict[str, Callable[[Any], bool]]] = {
    ("extraction", "node"): {"identity": _identity, "value": _boolean},
    ("conversation", "model"): {"speaker": _label, "other_person": _label},
    ("learning", "node"): {"protected": _boolean, "catch_all": _boolean},
}


def label_settings(profile: str, level: str) -> tuple[str, ...]:
    """The settings of `profile` at `level` whose value names a node type; renames and merges carry them."""
    return tuple(key for key, valid in SETTINGS.get((profile, level), {}).items() if valid is _label)


def setting(profiles: Profiles, profile: str, key: str, default: Any = None) -> Any:
    """`profiles[profile][key]`, or `default` when either is absent."""
    return profiles.get(profile, {}).get(key, default)


def no_implied_profiles(model: HygmModel) -> ImpliedProfiles:
    """The version-1 rule without an owner's: nothing beyond the node-level `identity`."""
    return {}, {label: {} for label in model.node_labels()}


def merge_profiles(base: Profiles, own: Profiles) -> dict[str, dict[str, Any]]:
    """`base` with `own` laid over it, setting by setting."""
    merged = plain_profiles(base)
    for name, settings in own.items():
        merged[name] = {**merged.get(name, {}), **settings}
    return merged


def combine_implied(*rules: Version1Profiles) -> Version1Profiles:
    """One version-1 rule from several owners' rules, laid over each other in order."""

    def combined(model: HygmModel) -> ImpliedProfiles:
        top: dict[str, dict[str, Any]] = {}
        nodes: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in model.node_labels()}
        for rule in rules:
            rule_top, rule_nodes = rule(model)
            top = merge_profiles(top, rule_top)
            for label, profiles in rule_nodes.items():
                nodes[label] = merge_profiles(nodes.get(label, {}), profiles)
        return top, nodes

    return combined


def with_version_1_profiles(model: HygmModel, rule: Version1Profiles) -> HygmModel:
    """`model` with what `rule` says a version-1 file implies added, its own settings winning."""
    top, nodes = rule(model)
    return replace(
        model,
        node_types=tuple(
            replace(t, profiles=merge_profiles(nodes.get(t.label, {}), t.profiles)) for t in model.node_types
        ),
        profiles=merge_profiles(top, model.profiles),
    )


def has_version_1_profiles(model: HygmModel, rule: Version1Profiles) -> bool:
    """Whether `model`'s profiles, apart from `extraction.identity`, are exactly what `rule` implies."""
    top, nodes = rule(model)
    return plain_profiles(model.profiles) == plain_profiles(top) and all(
        _without_identity(t.profiles) == plain_profiles(nodes.get(t.label, {})) for t in model.node_types
    )


def plain_profiles(profiles: Profiles) -> dict[str, dict[str, Any]]:
    """`profiles` as plain dicts, dropping profiles with no settings."""
    return {name: dict(settings) for name, settings in profiles.items() if settings}


def _without_identity(profiles: Profiles) -> dict[str, dict[str, Any]]:
    plain = plain_profiles(profiles)
    extraction = {k: v for k, v in plain.get("extraction", {}).items() if k != "identity"}
    if extraction:
        plain["extraction"] = extraction
    else:
        plain.pop("extraction", None)
    return plain
