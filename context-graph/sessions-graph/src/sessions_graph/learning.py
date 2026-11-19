"""The hygm `learning` profile sessions-graph owns: what derivation may change (hygm docs/contract.md, 5.5, 5.6).

Per node type: ``protected`` (never merged or retired, outside the
active-type cap) and ``catch_all`` (measured by the adoption gate). A
version-1 file implies the settings that were hardcoded before profiles
existed; ``VERSION_1`` adds unstructured2graph's rule for its two profiles,
so a stored version-1 ontology reads, and writes back, exactly as before.
"""

from hygm import HygmModel, ImpliedProfiles, combine_implied, setting, with_version_1_profiles
from unstructured2graph.profile import PERSON_LABEL, USER_LABEL, VALUE_LABELS
from unstructured2graph.profile import implied_profiles as extraction_and_conversation

#: Version-1 conventions: the fixed core derivation never merges or retires (#353).
CORE_LABELS: tuple[str, ...] = (USER_LABEL, PERSON_LABEL, *VALUE_LABELS)

#: Version-1 conventions: the types that collect what has no better type. The
#: adoption gate measures the share of mentions landing here.
CATCH_ALL_LABELS: tuple[str, ...] = ("Topic", "Artifact")


def protected_labels(model: HygmModel) -> tuple[str, ...]:
    """Node types the `learning` profile protects from merging and retirement."""
    return tuple(t.label for t in model.node_types if setting(t.profiles, "learning", "protected"))


def catch_all_labels(model: HygmModel) -> tuple[str, ...]:
    """Node types the `learning` profile marks as catch-alls."""
    return tuple(t.label for t in model.node_types if setting(t.profiles, "learning", "catch_all"))


def implied_profiles(model: HygmModel) -> ImpliedProfiles:
    """What a version-1 file implies for `learning`: the core is protected, the catch-alls are catch-alls."""
    nodes = {}
    for label in model.node_labels():
        learning = {}
        if label in CORE_LABELS:
            learning["protected"] = True
        if label in CATCH_ALL_LABELS:
            learning["catch_all"] = True
        nodes[label] = {"learning": learning} if learning else {}
    return {}, nodes


#: Everything a version-1 ontology file implies, for reading and writing stored versions.
VERSION_1 = combine_implied(extraction_and_conversation, implied_profiles)


def with_implied_profiles(model: HygmModel) -> HygmModel:
    """`model` with what a version-1 file of the same types implies, its own settings winning."""
    return with_version_1_profiles(model, VERSION_1)
