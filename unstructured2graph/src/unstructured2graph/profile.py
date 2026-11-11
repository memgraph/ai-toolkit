"""The two hygm profiles unstructured2graph owns: what their settings mean (hygm docs/contract.md, 5.3, 5.4, 5.6).

- ``extraction`` (per node type): ``identity``, when two mentions are one
  node, and ``value``, a type extracted as a claim about something else.
- ``conversation`` (per model): ``speaker``, the person a conversation is
  with, and ``other_person``, what the mention resolver re-types a speaker
  mention to when it is someone else (#358).

hygm checks the shape of these settings; this module is where they mean
something. A version-1 model file implies the settings that were hardcoded
before profiles existed (``implied_profiles``).
"""

from typing import Literal

from hygm import (
    HygmModel,
    ImpliedProfiles,
    NodeType,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    setting,
    with_version_1_profiles,
)

Identity = Literal["global", "chunk", "span"]

#: How a node type's mentions become nodes (#346, #361):
#:   global -- one node per normalized text across every chunk (named things that recur)
#:   chunk  -- one node per normalized text per chunk (generic nouns)
#:   span   -- one node per mention (values: every "3" in a session is a different fact)
IDENTITIES: tuple[Identity, ...] = ("global", "chunk", "span")

#: Version-1 conventions: the speaker, and the person a speaker mention that is someone else is re-typed to.
USER_LABEL = "User"
PERSON_LABEL = "Person"

#: Version-1 conventions: the value types, one node per mention.
VALUE_LABELS: tuple[str, ...] = ("Duration", "Quantity", "Money", "Date", "TimeWindow")


def identity_of(node: NodeType) -> Identity:
    """How `node`'s mentions merge into nodes; `chunk` unless its `extraction` profile says otherwise."""
    return setting(node.profiles, "extraction", "identity", "chunk")


def value_labels(model: HygmModel) -> tuple[str, ...]:
    """Node types the `extraction` profile marks as values."""
    return tuple(t.label for t in model.node_types if setting(t.profiles, "extraction", "value"))


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
    without the profile passes. Readers of conversations apply this beside
    hygm's validate_model().
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


def implied_profiles(model: HygmModel) -> ImpliedProfiles:
    """What a version-1 file implies for these two profiles: the value types, and the conversation roles."""
    labels = model.node_labels()
    nodes = {label: ({"extraction": {"value": True}} if label in VALUE_LABELS else {}) for label in labels}
    top = {"conversation": {"speaker": USER_LABEL, "other_person": PERSON_LABEL}} if USER_LABEL in labels else {}
    return top, nodes


def with_implied_profiles(model: HygmModel) -> HygmModel:
    """`model` with what a version-1 file of the same types implies for these profiles, its own settings winning."""
    return with_version_1_profiles(model, implied_profiles)
