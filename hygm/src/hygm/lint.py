"""Lint: the design guidelines a model is checked against (docs/contract.md, 9.1).

Validation is the gate; lint only reports. lint_model() never raises for a
finding and never refuses a model: a consumer that wants a gate builds it
from the codes.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from .types import CASES, HygmModel, Property, RelationType

Severity = Literal["warning", "info"]

#: Words that name the modeling construct rather than the thing (9.1, `meta_word_in_name`).
META_WORDS = ("class", "type", "property", "slot", "relation", "concept")

#: More direct subtypes than this suggests missing intermediate categories (9.1, `many_subtypes`).
MANY_SUBTYPES = 12

_CASE_PATTERNS = {
    "upper_camel": re.compile(r"^[A-Z][A-Za-z0-9]*$"),
    "lower_camel": re.compile(r"^[a-z][A-Za-z0-9]*$"),
    "snake": re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"),
    "upper_snake": re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$"),
}
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


@dataclass(frozen=True)
class Finding:
    """One guideline a model doesn't follow.

    Attributes:
        code: The guideline, as 9.1 names it.
        severity: `warning` or `info`.
        element: What it is about: `node:<label>`, `relation:<label>`,
            `property:<label>.<name>` (on a node type) or
            `relation_property:<label>.<name>`, or `ontology`.
        message: What was found, for people.
        suggestion: What to consider doing about it.
    """

    code: str
    severity: Severity
    element: str
    message: str
    suggestion: str


def lint_model(model: HygmModel) -> list[Finding]:
    """Every finding of the design guidelines on `model`, in a stable order: by guideline, then by element.

    Meant for a model that passed validate_model(); on one that didn't, it
    still returns what it can find rather than raising.
    """
    return [finding for check in _CHECKS for finding in check(_Model(model))]


class _Model:
    """The model with the indexes the checks share."""

    def __init__(self, model: HygmModel) -> None:
        self.model = model
        self.subtypes: dict[str, list[str]] = {t.label: [] for t in model.node_types}
        for node in model.node_types:
            for parent in node.parents:
                if parent in self.subtypes and node.label not in self.subtypes[parent]:
                    self.subtypes[parent].append(node.label)

    def declares(self, label: str, name: str) -> Property | None:
        node = self.model.node_type(label)
        return next((p for p in node.properties if p.name == name), None) if node else None

    def has(self, label: str, name: str) -> bool:
        return any(p.name == name for p in self.model.properties_of(label))


def _words(label: str) -> list[str]:
    """`label` split into lowercase words, at underscores and case changes."""
    return [word.lower() for part in label.split("_") for word in _WORD.findall(part)]


def _singular(word: str) -> str:
    """A rough English singular: enough to tell `Wines` from `Wine`."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _is_plural(label: str) -> bool:
    words = _words(label)
    return bool(words) and _singular(words[-1]) != words[-1]


def _node(label: str) -> str:
    return f"node:{label}"


# Hierarchy shape


def _single_subtype(m: _Model) -> Iterator[Finding]:
    for label, subtypes in m.subtypes.items():
        if len(subtypes) == 1:
            yield Finding(
                "single_subtype",
                "warning",
                _node(label),
                f"{label!r} has exactly one direct subtype, {subtypes[0]!r}",
                "add its sibling subtypes, or fold the subtype into its parent",
            )


def _many_subtypes(m: _Model) -> Iterator[Finding]:
    for label, subtypes in m.subtypes.items():
        if len(subtypes) > MANY_SUBTYPES:
            yield Finding(
                "many_subtypes",
                "info",
                _node(label),
                f"{label!r} has {len(subtypes)} direct subtypes",
                "group them under intermediate categories",
            )


def _empty_subtype(m: _Model) -> Iterator[Finding]:
    named = {label for r in m.model.relation_types for label in (*r.start_labels, *r.end_labels)}
    for node in m.model.node_types:
        if node.parents and not (node.properties or node.relation_restrictions or node.key or node.label in named):
            yield Finding(
                "empty_subtype",
                "info",
                _node(node.label),
                f"{node.label!r} adds no property, refinement or relation to {', '.join(node.parents)}",
                "fine in a terminological hierarchy; otherwise say what sets it apart, or drop it",
            )


def _unused_type(m: _Model) -> Iterator[Finding]:
    named = {label for r in m.model.relation_types for label in (*r.start_labels, *r.end_labels)} | {
        label for t in m.model.node_types for r in t.relation_restrictions for label in r.end_labels or ()
    }
    profiled = {
        value for settings in m.model.profiles.values() for value in settings.values() if isinstance(value, str)
    }
    instantiated = {i.type for i in m.model.individuals}
    used = named | profiled | instantiated
    for node in m.model.node_types:
        # A relation naming an ancestor admits the type too, so that uses it as well.
        if used & {node.label, *m.model.ancestors(node.label)} or m.subtypes[node.label]:
            continue
        yield Finding(
            "unused_type",
            "info",
            _node(node.label),
            f"no relation names {node.label!r} or an ancestor, no type inherits from it and no individual has it",
            "connect it with a relation type, or drop it if nothing needs it",
        )


# Endpoints


def _endpoint_lists(m: _Model) -> Iterator[tuple[RelationType, str, tuple[str, ...]]]:
    for relation in m.model.relation_types:
        yield relation, "start", relation.start_labels
        yield relation, "end", relation.end_labels


def _redundant_endpoint(m: _Model) -> Iterator[Finding]:
    for relation, side, labels in _endpoint_lists(m):
        for label in labels:
            covering = next((other for other in labels if other != label and m.model.is_a(label, other)), None)
            if covering is not None:
                yield Finding(
                    "redundant_endpoint",
                    "warning",
                    f"relation:{relation.label}",
                    f"its {side} labels name {label!r} and its ancestor {covering!r}, which already admits it",
                    f"drop {label!r}",
                )


def _endpoint_covers_all_subtypes(m: _Model) -> Iterator[Finding]:
    for relation, side, labels in _endpoint_lists(m):
        for parent, subtypes in m.subtypes.items():
            if len(subtypes) > 1 and parent not in labels and set(subtypes) <= set(labels):
                yield Finding(
                    "endpoint_covers_all_subtypes",
                    "info",
                    f"relation:{relation.label}",
                    f"its {side} labels name every subtype of {parent!r} but not {parent!r}",
                    f"name {parent!r} instead",
                )


# Attachment


def _property_on_every_subtype(m: _Model) -> Iterator[Finding]:
    for parent, subtypes in m.subtypes.items():
        if len(subtypes) < 2:
            continue
        shared = None
        for label in subtypes:
            node = m.model.node_type(label)
            own = {(p.name, p.type) for p in node.properties} if node else set()
            shared = own if shared is None else shared & own
        for name, kind in sorted(shared or ()):
            if not m.has(parent, name):
                yield Finding(
                    "property_on_every_subtype",
                    "info",
                    _node(parent),
                    f"every subtype of {parent!r} declares {name!r} ({kind}) and {parent!r} doesn't",
                    f"declare {name!r} on {parent!r}",
                )


# Naming


def _plural_of_parent(m: _Model) -> Iterator[Finding]:
    for node in m.model.node_types:
        for ancestor in m.model.ancestors(node.label):
            if node.label != ancestor and [_singular(w) for w in _words(node.label)] == [
                _singular(w) for w in _words(ancestor)
            ]:
                yield Finding(
                    "plural_of_parent",
                    "warning",
                    _node(node.label),
                    f"{node.label!r} is the plural or singular of its ancestor {ancestor!r}",
                    "a subtype is a kind of its parent: name what sets it apart",
                )


def _style(wanted: str | None, labels: list[str]) -> str | None:
    """The convention `labels` are held to: `wanted`, else the style most of them follow."""
    if wanted is not None or not labels:
        return wanted
    counts = Counter(case for label in labels for case in CASES if _CASE_PATTERNS[case].match(label))
    return max(CASES, key=lambda case: counts[case]) if counts else None


def _naming_case(m: _Model) -> Iterator[Finding]:
    conventions = m.model.ontology.conventions if m.model.ontology else None
    groups: list[tuple[str, str | None, list[tuple[str, str]]]] = [
        (
            "node",
            conventions.node_case if conventions else None,
            [(t.label, _node(t.label)) for t in m.model.node_types],
        ),
        (
            "relation",
            conventions.relation_case if conventions else None,
            [(r.label, f"relation:{r.label}") for r in m.model.relation_types],
        ),
        (
            "property",
            conventions.property_case if conventions else None,
            [(name, element) for name, element, _ in _properties(m)],
        ),
    ]
    for kind, wanted, named in groups:
        style = _style(wanted, [name for name, _ in named])
        if style is None:
            continue
        for name, element in named:
            if not _CASE_PATTERNS[style].match(name):
                yield Finding(
                    "naming_case",
                    "warning",
                    element,
                    f"{kind} name {name!r} isn't {style}",
                    f"rename it {style}" + ("" if wanted else ", the style most of the model's names follow"),
                )


def _naming_number(m: _Model) -> Iterator[Finding]:
    conventions = m.model.ontology.conventions if m.model.ontology else None
    labels = [t.label for t in m.model.node_types]
    wanted = conventions.number if conventions and conventions.number else None
    if wanted is None and labels:
        plural = sum(_is_plural(label) for label in labels)
        wanted = "plural" if plural > len(labels) - plural else "singular"
    for label in labels:
        if _is_plural(label) != (wanted == "plural"):
            yield Finding(
                "naming_number",
                "info",
                _node(label),
                f"type label {label!r} isn't {wanted}",
                f"name types in the {wanted}",
            )


def _properties(m: _Model) -> list[tuple[str, str, Property]]:
    """Every declared property: its name, its element and itself."""
    found = [(p.name, f"property:{t.label}.{p.name}", p) for t in m.model.node_types for p in t.properties]
    found += [
        (p.name, f"relation_property:{r.label}.{p.name}", p) for r in m.model.relation_types for p in r.properties
    ]
    return found


def _meta_word_in_name(m: _Model) -> Iterator[Finding]:
    named = [(t.label, _node(t.label)) for t in m.model.node_types]
    named += [(r.label, f"relation:{r.label}") for r in m.model.relation_types]
    named += [(name, element) for name, element, _ in _properties(m)]
    for name, element in named:
        meta = next((word for word in _words(name) if word in META_WORDS), None)
        if meta is not None:
            yield Finding(
                "meta_word_in_name",
                "warning",
                element,
                f"{name!r} contains the word {meta!r}, which names the modeling construct, not the thing",
                f"drop {meta!r} from the name",
            )


def _sibling_naming(m: _Model) -> Iterator[Finding]:
    for parent, subtypes in m.subtypes.items():
        parent_words = _words(parent)
        repeating = [s for s in subtypes if _contains(_words(s), parent_words)]
        others = [s for s in subtypes if s not in repeating]
        if repeating and others:
            yield Finding(
                "sibling_naming",
                "info",
                _node(parent),
                f"subtypes of {parent!r} name it inconsistently: {repeating} repeat {parent!r}, {others} don't",
                "name the siblings the same way",
            )


def _contains(words: list[str], part: list[str]) -> bool:
    return any(words[i : i + len(part)] == part for i in range(len(words) - len(part) + 1))


# Documentation


def _missing_description(m: _Model) -> Iterator[Finding]:
    described: list[tuple[str, str]] = []
    for node in m.model.node_types:
        described.append((_node(node.label), node.description))
        for prop in node.properties:
            inherited = m.model.inherited(node.label, prop.name)
            # A redeclaration that says nothing new keeps the description it inherits (4.15).
            text = prop.description or (inherited.description if inherited else "")
            described.append((f"property:{node.label}.{prop.name}", text))
    for relation in m.model.relation_types:
        described.append((f"relation:{relation.label}", relation.description))
        described += [(f"relation_property:{relation.label}.{p.name}", p.description) for p in relation.properties]
    for element, description in described:
        if not description:
            yield Finding(
                "missing_description",
                "warning",
                element,
                f"{element.split(':', 1)[1]!r} has no description",
                "say what it covers: people and agents read it",
            )


def _no_competency_questions(m: _Model) -> Iterator[Finding]:
    if m.model.ontology is None or not m.model.ontology.competency_questions:
        yield Finding(
            "no_competency_questions",
            "info",
            "ontology",
            "the model lists no competency questions",
            "add `ontology.competency_questions`: the questions the model must answer",
        )


def _inverse_not_declared(m: _Model) -> Iterator[Finding]:
    relations = m.model.relation_types
    for index, a in enumerate(relations):
        for b in relations[index + 1 :]:
            swapped = a.start_labels and a.end_labels and set(a.start_labels) != set(a.end_labels)
            swapped = swapped and (set(a.start_labels), set(a.end_labels)) == (set(b.end_labels), set(b.start_labels))
            if swapped and m.model.inverse(a.label) != b.label and m.model.inverse(b.label) != a.label:
                yield Finding(
                    "inverse_not_declared",
                    "info",
                    f"relation:{a.label}",
                    f"{a.label!r} and {b.label!r} connect the same types in opposite directions",
                    f"if one is the other read backwards, declare `inverse_of: {b.label}` on {a.label!r}",
                )


_CHECKS: tuple[Callable[[_Model], Iterator[Finding]], ...] = (
    _single_subtype,
    _many_subtypes,
    _redundant_endpoint,
    _endpoint_covers_all_subtypes,
    _property_on_every_subtype,
    _empty_subtype,
    _plural_of_parent,
    _naming_case,
    _naming_number,
    _meta_word_in_name,
    _sibling_naming,
    _missing_description,
    _no_competency_questions,
    _unused_type,
    _inverse_not_declared,
)

#: Every code lint_model() reports, in the order of the contract's table.
LINT_CODES: tuple[str, ...] = tuple(check.__name__.lstrip("_") for check in _CHECKS)
