"""LlmRecommendationStrategy: grow a model from a sample of the corpus it describes.

One derivation run takes the current model N and a delta of new sessions and
returns a candidate N+1 (map #431). The stages follow the derivation contract
(#353, revised by #366 and run in #372), with the aggregation rules of #434:

    propose      (LLM x K)  K batches of whole sessions, each told what N already
                            has and asked only for what it is missing
    consolidate  (LLM)      merges the K proposals, drops proposals that mean an
                            existing label, and may merge synonyms in N; the
                            only stage allowed to rename
    observe      (Observer) one permissive extraction pass over every type and
                            relation of the candidate, any endpoint allowed --
                            except that a relation into value types keeps that
                            tail, or it never fires into a value (#386)
    prune        (LLM)      sets each added relation's endpoints from the pairs
                            it was observed on, each added type's identity, and
                            drops additions with no real instances
    aggregate               sums observation counts into N's, retires to the pool
                            what falls below the share threshold or past the
                            active-type cap, and validates

A run never splits a type, never changes an existing relation's endpoints, and
never merges away, renames or retires a protected label: the fixed core, the
catch-alls, and anything pinned by a user-supplied schema.

hygm takes no LLM or extractor dependency: the caller supplies both.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from ..labels import CATCH_ALL_LABELS, CORE_LABELS, VALUE_LABELS
from ..types import IDENTITIES, HygmModel, NodeType, RelationType
from ..validation import PERSON_LABEL, USER_LABEL, validate_model

#: ``llm(system, prompt, schema) -> reply``: one call returning a mapping that matches the JSON schema.
Llm = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]


class Observer(Protocol):
    """Runs an extractor permissively over a sample and reports what it found.

    Implemented by the extracting package (unstructured2graph with GLiNER2), so
    hygm never depends on an extractor. The sample is passed through untouched.
    """

    def observe(self, model: HygmModel, sample: Sequence[Any]) -> Mapping[str, Any]:
        """Observation tables for `model` over `sample`, shaped::

        {"relations": {name: {"edges": int, "pairs": [{"head", "tail", "count", "examples"}]}},
         "types": {label: {"mentions": int, "sessions": int, "distinct_texts": int,
                           "texts_recurring_across_sessions": float, "capitalized_share": float,
                           "top_texts": [str]}}}
        """
        ...


@dataclass(frozen=True)
class DerivationLimits:
    """The knobs of one run.

    Attributes:
        batches: How many propose calls the delta is dealt across.
        batch_chars: The most conversation text one propose call reads; whole
            sessions only, the longest dropped first.
        max_node_types: Active node types allowed beside the protected ones (#434).
        max_relation_types: Active relation types allowed.
        retire_share: A type whose share of all mentions observed over every
            run (a relation: of all edges) falls below this retires to the pool.
            Additions of the run itself are exempt: they have only one run's
            evidence, and prune has just seen real instances of them.
        retire_min_total: Shares mean nothing on little evidence -- two
            sessions about code would retire Location -- so the threshold
            applies only once this many mentions (a relation: edges) have been
            observed over every run. The cap applies regardless.
    """

    batches: int = 4
    batch_chars: int = 120_000
    max_node_types: int = 40
    max_relation_types: int = 60
    retire_share: float = 0.002
    retire_min_total: int = 2000


@dataclass(frozen=True)
class Change:
    """One line of a run's changelog: add, merge, retire, restore (from the pool) or a vetoed merge."""

    op: Literal["add", "merge", "retire", "restore", "veto"]
    kind: Literal["node", "relation"]
    label: str
    into: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class Derivation:
    """A candidate model and what a gate or a report needs about how it was made.

    Attributes:
        model: The candidate: N with this run's merges, additions and retirements.
        pool: Retired types and relations, with their definitions, so they can return.
        counts: Observation counts summed over every run, ``{"nodes": {...}, "relations": {...}}``,
            for active and pooled labels alike.
        changelog: What this run changed, in order.
        unanchored_values: Value types no relation points into; reporting only,
            since such a type is never extracted (#366).
        observation: The observer's tables for this run.
        llm_calls: How many LLM calls the run made.
    """

    model: HygmModel
    pool: HygmModel
    counts: dict[str, dict[str, int]]
    changelog: tuple[Change, ...]
    unanchored_values: tuple[str, ...] = ()
    observation: Mapping[str, Any] = field(default_factory=dict)
    llm_calls: int = 0


class DerivationError(ValueError):
    """A run could not produce a valid candidate; `problems` lists why."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = tuple(problems)


SYSTEM = (
    "You design graph ontologies for an assistant's long-term memory. You answer only with the requested JSON. "
    "The ontology drives a local span extractor (GLiNER2) that reads each conversation turn and emits typed "
    "entities and directed, typed relations between them."
)

_RULES = """- entity_types: CamelCase label, one-line description. The extractor DOES read entity descriptions.
- relations: snake_case name. The extractor sees ONLY THE NAME, never a description, so the name alone must say what the relation means and which way it points. Give intended_head/intended_tail labels (existing or proposed) as documentation; the real endpoints are set later from observation. Relations are binary and directed; there are no symmetric or inverse relations.
- Value facts: when the conversations state a fact about the user as a value (a time, a duration, an amount, a count, a price, a date, a time window: "my best 5K is 25:50", "I spent $400 on it"), propose a relation into the matching value type ({values}). A value type nothing points into is never extracted.
- No modal or tense variants: one relation per kind of fact. Do not propose plans_to_X, wants_to_X, will_X, used_to_X or considering_X beside X; the fact's time is recorded separately.
- Prefer the fewest types and relations that cover what the user says; every extra label costs the extractor recall on the others. Do not model the assistant, the conversation itself, or generic advice the assistant gives."""

PROPOSE_PROMPT = """Below are {n} whole conversations between a user and an AI assistant. The user's memory graph is extracted under this model; everything in it is already extracted:
{model}

Propose only what this model is missing to remember what these conversations say about the user and their world:
{rules}

Types retired earlier for lack of evidence (re-propose one only if these conversations clearly need it):
{pool}

Conversations:
{sessions}"""

CONSOLIDATE_PROMPT = """{k} independent proposals of additions to one user's memory-graph model follow, each made from a different sample of their conversations. The current model:
{model}

Labels that may never be merged away: {protected}

Merge the proposals into one set of additions:
- Keep every distinct concept any proposal has: a domain one sample lacked is still in the corpus.
- Merge synonyms among the proposals into one label or name, picking the clearest; renaming is allowed here and nowhere later.
- A proposal that means the same as a label already in the model is not added; drop it.
- You may also merge two labels already in the model that are synonyms ("from" becomes an alias of "into"), but never merge away a label listed above.
{rules}

Proposals:
{proposals}"""

PRUNE_PROMPT = """These additions were proposed to a user's memory-graph model:
{additions}

A permissive extraction pass (every relation allowed between every pair of types) ran over {n} of the user's sessions with the current model plus these additions. Observation tables follow: for each relation, the (head type -> tail type) pairs it actually fired on, with counts and example edges; for each entity type, mention statistics.

Prune the additions only; the existing model is not yours to change:
- For each added relation you keep, set head and tail to the endpoint types it should connect, chosen ONLY from pairs it was observed on. Drop pairs that are extraction noise.
- Drop every added relation and type with zero observed instances, and any added relation whose observed edges are mostly noise.
- A relation whose head or tail includes User must also include Person there (a mention typed User may be a third party, re-typed to Person).
- For each added type you keep set identity: "global" (one node per distinct name across all sessions: named things that recur, like people, places or tools), "chunk" (one node per name per session: generic nouns whose same wording in two sessions is not the same thing), or "span" (one node per mention: values). Judge from the statistics, not the label.
- You may not rename anything or add any type or relation.

Observation:
{tables}"""

_ADDITIONS = {
    "entity_types": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
            "required": ["label", "description"],
        },
    },
    "relations": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "intended_head": {"type": "array", "items": {"type": "string"}},
                "intended_tail": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "intended_head", "intended_tail"],
        },
    },
}
PROPOSE_SCHEMA = {"type": "object", "properties": _ADDITIONS, "required": ["entity_types", "relations"]}
CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        **_ADDITIONS,
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["node", "relation"]},
                    "from": {"type": "string"},
                    "into": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["kind", "from", "into", "reason"],
            },
        },
    },
    "required": ["entity_types", "relations", "merges"],
}
PRUNE_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "head": {"type": "array", "items": {"type": "string"}},
                    "tail": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["name", "head", "tail", "reason"],
            },
        },
        "entity_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "identity": {"type": "string", "enum": list(IDENTITIES)},
                    "reason": {"type": "string"},
                },
                "required": ["label", "identity", "reason"],
            },
        },
    },
    "required": ["relations", "entity_types"],
}

_TYPE_LABEL = re.compile(r"^[A-Z][A-Za-z0-9]{1,40}$")
_RELATION_NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


class LlmRecommendationStrategy:
    """Derives the next version of a model from a delta of new sessions.

    Args:
        llm: ``llm(system, prompt, schema) -> reply``; hygm takes no LLM SDK dependency.
        observer: The permissive extraction pass the derived endpoints come from.
        limits: Batching, the active-type cap and the retirement threshold.
    """

    def __init__(self, llm: Llm, observer: Observer, limits: DerivationLimits | None = None) -> None:
        self.llm = llm
        self.observer = observer
        self.limits = limits or DerivationLimits()

    def derive(
        self,
        current: HygmModel,
        *,
        conversations: Sequence[str],
        observe_sample: Sequence[Any],
        pinned: Sequence[str] = (),
        counts: Mapping[str, Mapping[str, int]] | None = None,
        pool: HygmModel | None = None,
        seed: int = 0,
    ) -> Derivation:
        """One derivation run.

        Args:
            current: Model N, the one sessions are extracted under now.
            conversations: The delta's sessions as text, one per session, for
                the propose calls to read.
            observe_sample: What the observer runs over (e.g. the same sessions
                as documents); passed through untouched.
            pinned: Labels a user-supplied schema fixed: never merged away or retired.
            counts: Observation counts carried from earlier runs.
            pool: Types and relations retired earlier.
            seed: Shuffles which sessions share a propose batch; runs with
                different seeds are independent candidates for the gate.

        Returns:
            The candidate with its pool, counts and changelog.

        Raises:
            DerivationError: if there is nothing to derive from, or the
                candidate fails validate_model() or loses a core type.
        """
        if not conversations:
            raise DerivationError(["no conversations to derive from"])
        protected = frozenset(CORE_LABELS) | frozenset(CATCH_ALL_LABELS) | frozenset(pinned)
        pool = pool or HygmModel(node_types=())
        rules = _RULES.format(values=", ".join(VALUE_LABELS))
        has_pool = bool(pool.node_types or pool.relation_types)

        proposals = [
            self.llm(
                SYSTEM,
                PROPOSE_PROMPT.format(
                    n=len(batch),
                    model=_describe(current),
                    rules=rules,
                    pool=_describe(pool) if has_pool else "(none)",
                    sessions="\n\n".join(batch),
                ),
                PROPOSE_SCHEMA,
            )
            for batch in _batches(conversations, self.limits, seed)
        ]
        consolidated = self.llm(
            SYSTEM,
            CONSOLIDATE_PROMPT.format(
                k=len(proposals),
                model=_describe(current),
                protected=", ".join(sorted(protected)),
                rules=rules,
                proposals=json.dumps(proposals, indent=1),
            ),
            CONSOLIDATE_SCHEMA,
        )

        changelog: list[Change] = []
        merged, renames = _merge(current, consolidated.get("merges", []), protected, changelog)
        added_nodes, added_relations = _additions(merged, pool, consolidated, changelog)

        candidate = HygmModel(
            node_types=merged.node_types + added_nodes, relation_types=merged.relation_types + added_relations
        )
        permissive = replace(
            candidate,
            relation_types=tuple(
                replace(r, start_labels=(), end_labels=r.end_labels if _into_values(r.end_labels) else ())
                for r in candidate.relation_types
            ),
        )
        observation = self.observer.observe(permissive, observe_sample)

        pruned = self.llm(
            SYSTEM,
            PRUNE_PROMPT.format(
                additions=_describe(HygmModel(node_types=added_nodes, relation_types=added_relations)),
                n=len(observe_sample),
                tables=json.dumps(observation, indent=1),
            ),
            PRUNE_SCHEMA,
        )
        candidate = _apply_prune(merged, added_nodes, added_relations, pruned, observation, changelog)
        counts = _counts(counts, renames, observation)
        added = frozenset(change.label for change in changelog if change.op in {"add", "restore"})
        candidate, pool = _retire(candidate, pool, counts, protected, added, self.limits, changelog)

        problems = [issue.message for issue in validate_model(candidate).critical_issues]
        problems += [f"core type {label!r} is missing" for label in CORE_LABELS if label not in candidate.node_labels()]
        if problems:
            raise DerivationError(problems)
        tails = {label for r in candidate.relation_types for label in candidate.endpoint_labels(r, "end")}
        return Derivation(
            model=candidate,
            pool=pool,
            counts=counts,
            changelog=tuple(changelog),
            unanchored_values=tuple(label for label in VALUE_LABELS if label not in tails),
            observation=observation,
            llm_calls=len(proposals) + 2,
        )


def _batches(conversations: Sequence[str], limits: DerivationLimits, seed: int) -> list[list[str]]:
    """The delta shuffled by `seed`, dealt round-robin into batches, each cut to the character budget."""
    order = list(conversations)
    random.Random(seed).shuffle(order)
    batches: list[list[str]] = [[] for _ in range(min(limits.batches, len(order)))]
    for index, text in enumerate(order):
        batches[index % len(batches)].append(text)
    for batch in batches:  # whole sessions only: drop the longest until under budget
        while len(batch) > 1 and sum(len(text) for text in batch) > limits.batch_chars:
            batch.remove(max(batch, key=len))
    return batches


def _describe(model: HygmModel) -> str:
    nodes = [f"- {t.label}: {t.description}" for t in model.node_types]
    relations = [
        f"- {r.label}: {'|'.join(r.start_labels) or 'any'} -> {'|'.join(r.end_labels) or 'any'}"
        for r in model.relation_types
    ]
    return "entity types:\n" + "\n".join(nodes or ["(none)"]) + "\nrelations:\n" + "\n".join(relations or ["(none)"])


def _merge(
    model: HygmModel, merges: Sequence[Mapping[str, Any]], protected: frozenset[str], changelog: list[Change]
) -> tuple[HygmModel, dict[str, dict[str, str]]]:
    """`model` with consolidate's merges applied; vetoed ones are logged, not applied.

    The survivor keeps its own definition. Relations pointing at a merged node
    type point at the survivor; a merged relation's endpoints join the survivor's.

    Returns:
        The merged model and the renames applied, ``{"nodes": {from: into}, "relations": {...}}``.
    """
    renames: dict[str, dict[str, str]] = {"nodes": {}, "relations": {}}
    nodes = {t.label: t for t in model.node_types}
    relations = {r.label: r for r in model.relation_types}
    for merge in merges:
        kind, source, target = merge.get("kind"), merge.get("from", ""), merge.get("into", "")
        if kind == "node":
            veto = _node_merge_veto(nodes, source, target, protected)
            if not veto:
                del nodes[source]
                relations = {label: _rename_endpoint(r, source, target) for label, r in relations.items()}
        elif kind == "relation":
            veto = _relation_merge_veto(relations, source, target, protected)
            if not veto:
                gone, survivor = relations.pop(source), relations[target]
                relations[target] = replace(
                    survivor,
                    start_labels=_union(survivor.start_labels, gone.start_labels),
                    end_labels=_union(survivor.end_labels, gone.end_labels),
                )
        else:
            continue
        if veto:
            changelog.append(Change("veto", kind, source, target, veto))
            continue
        renames["nodes" if kind == "node" else "relations"][source] = target
        changelog.append(Change("merge", kind, source, target, merge.get("reason", "")))
    return HygmModel(node_types=tuple(nodes.values()), relation_types=tuple(relations.values())), renames


def _node_merge_veto(nodes: Mapping[str, NodeType], source: str, target: str, protected: frozenset[str]) -> str:
    if source in protected:
        return f"{source} is protected"
    if target in CATCH_ALL_LABELS:
        return f"{target} is a catch-all; merging into it would hide {source} rather than name it"
    if source not in nodes or target not in nodes or source == target:
        return "unknown or identical types"
    if nodes[source].identity != nodes[target].identity:
        return f"identities differ ({nodes[source].identity} vs {nodes[target].identity})"
    return ""


def _relation_merge_veto(
    relations: Mapping[str, RelationType], source: str, target: str, protected: frozenset[str]
) -> str:
    if source in protected:
        return f"{source} is protected"
    if source not in relations or target not in relations or source == target:
        return "unknown or identical relations"
    a, b = relations[source], relations[target]
    for side, x, y in (("start", a.start_labels, b.start_labels), ("end", a.end_labels, b.end_labels)):
        if x and y and not set(x) & set(y):
            return f"incompatible {side} labels ({'|'.join(x)} vs {'|'.join(y)})"
    return ""


def _rename_endpoint(relation: RelationType, source: str, target: str) -> RelationType:
    def renamed(labels: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(target if label == source else label for label in labels))

    return replace(relation, start_labels=renamed(relation.start_labels), end_labels=renamed(relation.end_labels))


def _union(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    # An empty side means "any label", so a union with one stays unconstrained.
    return tuple(dict.fromkeys(a + b)) if a and b else ()


def _additions(
    model: HygmModel, pool: HygmModel, consolidated: Mapping[str, Any], changelog: list[Change]
) -> tuple[tuple[NodeType, ...], tuple[RelationType, ...]]:
    """Consolidate's proposals that are new to `model`, as types with open endpoints.

    A relation intended only into value types keeps that tail, so observe can
    see it fire into a value. A proposal naming a pooled node type restores it
    with its old definition.
    """
    pooled = {t.label: t for t in pool.node_types}
    nodes: dict[str, NodeType] = {}
    for item in consolidated.get("entity_types", []):
        label = item.get("label", "")
        if label in model.node_labels() or label in nodes or not _TYPE_LABEL.match(label):
            continue
        if label in pooled:
            nodes[label] = pooled[label]
            changelog.append(Change("restore", "node", label))
        else:
            nodes[label] = NodeType(label, item.get("description", ""))
    relations: dict[str, RelationType] = {}
    for item in consolidated.get("relations", []):
        name = item.get("name", "")
        if name in model.relation_labels() or name in relations or not _RELATION_NAME.match(name):
            continue
        tail = tuple(dict.fromkeys(item.get("intended_tail", [])))
        relations[name] = RelationType(name, end_labels=tail if _into_values(tail) else ())
    return tuple(nodes.values()), tuple(relations.values())


def _into_values(labels: Sequence[str]) -> bool:
    return bool(labels) and set(labels) <= set(VALUE_LABELS)


def _apply_prune(
    merged: HygmModel,
    added_nodes: tuple[NodeType, ...],
    added_relations: tuple[RelationType, ...],
    pruned: Mapping[str, Any],
    observation: Mapping[str, Any],
    changelog: list[Change],
) -> HygmModel:
    """`merged` plus the additions prune kept: types with an identity and real mentions, relations on an observed pair."""
    identity = {t.get("label"): t.get("identity") for t in pruned.get("entity_types", [])}
    observed_types = observation.get("types", {})
    kept_nodes = []
    for t in added_nodes:
        restored = any(c.op == "restore" and c.label == t.label for c in changelog)
        chosen = t.identity if restored else identity.get(t.label)
        if chosen in IDENTITIES and observed_types.get(t.label, {}).get("mentions", 0) > 0:
            kept_nodes.append(replace(t, identity=chosen))
            if not restored:
                changelog.append(Change("add", "node", t.label))
    declared = set(merged.node_labels()) | {t.label for t in kept_nodes}

    proposed = {r.label for r in added_relations}
    observed_relations = observation.get("relations", {})
    kept_relations = []
    for item in pruned.get("relations", []):
        name = item.get("name")
        if name not in proposed or any(r.label == name for r in kept_relations):
            continue
        pairs = {(p.get("head"), p.get("tail")) for p in observed_relations.get(name, {}).get("pairs", [])}
        head = _with_person(tuple(label for label in dict.fromkeys(item.get("head", [])) if label in declared))
        tail = _with_person(tuple(label for label in dict.fromkeys(item.get("tail", [])) if label in declared))
        if not any((h, t) in pairs for h in head for t in tail):
            continue
        kept_relations.append(RelationType(name, item.get("reason", ""), head, tail))
        changelog.append(Change("add", "relation", name))
    return HygmModel(
        node_types=merged.node_types + tuple(kept_nodes),
        relation_types=merged.relation_types + tuple(kept_relations),
    )


def _with_person(labels: tuple[str, ...]) -> tuple[str, ...]:
    # A mention typed User may be a third party re-typed Person, so a side that takes one takes both (#358).
    return (*labels, PERSON_LABEL) if USER_LABEL in labels and PERSON_LABEL not in labels else labels


def _counts(
    carried: Mapping[str, Mapping[str, int]] | None,
    renames: Mapping[str, Mapping[str, str]],
    observation: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Earlier runs' counts, merges summed into their survivors, plus this run's observation."""
    total: dict[str, dict[str, int]] = {"nodes": {}, "relations": {}}
    for kind in ("nodes", "relations"):
        for label, count in (carried or {}).get(kind, {}).items():
            survivor = renames[kind].get(label, label)
            total[kind][survivor] = total[kind].get(survivor, 0) + int(count)
    for label, stats in observation.get("types", {}).items():
        total["nodes"][label] = total["nodes"].get(label, 0) + int(stats.get("mentions", 0))
    for name, stats in observation.get("relations", {}).items():
        total["relations"][name] = total["relations"].get(name, 0) + int(stats.get("edges", 0))
    return total


def _retire(
    model: HygmModel,
    pool: HygmModel,
    counts: Mapping[str, Mapping[str, int]],
    protected: frozenset[str],
    added: frozenset[str],
    limits: DerivationLimits,
    changelog: list[Change],
) -> tuple[HygmModel, HygmModel]:
    """Pool what falls below the share threshold, then the lowest-share labels until the cap holds.

    Protected labels never retire and don't count against the cap. The run's
    own additions count against the cap but are exempt from the threshold. A
    relation loses a retired type from its endpoints, and retires itself when
    a side it constrained empties.
    """

    def share(kind: str, label: str) -> float:
        return counts[kind].get(label, 0) / (sum(counts[kind].values()) or 1)

    def retiring(kind: str, labels: Sequence[str], cap: int) -> list[str]:
        ranked = sorted((label for label in labels if label not in protected), key=lambda label: share(kind, label))
        over = max(0, len(ranked) - cap)
        enough = sum(counts[kind].values()) >= limits.retire_min_total
        return [
            label
            for rank, label in enumerate(ranked)
            if rank < over or (enough and label not in added and share(kind, label) < limits.retire_share)
        ]

    nodes = {t.label: t for t in model.node_types}
    pooled_nodes = {t.label: t for t in pool.node_types if t.label not in nodes}
    for label in retiring("nodes", list(nodes), limits.max_node_types):
        pooled_nodes[label] = nodes.pop(label)
        changelog.append(Change("retire", "node", label, reason=f"share {share('nodes', label):.4f}"))

    relations: dict[str, RelationType] = {}
    pooled_relations = {r.label: r for r in pool.relation_types}
    for relation in model.relation_types:
        start = tuple(label for label in relation.start_labels if label in nodes)
        end = tuple(label for label in relation.end_labels if label in nodes)
        if (relation.start_labels and not start) or (relation.end_labels and not end):
            pooled_relations[relation.label] = relation
            changelog.append(Change("retire", "relation", relation.label, reason="an endpoint type retired"))
        else:
            relations[relation.label] = replace(relation, start_labels=start, end_labels=end)
    for label in retiring("relations", list(relations), limits.max_relation_types):
        pooled_relations[label] = relations.pop(label)
        changelog.append(Change("retire", "relation", label, reason=f"share {share('relations', label):.4f}"))

    return (
        HygmModel(node_types=tuple(nodes.values()), relation_types=tuple(relations.values())),
        HygmModel(
            node_types=tuple(pooled_nodes.values()),
            relation_types=tuple(r for label, r in pooled_relations.items() if label not in relations),
        ),
    )
