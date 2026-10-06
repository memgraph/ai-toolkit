"""One derivation run, with the LLM and the observer stood in for."""

from dataclasses import replace

import pytest

from hygm import (
    Change,
    DerivationError,
    DerivationLimits,
    HygmModel,
    LlmRecommendationStrategy,
    NodeType,
    RelationType,
    default_model,
)
from hygm.strategies.llm import CONSOLIDATE_SCHEMA, PROPOSE_SCHEMA, PRUNE_SCHEMA, _batches


class ScriptedLlm:
    """Answers each stage from a script, keyed by the stage's schema, and records the prompts."""

    def __init__(self, *, consolidate, prune, propose=None):
        self.replies = {
            id(PROPOSE_SCHEMA): propose or {"entity_types": [], "relations": []},
            id(CONSOLIDATE_SCHEMA): {"entity_types": [], "relations": [], "merges": [], **consolidate},
            id(PRUNE_SCHEMA): {"entity_types": [], "relations": [], **prune},
        }
        self.prompts = []

    def __call__(self, system, prompt, schema):
        self.prompts.append(prompt)
        return self.replies[id(schema)]


class FixedObserver:
    def __init__(self, types=None, relations=None):
        self.tables = {"types": types or {}, "relations": relations or {}}
        self.models = []

    def observe(self, model, sample):
        self.models.append(model)
        return self.tables


def _observed(**mentions):
    return {label: {"mentions": n} for label, n in mentions.items()}


def _fired(name, *pairs, edges=None):
    return {
        name: {
            "edges": edges or sum(n for _, _, n in pairs),
            "pairs": [{"head": h, "tail": t, "count": n} for h, t, n in pairs],
        }
    }


def _derive(llm, observer, current=None, **kwargs):
    strategy = LlmRecommendationStrategy(llm, observer, kwargs.pop("limits", None))
    return strategy.derive(
        current or default_model(), conversations=["one session", "another"], observe_sample=["s1", "s2"], **kwargs
    )


LIBRARY = {"label": "Library", "description": "a code library"}
MAINTAINS = {"name": "maintains", "intended_head": ["User"], "intended_tail": ["Library"]}


def test_an_observed_addition_joins_the_model_with_its_observed_endpoints():
    llm = ScriptedLlm(
        consolidate={"entity_types": [LIBRARY, {"label": "lowercase", "description": "x"}], "relations": [MAINTAINS]},
        prune={
            "entity_types": [{"label": "Library", "identity": "global", "reason": ""}],
            "relations": [{"name": "maintains", "head": ["User"], "tail": ["Library"], "reason": "the user's code"}],
        },
    )
    observer = FixedObserver(types=_observed(Library=12), relations=_fired("maintains", ("User", "Library", 5)))

    derivation = _derive(llm, observer)

    assert derivation.model.node_type("Library") == NodeType("Library", "a code library", "global")
    maintains = next(r for r in derivation.model.relation_types if r.label == "maintains")
    assert (maintains.start_labels, maintains.end_labels) == (("User", "Person"), ("Library",))
    assert derivation.changelog == (Change("add", "node", "Library"), Change("add", "relation", "maintains"))
    assert derivation.counts["nodes"]["Library"] == 12
    assert derivation.llm_calls == 2 + 2  # two sessions make two propose batches, then consolidate and prune


def test_the_observer_sees_every_relation_unconstrained_but_value_tails():
    """A relation into a value type that may end anywhere never fires into a value (#386)."""
    best = {"name": "personal_best", "intended_head": ["User"], "intended_tail": ["Duration"]}
    llm = ScriptedLlm(consolidate={"relations": [MAINTAINS, best]}, prune={})
    observer = FixedObserver()

    _derive(llm, observer)

    seen = {r.label: r for r in observer.models[0].relation_types}
    assert observer.models[0].node_labels() == default_model().node_labels()
    assert all(not r.start_labels for r in seen.values())
    assert (seen["maintains"].end_labels, seen["works_for"].end_labels) == ((), ())
    assert (seen["personal_best"].end_labels, seen["lasted"].end_labels) == (("Duration",), ("Duration",))


def test_additions_without_real_instances_are_dropped():
    llm = ScriptedLlm(
        consolidate={"entity_types": [LIBRARY], "relations": [MAINTAINS]},
        prune={
            "entity_types": [{"label": "Library", "identity": "global", "reason": ""}],
            "relations": [{"name": "maintains", "head": ["User"], "tail": ["Library"], "reason": ""}],
        },
    )
    observer = FixedObserver(types=_observed(Library=0), relations=_fired("maintains", ("User", "Artifact", 4)))

    derivation = _derive(llm, observer)

    assert "Library" not in derivation.model.node_labels()
    assert "maintains" not in derivation.model.relation_labels()
    assert derivation.changelog == ()


def _with_frameworks():
    base = default_model()
    return HygmModel(
        node_types=(
            *base.node_types,
            NodeType("Framework", "a framework", "global"),
            NodeType("Library", "a library", "global"),
        ),
        relation_types=(
            *base.relation_types,
            RelationType("builds_with", "", ("User", "Person"), ("Framework",)),
        ),
    )


def test_a_merge_relabels_endpoints_and_sums_counts_into_the_survivor():
    llm = ScriptedLlm(
        consolidate={"merges": [{"kind": "node", "from": "Framework", "into": "Library", "reason": "synonyms"}]},
        prune={},
    )
    observer = FixedObserver(types=_observed(Library=4))

    derivation = _derive(
        llm, observer, current=_with_frameworks(), counts={"nodes": {"Framework": 5, "Library": 3}, "relations": {}}
    )

    assert "Framework" not in derivation.model.node_labels()
    builds_with = next(r for r in derivation.model.relation_types if r.label == "builds_with")
    assert builds_with.end_labels == ("Library",)
    assert derivation.counts["nodes"]["Library"] == 5 + 3 + 4
    assert Change("merge", "node", "Framework", "Library", "synonyms") in derivation.changelog


@pytest.mark.parametrize(
    ("merge", "why"),
    [
        ({"kind": "node", "from": "Person", "into": "Library"}, "protected"),
        ({"kind": "node", "from": "Framework", "into": "Topic"}, "catch-all"),
        ({"kind": "node", "from": "Framework", "into": "Duration"}, "identities differ"),
        ({"kind": "relation", "from": "builds_with", "into": "lives_in"}, "incompatible end labels"),
    ],
)
def test_a_merge_that_breaks_a_rule_is_vetoed_and_logged(merge, why):
    llm = ScriptedLlm(consolidate={"merges": [{**merge, "reason": ""}]}, prune={})

    derivation = _derive(llm, FixedObserver(), current=_with_frameworks())

    assert merge["from"] in derivation.model.node_labels() + derivation.model.relation_labels()
    (veto,) = derivation.changelog
    assert veto.op == "veto" and why in veto.reason


def test_a_type_below_the_share_threshold_retires_with_its_relations_and_can_return():
    llm = ScriptedLlm(consolidate={}, prune={})
    counts = {"nodes": {"User": 1000, "Framework": 1, "Library": 300}, "relations": {"builds_with": 1}}

    derivation = _derive(
        llm, FixedObserver(), current=_with_frameworks(), counts=counts, limits=DerivationLimits(retire_min_total=1000)
    )

    assert "Framework" not in derivation.model.node_labels()
    assert derivation.pool.node_type("Framework") == NodeType("Framework", "a framework", "global")
    assert "builds_with" in derivation.pool.relation_labels()
    assert Change("retire", "relation", "builds_with", reason="an endpoint type retired") in derivation.changelog

    back = ScriptedLlm(consolidate={"entity_types": [{"label": "Framework", "description": "reworded"}]}, prune={})
    restored = _derive(
        back,
        FixedObserver(types=_observed(Framework=50)),
        pool=derivation.pool,
        counts=derivation.counts,
        limits=DerivationLimits(retire_min_total=1000),
    )
    assert restored.model.node_type("Framework") == NodeType("Framework", "a framework", "global")
    assert Change("restore", "node", "Framework") in restored.changelog


def test_the_cap_retires_the_lowest_share_types_but_never_protected_ones():
    llm = ScriptedLlm(consolidate={}, prune={})
    counts = {
        "nodes": {"Framework": 40, "Library": 60, "Organization": 50, "Location": 50, "Event": 50},
        "relations": {},
    }

    derivation = _derive(
        llm,
        FixedObserver(),
        current=_with_frameworks(),
        counts=counts,
        pinned=["Event"],
        limits=DerivationLimits(max_node_types=3, retire_share=0),
    )

    active = set(derivation.model.node_labels())
    assert {"Library", "Organization", "Location"} <= active and "Framework" not in active
    assert {"User", "Topic", "Artifact", "Event"} <= active  # core, catch-alls and pinned sit outside the cap


def test_an_addition_is_exempt_from_the_share_threshold():
    llm = ScriptedLlm(
        consolidate={"entity_types": [LIBRARY]},
        prune={"entity_types": [{"label": "Library", "identity": "global", "reason": ""}]},
    )
    observer = FixedObserver(types=_observed(Library=1))

    derivation = _derive(llm, observer, counts={"nodes": {"User": 100_000}, "relations": {}})

    assert "Library" in derivation.model.node_labels()


def test_shares_wait_for_enough_evidence_before_retiring_anything():
    """Two sessions about code mention no place; that is no reason to drop Location."""
    derivation = _derive(ScriptedLlm(consolidate={}, prune={}), FixedObserver(types=_observed(User=40, Artifact=30)))

    assert derivation.model == default_model()
    assert derivation.changelog == ()


def test_value_types_nothing_points_into_are_reported():
    base = default_model()
    no_money = replace(base, relation_types=tuple(r for r in base.relation_types if "Money" not in r.end_labels))

    derivation = _derive(ScriptedLlm(consolidate={}, prune={}), FixedObserver(), current=no_money)

    assert derivation.unanchored_values == ("Money",)


def test_a_candidate_without_the_core_is_refused():
    base = default_model()
    no_duration = HygmModel(
        node_types=tuple(t for t in base.node_types if t.label != "Duration"),
        relation_types=tuple(r for r in base.relation_types if "Duration" not in r.end_labels),
    )

    with pytest.raises(DerivationError, match="core type 'Duration' is missing"):
        _derive(ScriptedLlm(consolidate={}, prune={}), FixedObserver(), current=no_duration)


def test_nothing_to_derive_from_is_refused():
    strategy = LlmRecommendationStrategy(ScriptedLlm(consolidate={}, prune={}), FixedObserver())

    with pytest.raises(DerivationError, match="no conversations"):
        strategy.derive(default_model(), conversations=[], observe_sample=[])


def test_batches_are_whole_sessions_dealt_by_seed_under_the_budget():
    sessions = [f"session {i} " + "x" * (10 * i) for i in range(6)]
    limits = DerivationLimits(batches=2, batch_chars=80)

    batches = _batches(sessions, limits, seed=1)

    assert len(batches) == 2
    assert all(sum(map(len, batch)) <= 80 or len(batch) == 1 for batch in batches)
    assert {text for batch in batches for text in batch} <= set(sessions)
    assert batches == _batches(sessions, limits, seed=1)
    assert batches != _batches(sessions, limits, seed=2)
