"""Tests for the gold-slice question.

One question, deliberately (#307): a fact that exists only inside a subagent.
Top-level recall is already covered by Tier 1, and the nested carrier is the one
with a demonstrated silent-failure mode -- #281 found get_session_actions() does
a single-hop HAS_ACTION match, so once subagent activity moved under (:Agent),
reconciliation would stop seeing it with no error at all.
"""

import re

from context_graph_eval.convert.longmemeval import DEFAULT_REVISION
from context_graph_eval.goldslice import (
    GOLD_SLICE_FACT,
    GOLD_SLICE_PROMPT,
    evidence_is_nested,
    evidence_is_planted,
    gold_slice_goldens,
)

from actions_graph import ActionsGraph, Agent, Message, MessageRole, Session


def test_there_is_exactly_one_gold_slice_question():
    """#307: one to start. Each is a real, billed Claude Code session, so the
    slice grows by carrier rather than by round number."""
    assert len(gold_slice_goldens()) == 1


def test_the_question_is_tier_two():
    """It is authored, not adopted, so it is scored apart from Tier 1 (#303)."""
    (golden,) = gold_slice_goldens()

    assert golden.additional_metadata["tier"] == 2


def test_the_golden_carries_its_own_planting_prompt():
    """#307 put the session prompt on the Golden rather than in a joined
    fixture file, so a question cannot survive while its planting script
    drifts."""
    (golden,) = gold_slice_goldens()

    assert golden.additional_metadata["session_prompt"] == GOLD_SLICE_PROMPT


def test_the_expected_answer_tracks_the_pinned_revision():
    """The fact is the corpus pin itself. If someone bumps the revision without
    updating this answer, the gold slice would fail for a bookkeeping reason and
    look like a recall regression."""
    (golden,) = gold_slice_goldens()

    assert DEFAULT_REVISION.startswith(golden.expected_output)


def test_the_prompt_never_states_the_fact_itself():
    """The whole point of the nested carrier: if the top-level prompt contained
    the answer, the fact would land in a top-level Action too, and recall would
    succeed even with nesting completely broken."""
    (golden,) = gold_slice_goldens()

    assert golden.expected_output not in GOLD_SLICE_PROMPT
    assert DEFAULT_REVISION not in GOLD_SLICE_PROMPT


def test_the_prompt_contains_no_literal_skill_path():
    """SkillGraphConnector scans metadata for anything resembling a resolvable
    SKILL.md path and records it as a skill read that never happened -- the open
    false-positive at #293. test-graph-model had to be written around this; so
    does the gold slice."""
    assert not re.search(r"SKILL\.md", GOLD_SLICE_PROMPT)


def test_the_prompt_forces_delegation_to_exactly_one_subagent():
    lowered = GOLD_SLICE_PROMPT.lower()

    assert "task tool" in lowered
    assert "exactly one" in lowered


def test_an_unplanted_fixture_is_detected(eval_graph: ActionsGraph):
    """The gold slice's fixture is planted by a real harness session, not
    injected like Tier 1's. Scoring the question without having driven that
    session measures a fact the graph never contained and reports a guaranteed
    zero as a recall failure -- so the run refuses instead."""
    assert evidence_is_planted(eval_graph) is False


def test_a_planted_fixture_is_detected(eval_graph: ActionsGraph):
    eval_graph.create_session(Session(session_id="s-planted"))
    eval_graph.record_action(
        Message(session_id="s-planted", role=MessageRole.ASSISTANT, content=f"the revision is {GOLD_SLICE_FACT}")
    )

    assert evidence_is_planted(eval_graph) is True


def test_evidence_only_in_a_top_level_action_is_not_nested(eval_graph: ActionsGraph):
    """A false pass this guards against: if the model declines to delegate, the
    fact lands top-level, recall succeeds trivially, and the question silently
    stops testing nesting at all."""
    eval_graph.create_session(Session(session_id="s-flat"))
    eval_graph.record_action(Message(session_id="s-flat", role=MessageRole.ASSISTANT, content="the answer is 98d7416"))

    assert evidence_is_nested(eval_graph, "s-flat", "98d7416") is False


def test_evidence_inside_a_subagent_is_nested(eval_graph: ActionsGraph):
    eval_graph.create_session(Session(session_id="s-nested"))
    eval_graph.start_agent(Agent(agent_id="a1", session_id="s-nested", agent_type="Explore"))
    eval_graph.record_action(
        Message(session_id="s-nested", role=MessageRole.ASSISTANT, content="the answer is 98d7416"),
        container_agent_id="a1",
    )

    assert evidence_is_nested(eval_graph, "s-nested", "98d7416") is True


def test_a_session_with_no_such_evidence_is_not_nested(eval_graph: ActionsGraph):
    eval_graph.create_session(Session(session_id="s-empty"))

    assert evidence_is_nested(eval_graph, "s-empty", "98d7416") is False
