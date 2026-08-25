"""Tests for the runner that drives a whole eval batch.

The runner owns the *pipeline* loop -- inject, reconcile, retrieve -- because
deepeval knows nothing about those stages; they must all happen before an
actual_output exists to score. deepeval owns the *scoring* loop underneath.

Scoring is LLM-backed and gated. What is tested here without a key is the
sequencing the runner is responsible for.
"""

import pytest
from context_graph_eval.convert.longmemeval import to_golden
from context_graph_eval.runner import RunPlan, run_batch

from actions_graph import ActionsGraph


def _record(question_id: str, *, answer: str = "A beagle.", fact: str = "I adopted a beagle named Max"):
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "What breed is the dog?",
        "answer": answer,
        "question_date": "2023/06/15 (Thu) 09:12",
        "haystack_session_ids": [f"{question_id}-s1"],
        "haystack_dates": ["2023/05/20 (Sat) 14:03"],
        "haystack_sessions": [[{"role": "user", "content": fact, "has_answer": True}]],
        "answer_session_ids": [f"{question_id}-s1"],
    }


class _StubLLM:
    """Answers by echoing whatever the graph returned."""

    def __init__(self):
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.calls % 2 == 1:
            return "MATCH (a:Action) RETURN a.properties AS props"
        return "A beagle."


async def test_a_run_scores_every_question_in_the_corpus(eval_graph: ActionsGraph):
    goldens = [to_golden(_record("q1")), to_golden(_record("q2"))]

    report = await run_batch(
        goldens,
        records=[_record("q1"), _record("q2")],
        graph=eval_graph,
        llm=_StubLLM(),
        plan=RunPlan(reconcile=False, judge=None),
    )

    assert report.by_tier[1].questions == 2


async def test_fixtures_are_injected_before_retrieval_runs(eval_graph: ActionsGraph):
    """Ordering is the runner's whole job: retrieving before injection would
    query an empty graph and score every question as a miss."""
    await run_batch(
        [to_golden(_record("q1"))],
        records=[_record("q1")],
        graph=eval_graph,
        llm=_StubLLM(),
        plan=RunPlan(reconcile=False, judge=None),
    )

    assert eval_graph.get_session("q1-s1") is not None


async def test_retrieval_payload_is_measured_even_without_a_judge(eval_graph: ActionsGraph):
    """Efficiency is deterministic (#304), so it must be reported whether or not
    an LLM judge ran."""
    report = await run_batch(
        [to_golden(_record("q1"))],
        records=[_record("q1")],
        graph=eval_graph,
        llm=_StubLLM(),
        plan=RunPlan(reconcile=False, judge=None),
    )

    assert report.scored[0].efficiency_tokens > 0


async def test_a_run_reports_which_questions_it_scored(eval_graph: ActionsGraph):
    report = await run_batch(
        [to_golden(_record("q1")), to_golden(_record("q2"))],
        records=[_record("q1"), _record("q2")],
        graph=eval_graph,
        llm=_StubLLM(),
        plan=RunPlan(reconcile=False, judge=None),
    )

    assert {s.name for s in report.scored} == {"q1", "q2"}


async def test_an_empty_corpus_runs_without_error(eval_graph: ActionsGraph):
    report = await run_batch(
        [],
        records=[],
        graph=eval_graph,
        llm=_StubLLM(),
        plan=RunPlan(reconcile=False, judge=None),
    )

    assert report.by_tier == {}


async def test_a_question_whose_retrieval_fails_is_still_reported(eval_graph: ActionsGraph):
    """A batch must report a miss rather than losing the question: a coverage
    rate computed over a silently shortened corpus is wrong, not just noisy."""

    class BrokenLLM:
        async def complete(self, prompt: str) -> str:
            raise RuntimeError("model unavailable")

    report = await run_batch(
        [to_golden(_record("q1"))],
        records=[_record("q1")],
        graph=eval_graph,
        llm=BrokenLLM(),
        plan=RunPlan(reconcile=False, judge=None),
    )

    assert report.by_tier[1].questions == 1
    assert report.by_tier[1].covered == 0


def test_a_run_refuses_to_upload_to_a_third_party(monkeypatch):
    """#302 kept the corpus out of a vendor cloud on the project's own
    owned-IP grounds. A stray CONFIDENT_API_KEY would send results there, so
    the runner refuses rather than silently exporting."""
    from context_graph_eval.runner import check_offline

    monkeypatch.setenv("CONFIDENT_API_KEY", "sk-whatever")

    with pytest.raises(RuntimeError, match="CONFIDENT_API_KEY"):
        check_offline()
