"""Tests for the pipeline: when a kept graph is reused, resumed or rebuilt, and a whole run end to end.

The end-to-end tests run the real build -- injection, GLiNER2 reconciliation,
embedding in Memgraph -- against the eval instance, with a small benchmark
whose judge is plain code, so what is tested is the pipeline and not an LLM.
"""

from dataclasses import replace
from typing import Any

import pytest
from conftest import EVAL_MEMGRAPH_URL, ScriptedLLM
from context_graph_eval.answering import RecallConfig
from context_graph_eval.convert.longmemeval import SessionFixture, Turn
from context_graph_eval.pipeline import (
    BuildPlan,
    Connection,
    Manifest,
    RunSettings,
    build_inputs,
    decide,
    fingerprint,
    read_manifest,
    require_built,
    run,
)
from context_graph_eval.results import Row, load
from context_graph_eval.workloads import Workload
from deepeval.dataset import Golden

WORKLOAD = Workload(name="test", kind="dev", benchmark="longmemeval-s", description="", port=0, revision="r")
QUESTIONS = [Golden(input="Where did I move?", expected_output="Paris", name="q1")]


def _manifest(inputs, *, status="complete") -> Manifest:
    return Manifest(fingerprint=fingerprint(inputs), inputs=inputs, status=status, commit="abc1234")


def test_a_graph_built_from_the_same_inputs_is_reused():
    inputs = build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan())

    assert decide(_manifest(inputs), inputs).action == "reuse"


def test_an_interrupted_build_of_the_same_inputs_is_resumed():
    inputs = build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan())

    assert decide(_manifest(inputs, status="building"), inputs).action == "resume"


def test_a_graph_built_differently_is_rebuilt_and_the_reason_names_what_changed():
    built = build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan())
    wanted = build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan(ontology="learned"))

    decision = decide(_manifest(built), wanted)

    assert decision.action == "build"
    assert "ontology" in decision.reason


def test_a_change_to_the_code_that_shapes_the_graph_rebuilds_it():
    built = {**build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan()), "construction_code": "older"}

    decision = decide(_manifest(built), build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan()))

    assert (decision.action, decision.reason) == ("build", "construction_code changed since the last build")


def test_reuse_keeps_a_stale_graph_of_the_same_questions_but_never_one_of_other_questions():
    built = {**build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan()), "construction_code": "older"}
    wanted = build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan())
    other = build_inputs(WORKLOAD, None, [Golden(input="?", name="q2")], BuildPlan())

    assert decide(_manifest(built), wanted, reuse=True).action == "reuse"
    assert decide(_manifest(built), other, reuse=True).action == "build"


def test_rebuild_and_an_empty_instance_both_build():
    inputs = build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan())

    assert decide(_manifest(inputs), inputs, rebuild=True).action == "build"
    assert decide(None, inputs).action == "build"


def test_the_shard_and_the_session_cap_are_part_of_what_built_a_graph():
    capped = replace(WORKLOAD, max_sessions_per_question=10)

    assert fingerprint(build_inputs(WORKLOAD, 0, QUESTIONS, BuildPlan())) != fingerprint(
        build_inputs(WORKLOAD, 1, QUESTIONS, BuildPlan())
    )
    assert fingerprint(build_inputs(WORKLOAD, None, QUESTIONS, BuildPlan())) != fingerprint(
        build_inputs(capped, None, QUESTIONS, BuildPlan())
    )


class _Benchmark:
    """Two users who each moved somewhere; a question is right when the answer names the place."""

    judge_model = "code"

    def __init__(self):
        self.fixture_calls = 0

    def questions(self, workload):
        return [
            Golden(input="Where did I move?", expected_output="Paris", name="u1"),
            Golden(input="Where did I move?", expected_output="Lisbon", name="u2"),
        ]

    def fixtures(self, workload, questions):
        self.fixture_calls += 1
        city = {"u1": "Paris", "u2": "Lisbon"}
        return [
            SessionFixture(
                session_id=f"{q.name}--s1",
                date="2023/05/20 (Sat) 14:03",
                turns=[Turn(role="user", content=f"I moved to {city[q.name]} last week.")],
                holds_evidence=True,
                user_id=q.name,
            )
            for q in questions
        ]

    def user_id(self, question):
        return question.name

    def today(self, question):
        return None

    async def judge(self, questions, retrieved):
        return [
            Row(
                name=q.name,
                user_id=q.name,
                question_type="single-session-user",
                score=float(q.expected_output in "\n".join(r.retrieval_context)),
            )
            for q, r in zip(questions, retrieved, strict=True)
        ]

    def headline(self, rows):
        return sum(row.score for row in rows) / len(rows)

    def notes(self, rows):
        return []


@pytest.fixture
def connection(eval_graph):
    from sessions_graph.embeddings import EmbeddingUnavailableError, check_available

    try:
        check_available(eval_graph.db)
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"eval Memgraph can't embed (needs MAGE): {exc}")
    return Connection(db=eval_graph.db, url=EVAL_MEMGRAPH_URL)


def _settings(**overrides) -> RunSettings:
    fields: dict[str, Any] = {
        "label": "test",
        "agent_model": "scripted",
        "recall": RecallConfig(lanes=("turns", "text")),
    }
    fields.update(overrides)
    return RunSettings(**fields)


async def test_a_run_builds_the_graph_asks_it_and_records_the_build(connection):
    result = await run(WORKLOAD, _Benchmark(), connection=connection, llm=ScriptedLLM("ok"), settings=_settings())

    assert result is not None
    assert [(row.name, row.score) for row in result.rows] == [("u1", 1.0), ("u2", 1.0)]
    manifest = read_manifest(connection.db)
    assert manifest is not None and manifest.status == "complete"
    assert result.meta.build["fingerprint"] == manifest.fingerprint
    assert result.meta.build["current"] is True


async def test_a_second_run_reuses_the_kept_graph_and_rebuild_starts_over(connection):
    await run(WORKLOAD, _Benchmark(), connection=connection, llm=ScriptedLLM("ok"), settings=_settings())
    first = read_manifest(connection.db)
    assert first is not None
    connection.db.query("CREATE (:Marker)")

    await run(WORKLOAD, _Benchmark(), connection=connection, llm=ScriptedLLM("ok"), settings=_settings())
    reused = read_manifest(connection.db)
    assert reused == first
    assert connection.db.query("MATCH (m:Marker) RETURN count(m) AS n")[0]["n"] == 1

    await run(WORKLOAD, _Benchmark(), connection=connection, llm=ScriptedLLM("ok"), settings=_settings(rebuild=True))
    assert connection.db.query("MATCH (m:Marker) RETURN count(m) AS n")[0]["n"] == 0


async def test_build_only_builds_and_asks_nothing(connection):
    llm = ScriptedLLM("ok")

    result = await run(WORKLOAD, _Benchmark(), connection=connection, llm=llm, settings=_settings(build_only=True))

    assert result is None
    assert llm.calls == 0
    manifest = read_manifest(connection.db)
    assert manifest is not None and manifest.status == "complete"


async def test_a_sharded_run_saves_each_shard_and_skips_the_ones_already_done(connection, tmp_path):
    sharded = replace(WORKLOAD, shard_size=1)

    first = await run(
        sharded, _Benchmark(), connection=connection, llm=ScriptedLLM("ok"), settings=_settings(), shard_dir=tmp_path
    )
    assert first is not None
    assert sorted(path.name for path in tmp_path.iterdir()) == ["shard-00.json", "shard-01.json"]
    assert first.meta.build["shards"] == 2

    (tmp_path / "shard-01.json").unlink()
    benchmark = _Benchmark()
    again = await run(
        sharded, benchmark, connection=connection, llm=ScriptedLLM("ok"), settings=_settings(), shard_dir=tmp_path
    )

    assert again is not None
    assert benchmark.fixture_calls == 1
    assert sorted(row.name for row in again.rows) == sorted(row.name for row in first.rows)
    assert load(tmp_path / "shard-01.json").rows[0].name in {"u1", "u2"}


def test_a_graph_missing_sessions_or_still_pending_is_not_asked(eval_graph):
    from context_graph_eval.inject import inject

    fixtures = _Benchmark().fixtures(WORKLOAD, _Benchmark().questions(WORKLOAD))

    with pytest.raises(ValueError, match="lacks 2 of 2 sessions"):
        require_built(eval_graph.db, fixtures)

    inject(fixtures, graph=eval_graph)
    with pytest.raises(ValueError, match="pending reconciliation"):
        require_built(eval_graph.db, fixtures)
