"""Tests for converting LongMemEval v1 records into deepeval Goldens.

Record shape follows the schema documented in the LongMemEval repository
(https://github.com/xiaowu0162/LongMemEval), surveyed in
docs/research/2026-08-memory-benchmarks.md.
"""

import re

from context_graph_eval.convert.longmemeval import to_golden, to_session_fixtures


def _record(**overrides):
    """A minimal LongMemEval record, shaped like the real dataset."""
    record = {
        "question_id": "gpt4_1a2b3c",
        "question_type": "single-session-user",
        "question": "What breed is the dog I adopted?",
        "answer": "A beagle.",
        "question_date": "2023/06/15 (Thu) 09:12",
        "haystack_session_ids": ["answer_1", "distractor_1"],
        "haystack_dates": ["2023/05/20 (Sat) 14:03", "2023/05/22 (Mon) 10:41"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I adopted a beagle named Max", "has_answer": True},
                {"role": "assistant", "content": "Congratulations on Max!"},
            ],
            [
                {"role": "user", "content": "What time does the hardware store close?"},
                {"role": "assistant", "content": "Most close around 6pm."},
            ],
        ],
        "answer_session_ids": ["answer_1"],
    }
    record.update(overrides)
    return record


def test_question_and_answer_become_input_and_expected_output():
    golden = to_golden(_record())

    assert golden.input == "What breed is the dog I adopted?"
    assert golden.expected_output == "A beagle."


def test_non_string_answers_are_coerced():
    """Real LongMemEval data answers counting questions with a bare integer.
    Golden.expected_output is typed str, so an uncoerced answer aborts the whole
    corpus build -- found by running against the real dataset, not by the
    hand-built records above."""
    golden = to_golden(_record(question="How many dogs do I own?", answer=3))

    assert golden.expected_output == "3"


def test_context_holds_only_the_turns_flagged_as_evidence():
    golden = to_golden(_record())

    assert golden.context == ["user: I adopted a beagle named Max"]


def test_golden_is_traceable_to_its_upstream_record():
    golden = to_golden(_record())

    assert golden.name == "gpt4_1a2b3c"
    assert golden.source_file == "longmemeval-v1"


def test_golden_carries_the_tier_and_question_type_it_is_scored_under():
    golden = to_golden(_record())

    assert golden.additional_metadata["tier"] == 1
    assert golden.additional_metadata["question_type"] == "single-session-user"


def test_abstention_questions_are_marked_so_they_can_be_scored_apart():
    """Abstention questions -- where the correct answer is "that isn't in
    memory" -- are identified upstream by an ``_abs`` suffix on question_id, NOT
    by question_type, which keeps its original value. Verified against the real
    dataset: 30 of 500 records end in ``_abs`` and none has question_type
    "abstention", despite the upstream README listing it as a type."""
    golden = to_golden(_record(question_id="gpt4_1a2b3c_abs"))

    assert golden.additional_metadata["abstention"] is True


def test_an_abstention_question_keeps_its_original_question_type():
    golden = to_golden(_record(question_id="gpt4_1a2b3c_abs", question_type="temporal-reasoning"))

    assert golden.additional_metadata["question_type"] == "temporal-reasoning"


def test_ordinary_questions_are_not_marked_as_abstention():
    golden = to_golden(_record())

    assert golden.additional_metadata["abstention"] is False


def test_every_haystack_session_becomes_an_injectable_fixture():
    fixtures = to_session_fixtures(_record())

    assert [f.session_id for f in fixtures] == [
        "gpt4_1a2b3c--answer_1",
        "gpt4_1a2b3c--distractor_1",
    ]


def test_namespaced_ids_satisfy_the_actions_graph_constraint():
    """actions-graph validates session_id against ^[a-zA-Z0-9_-]{1,128}$, so a
    ':' separator is rejected outright at injection. Real ids are at worst 17 +
    27 characters, leaving ample room under the limit."""
    fixtures = to_session_fixtures(_record())

    for fixture in fixtures:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", fixture.session_id)


def test_session_ids_are_namespaced_per_question():
    """Upstream reuses sessions across questions from a shared distractor pool
    -- 3,942 of 23,867 haystack ids in the real dataset are duplicates. Injected
    raw, colliding ids would MERGE onto one Session node, piling turns from
    different questions together and duplicating content on every reuse."""
    first = to_session_fixtures(_record(question_id="q1"))
    second = to_session_fixtures(_record(question_id="q2"))

    assert {f.session_id for f in first}.isdisjoint({f.session_id for f in second})


def test_a_fixture_carries_its_session_date_and_turns():
    answer_session, _ = to_session_fixtures(_record())

    assert answer_session.date == "2023/05/20 (Sat) 14:03"
    assert [(t.role, t.content) for t in answer_session.turns] == [
        ("user", "I adopted a beagle named Max"),
        ("assistant", "Congratulations on Max!"),
    ]


def test_distractor_sessions_are_kept_not_filtered_out():
    """Distractors are the point: they are what make retrieval precision mean
    anything. A haystack of only evidence would score near-perfectly by
    construction."""
    fixtures = to_session_fixtures(_record())

    distractor = fixtures[1]
    assert distractor.session_id == "gpt4_1a2b3c--distractor_1"
    assert distractor.holds_evidence is False


def test_evidence_sessions_are_marked_as_such():
    answer_session, _ = to_session_fixtures(_record())

    assert answer_session.holds_evidence is True
