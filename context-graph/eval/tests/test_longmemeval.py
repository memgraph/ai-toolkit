"""Tests for converting LongMemEval v1 records into deepeval Goldens.

Record shape follows the schema documented in the LongMemEval repository
(https://github.com/xiaowu0162/LongMemEval), surveyed in
docs/research/2026-08-memory-benchmarks.md.
"""

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
    golden = to_golden(_record(question_type="abstention"))

    assert golden.additional_metadata["abstention"] is True


def test_ordinary_questions_are_not_marked_as_abstention():
    golden = to_golden(_record())

    assert golden.additional_metadata["abstention"] is False


def test_every_haystack_session_becomes_an_injectable_fixture():
    fixtures = to_session_fixtures(_record())

    assert [f.session_id for f in fixtures] == ["answer_1", "distractor_1"]


def test_a_fixture_carries_its_session_date_and_turns():
    answer_session, _ = to_session_fixtures(_record())

    assert answer_session.date == "2023/05/20 (Sat) 14:03"
    assert answer_session.turns == [
        "user: I adopted a beagle named Max",
        "assistant: Congratulations on Max!",
    ]


def test_distractor_sessions_are_kept_not_filtered_out():
    """Distractors are the point: they are what make retrieval precision mean
    anything. A haystack of only evidence would score near-perfectly by
    construction."""
    fixtures = to_session_fixtures(_record())

    distractor = fixtures[1]
    assert distractor.session_id == "distractor_1"
    assert distractor.holds_evidence is False


def test_evidence_sessions_are_marked_as_such():
    answer_session, _ = to_session_fixtures(_record())

    assert answer_session.holds_evidence is True
