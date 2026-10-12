"""BEAM: conversion, the vendored judge, and the report."""

import math
from types import SimpleNamespace

import pytest
from context_graph_eval.beam import BeamRow, summarize
from context_graph_eval.beam_judge import BeamJudge, kendall_tau_b, ordering_score, rubric_prompt, rubric_score
from context_graph_eval.cli import parse_chat_ids
from context_graph_eval.convert.beam import BeamChat, to_goldens, to_session_fixtures


def _message(message_id, role, content, anchor=None):
    message = {"id": message_id, "role": role, "content": content}
    if anchor:
        message["time_anchor"] = anchor
    return message


CHAT = BeamChat(
    size="100K",
    chat_id=7,
    batches=[
        # Upstream leaves the first batch's own anchor None and sets it on its first message.
        {
            "batch_number": 1,
            "time_anchor": None,
            "turns": [
                [_message(0, "user", "My sprint ends March 29.", "March-15-2024"), _message(1, "assistant", "Ok.")]
            ],
        },
        {
            "batch_number": 2,
            "time_anchor": "April-05-2024",
            "turns": [[_message(2, "user", "Now it ends April 2."), _message(3, "assistant", "Noted.")]],
        },
    ],
    probing_questions={
        "knowledge_update": [
            {
                "question": "When does my sprint end?",
                "answer": "April 2",
                "source_chat_ids": {"original_info": [0], "updated_info": [2]},
                "rubric": ["LLM response should state: April 2"],
            }
        ],
        "abstention": [
            {"question": "Who is my manager?", "ideal_response": "Not mentioned.", "rubric": ["no information"]}
        ],
    },
)


def test_each_batch_is_one_dated_session_of_the_chats_user():
    fixtures = to_session_fixtures(CHAT)

    assert [(f.session_id, f.date, f.user_id) for f in fixtures] == [
        ("beam-100K-7--b1", "2024/03/15 (Fri) 00:00", "beam-100K-7"),
        ("beam-100K-7--b2", "2024/04/05 (Fri) 00:00", "beam-100K-7"),
    ]
    assert [turn.content for turn in fixtures[1].turns] == ["Now it ends April 2.", "Noted."]


def test_goldens_carry_the_rubric_the_user_and_the_cited_messages():
    abstention, update = to_goldens(CHAT)
    update_metadata = update.additional_metadata or {}

    assert update.name == "beam-100K-7-knowledge_update-0"
    assert update.expected_output == "April 2"
    assert update.context == ["user: My sprint ends March 29.", "user: Now it ends April 2."]
    assert update_metadata["rubric"] == ["LLM response should state: April 2"]
    assert update_metadata["user_id"] == "beam-100K-7"
    assert abstention.expected_output == "Not mentioned."
    assert (abstention.additional_metadata or {})["abstention"] is True


def test_chat_ids_parse_lists_and_ranges():
    assert parse_chat_ids("1") == [1]
    assert parse_chat_ids("3,1-3") == [3, 1, 2]
    with pytest.raises(ValueError):
        parse_chat_ids("one")


def test_rubric_prompt_fills_upstreams_placeholders():
    prompt = rubric_prompt("LLM response should state: 8 weeks", "It is 8 weeks.")

    assert "RUBRIC CRITERION (what to check): LLM response should state: 8 weeks" in prompt
    assert "RESPONSE TO EVALUATE: It is 8 weeks." in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"score": 0.5, "reason": "partial"}', (0.5, "partial")),
        ('```json\n{"score": 1.0, "reason": "ok"}\n```', (1.0, "ok")),
        ('{"score": 1.0, "reason": "unterminated', (1.0, "")),
        ("no json here", None),
    ],
)
def test_rubric_replies_are_read_as_upstream_reads_them(reply, expected):
    assert rubric_score(reply) == expected


def test_kendall_tau_b_matches_known_values():
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    # scipy.stats.kendalltau([1, 2, 2, 3], [1, 3, 2, 3], variant="b")
    assert kendall_tau_b([1, 2, 2, 3], [1, 3, 2, 3]) == pytest.approx(0.8)
    assert math.isnan(kendall_tau_b([1], [1]))


def test_ordering_scores_a_correct_order_one_and_a_reversed_order_zero():
    reference = ["a", "b", "c"]

    assert ordering_score(reference, ["a", "b", "c"])["tau_norm"] == 1.0
    assert ordering_score(reference, ["c", "b", "a"])["tau_norm"] == 0.0


class _Replies:
    """An OpenAI client that answers rubric prompts from ``scores`` in order and equivalence by exact match."""

    def __init__(self, scores):
        self._scores = iter(scores)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, model, messages, temperature):
        if messages[0]["role"] == "system":
            first, _, second = messages[1]["content"].partition("Second snippet: ")
            same = first.removeprefix("First snippet: ").strip() == second.strip()
            content = "YES" if same else "NO"
        else:
            score = next(self._scores)
            if isinstance(score, Exception):
                raise score
            content = f'{{"score": {score}, "reason": "r"}}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def test_a_question_scores_the_mean_of_its_rubric_items():
    verdict = await BeamJudge(client=_Replies([1.0, 0.5])).judge("information_extraction", ["x", "y"], "answer")

    assert verdict.score == 0.75
    assert verdict.item_scores == [1.0, 0.5]


async def test_a_failed_rubric_call_leaves_the_question_unjudged_not_zero():
    verdict = await BeamJudge(client=_Replies([1.0, RuntimeError("outage")])).judge("summarization", ["x", "y"], "a")

    assert verdict.score is None


async def test_event_ordering_is_scored_by_normalised_tau():
    verdict = await BeamJudge(client=_Replies([1.0, 1.0])).judge("event_ordering", ["b", "a"], "a\nb")

    assert verdict.llm_judge_score == 1.0
    assert verdict.score == 0.0


def test_summary_weights_abilities_equally_and_skips_unjudged_questions():
    def row(ability, score):
        return BeamRow(
            name="", user_id="u", ability=ability, question="", expected="", answer="", rubric=[], score=score
        )

    scores, overall = summarize([row("abstention", 1.0), row("abstention", None), row("summarization", 0.0)])

    assert [(s.ability, s.questions, s.judged, s.mean_score) for s in scores] == [
        ("abstention", 2, 1, 1.0),
        ("summarization", 1, 1, 0.0),
    ]
    assert overall == 0.5


def test_a_limited_run_asks_an_equal_share_per_ability_spread_over_the_chats():
    from collections import Counter

    from context_graph_eval.beam import select_questions
    from context_graph_eval.convert.beam import ABILITIES

    chats = [
        BeamChat(
            size="100K",
            chat_id=chat_id,
            batches=CHAT.batches,
            probing_questions={ability: [{"question": "q", "rubric": ["r"]}] * 2 for ability in ABILITIES},
        )
        for chat_id in range(1, 5)
    ]
    goldens = [golden for chat in chats for golden in to_goldens(chat)]

    selected = select_questions(goldens, 20)

    assert Counter((g.additional_metadata or {})["question_type"] for g in selected) == Counter(
        {ability: 2 for ability in ABILITIES}
    )
    assert set(Counter((g.additional_metadata or {})["user_id"] for g in selected).values()) == {5}
    assert select_questions(goldens, None) == goldens
