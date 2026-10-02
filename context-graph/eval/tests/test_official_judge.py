"""LongMemEval's own judge, vendored: templates, parsing and call settings."""

from types import SimpleNamespace

import pytest
from context_graph_eval.official_judge import (
    OFFICIAL_JUDGE_MODEL,
    anscheck_prompt,
    is_correct,
    judge_all,
    judges,
)

TYPES = (
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
)


def test_every_longmemeval_type_has_upstreams_template():
    for question_type in TYPES:
        prompt = anscheck_prompt(question_type, "Q?", "A.", "R.", abstention=False)
        assert prompt.endswith("Answer yes or no only.")
        assert "Question: Q?" in prompt and "Model Response: R." in prompt

    assert "do not penalize off-by-one errors" in anscheck_prompt("temporal-reasoning", "", "", "", abstention=False)
    assert "updated answer is the required answer" in anscheck_prompt("knowledge-update", "", "", "", abstention=False)
    assert "Rubric: A." in anscheck_prompt("single-session-preference", "Q?", "A.", "R.", abstention=False)


def test_an_abstention_question_is_judged_on_identifying_it_as_unanswerable():
    prompt = anscheck_prompt("multi-session", "Q?", "Not mentioned.", "R.", abstention=True)

    assert "correctly identifies the question as unanswerable" in prompt
    assert "Explanation: Not mentioned." in prompt


def test_a_question_outside_longmemevals_types_is_not_the_official_judges():
    assert not judges(None, abstention=False)
    assert judges(None, abstention=True)
    with pytest.raises(NotImplementedError):
        anscheck_prompt("gold-slice", "Q?", "A.", "R.", abstention=False)


def test_upstreams_parsing_is_yes_anywhere_in_the_reply():
    assert is_correct("Yes.")
    assert is_correct("  yes")
    assert not is_correct("No.")
    assert not is_correct("")


class _Completions:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies[kwargs["messages"][0]["content"]]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


class _Client:
    def __init__(self, replies):
        self.chat = SimpleNamespace(completions=_Completions(replies))


@pytest.mark.asyncio
async def test_judging_uses_upstreams_call_settings_and_a_failed_call_is_unjudged():
    client = _Client({"p1": "Yes", "p2": "No", "p3": RuntimeError("rate limited")})

    labels = await judge_all({"q1": "p1", "q2": "p2", "q3": "p3"}, client=client)

    assert labels == {"q1": True, "q2": False, "q3": None}
    call = client.chat.completions.calls[0]
    assert (call["model"], call["temperature"], call["max_tokens"], call["n"]) == (OFFICIAL_JUDGE_MODEL, 0, 10, 1)
    assert call["messages"] == [{"role": "user", "content": "p1"}]
