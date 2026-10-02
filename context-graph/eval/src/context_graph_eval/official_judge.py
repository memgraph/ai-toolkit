"""LongMemEval's own answer judge, vendored verbatim: the headline score (#409).

Our deepeval rubrics are ours to write and adjust, which makes a number they
produce us grading our own system. This judge is fixed outside our control and
is what published LongMemEval results are scored with, so it decides
``covered`` whenever it runs; the rubrics stay beside it as diagnostics.

The templates, call settings and parsing are copied from ``get_anscheck_prompt``
and its caller in UPSTREAM. Do not edit them -- comparability with that script
is the whole point; a change belongs upstream or in a new pinned commit.
"""

from __future__ import annotations

import asyncio
from typing import Any

UPSTREAM = "xiaowu0162/LongMemEval@d6dc8b50a2d9ac0c99485ea28fa5755c62414c34 src/evaluation/evaluate_qa.py"

#: The upstream script's 'gpt-4o' judge.
OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"

_ANSWER = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
    "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
    "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: "
    "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
    "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
    "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, do not penalize "
    "off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and "
    "the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is "
    "still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)
_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
    "contains the correct answer. Otherwise, answer no. If the response contains some previous information along "
    "with an updated answer, the response should be considered as correct as long as the updated answer is the "
    "required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only."
)
_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response from a model. Please "
    "answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's "
    "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only."
)
_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if "
    "the model correctly identifies the question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
    "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)
_TEMPLATES = {
    "single-session-user": _ANSWER,
    "single-session-assistant": _ANSWER,
    "multi-session": _ANSWER,
    "temporal-reasoning": _TEMPORAL,
    "knowledge-update": _KNOWLEDGE_UPDATE,
    "single-session-preference": _PREFERENCE,
}


def judges(question_type: str | None, *, abstention: bool) -> bool:
    """Whether the official judge has a template for this question; questions outside LongMemEval's types don't."""
    return abstention or question_type in _TEMPLATES


def anscheck_prompt(question_type: str, question: str, answer: str, response: str, *, abstention: bool) -> str:
    """Upstream's ``get_anscheck_prompt``: the judge prompt for one question.

    Raises:
        NotImplementedError: ``question_type`` is not a LongMemEval type, as upstream does.
    """
    if abstention:
        return _ABSTENTION.format(question, answer, response)
    if question_type not in _TEMPLATES:
        raise NotImplementedError(question_type)
    return _TEMPLATES[question_type].format(question, answer, response)


def is_correct(reply: str) -> bool:
    """Upstream's parsing: correct when 'yes' appears anywhere in the reply."""
    return "yes" in reply.strip().lower()


async def judge_all(
    prompts: dict[str, str],
    *,
    model: str = OFFICIAL_JUDGE_MODEL,
    client: Any = None,
    max_concurrent: int = 8,
) -> dict[str, bool | None]:
    """Judge each named prompt with upstream's call settings; None where the call failed.

    A failed call is unjudged, not wrong: counting it as a miss would report
    an outage as a score, the failure ``aggregate`` already refuses.
    """
    if client is None:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(max_retries=5)
    limiter = asyncio.Semaphore(max_concurrent)

    async def one(prompt: str) -> bool | None:
        async with limiter:
            try:
                completion = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], n=1, temperature=0, max_tokens=10
                )
            except Exception:
                return None
            return is_correct(completion.choices[0].message.content or "")

    labels = await asyncio.gather(*(one(prompt) for prompt in prompts.values()))
    return dict(zip(prompts, labels, strict=True))
