"""BEAM's own judge, vendored verbatim: the headline score for a BEAM run.

Same reasoning as ``official_judge`` for LongMemEval: this judge is fixed
outside our control and is what published BEAM results are scored with.

Every ability is scored the same way upstream: each rubric item is judged on
its own (1.0, 0.5 or 0.0) and the question's score is their mean. Event
ordering is reported by ``tau_norm`` instead -- a Kendall tau-b over the
answer's lines aligned to the rubric by LLM equivalence -- which is what
upstream's ``report_results.py`` averages for that column.

The prompts, model and call settings are copied from UPSTREAM. Do not edit
them -- comparability with that script is the whole point.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

UPSTREAM = (
    "mohammadtavakoli78/BEAM@b2da22eac88bb0874c64665f13457eb99835774a src/evaluation/compute_metrics.py, src/prompts.py"
)

#: Upstream's ``gpt_llm``: ChatOpenAI at temperature 0.
BEAM_JUDGE_MODEL = "gpt-4.1-mini"

#: Line by line rather than one triple-quoted block, so whitespace hooks cannot
#: strip the trailing spaces upstream's prompt carries.
_RUBRIC_PROMPT = (
    "\n"
    "You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.\n"
    "\n"
    "## EVALUATION INPUTS\n"
    "- RUBRIC CRITERION (what to check): <rubric_item>\n"
    "- RESPONSE TO EVALUATE: <llm_response>\n"
    "\n"
    "## EVALUATION RUBRIC:\n"
    "The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. \n"
    "\n"
    "**IMPORTANT**: Pay careful attention to whether the rubric specifies:\n"
    "- **Positive requirements** (things the response SHOULD include/do)\n"
    '- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")\n'
    "\n"
    "## RESPONSIVENESS REQUIREMENT\n"
    "A compliant response must be **on-topic** and attempt to answer it.\n"
    "- If the response does not address the QUESTION, score **0.0** and stop.\n"
    "- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.\n"
    "\n"
    "## SEMANTIC TOLERANCE RULES:\n"
    "Judge by meaning, not exact wording.\n"
    "- Accept **paraphrases** and **synonyms** that preserve intent.\n"
    "- **Case/punctuation/whitespace** differences must be ignored.\n"
    "- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.\n"
    "- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.\n"
    "\n"
    "## STYLE NEUTRALITY (prevents style contamination):\n"
    "Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).\n"
    "- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.\n"
    "- Only evaluate format when the rubric **explicitly** mandates it.\n"
    "\n"
    "## SCORING SCALE:\n"
    "- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.\n"
    "  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).\n"
    "  - Negative: prohibited element **absent** AND response is **responsive**.\n"
    "  \n"
    "- **0.5 (Partial Compliance)**: Partially complies.\n"
    "  - Positive: element present but minor inaccuracies/incomplete execution.\n"
    "  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.\n"
    "  \n"
    "- **0.0 (No Compliance)**: Fails to comply.\n"
    "  - Positive: required element missing or incorrect.\n"
    "  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.\n"
    "\n"
    "## EVALUATION INSTRUCTIONS:\n"
    "1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).\n"
    "\n"
    '2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:\n'
    "   - **All elements** must be present for full compliance (1.0)\n"
    "   - **Some elements** present indicates partial compliance (0.5)\n"
    "   - **No elements** present indicates no compliance (0.0)\n"
    "   \n"
    "3. **Check Compliance**: \n"
    "   - For positive requirements: Look for the presence and quality of the required element\n"
    "   - For negative constraints: Look for the absence of the prohibited element\n"
    "\n"
    "4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.\n"
    "\n"
    "5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.\n"
    "\n"
    "## OUTPUT FORMAT:\n"
    "Return your evaluation in JSON format with two fields:\n"
    "\n"
    "{\n"
    '   "score": [your score: 1.0, 0.5, or 0.0],\n'
    '   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"\n'
    "}\n"
    "\n"
    "NOTE: ONLY output the json object, without any explanation before or after that\n"
)

_EQUIVALENCE_SYSTEM = "\n            You are a binary classifier.\n            If the TWO snippets describe the SAME event/fact, reply **YES**\n            Otherwise reply **NO**. No extra words.\n            DO NOT provide any exaplanation.\n        "
_EQUIVALENCE_USER = "First snippet: {first_paragraph} \n\n                       Second snippet: {second_paragraph}\n                    "


def rubric_prompt(rubric_item: str, llm_response: str) -> str:
    """Upstream's ``unified_llm_judge_base_prompt`` for one rubric item. It is never shown the question."""
    return _RUBRIC_PROMPT.replace("<rubric_item>", rubric_item).replace("<llm_response>", llm_response)


def parse_json_response(response: str) -> Any:
    """Upstream's ``parse_json_response``.

    Raises:
        ValueError: no JSON in the response.
    """
    response = response.strip()

    if response.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*?\}|\[.*?\])", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as error:
            raise ValueError(f"Found possible JSON but failed to parse it: {error}") from error

    raise ValueError("No valid JSON found in response.")


def rubric_score(reply: str) -> tuple[float, str] | None:
    """The score and reason in a rubric reply; None when it can't be read.

    Upstream falls back to ``json_repair`` where this falls back to reading
    the ``score`` field directly -- both recover the same malformed replies
    in practice, and this needs no extra dependency.
    """
    try:
        parsed = parse_json_response(reply)
        return float(parsed["score"]), str(parsed.get("reason", ""))
    except (ValueError, KeyError, TypeError):
        match = re.search(r'"score"\s*:\s*([0-9.]+)', reply)
        return (float(match.group(1)), "") if match else None


def kendall_tau_b(x: list[int], y: list[int]) -> float:
    """Kendall's tau-b, ties corrected, as ``scipy.stats.kendalltau(variant="b")``; nan when undefined."""
    concordant = discordant = x_ties = y_ties = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                x_ties += 1
                y_ties += 1
            elif dx == 0:
                x_ties += 1
            elif dy == 0:
                y_ties += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    pairs = len(x) * (len(x) - 1) // 2
    denominator = math.sqrt((pairs - x_ties) * (pairs - y_ties))
    return (concordant - discordant) / denominator if denominator else math.nan


def ordering_score(reference: list[str], system_canon: list[str]) -> dict[str, float]:
    """Upstream's ``event_ordering_score`` after alignment: precision/recall/F1 and normalised tau.

    An undefined tau (fewer than two distinct items) is reported as 0.0;
    upstream's nan would poison every average it enters.
    """
    tp = len(set(reference) & set(system_canon))
    fp = len([item for item in system_canon if item not in reference])
    fn = len([item for item in reference if item not in system_canon])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    union = list(dict.fromkeys(reference + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq: list[str]) -> list[int]:
        ranks = {item: i + 1 for i, item in enumerate(seq)}
        return [ranks.get(item, tie_rank) for item in union]

    tau = kendall_tau_b(to_rank(reference), to_rank(system_canon))
    tau_norm = 0.0 if math.isnan(tau) else (tau + 1) / 2
    return {"precision": precision, "recall": recall, "f1": f1, "tau_norm": tau_norm, "final_score": tau_norm * f1}


@dataclass(frozen=True)
class BeamVerdict:
    """One question's judgement. ``score`` is the number upstream's report averages."""

    score: float | None
    llm_judge_score: float | None
    item_scores: list[float | None] = field(default_factory=list)
    item_reasons: list[str] = field(default_factory=list)
    ordering: dict[str, float] | None = None


class BeamJudge:
    """Upstream's evaluator, calling the OpenAI API with its settings."""

    def __init__(self, *, model: str = BEAM_JUDGE_MODEL, client: Any = None, max_concurrent: int = 8):
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(max_retries=5)
        self._client: Any = client
        self._model = model
        self._limiter = asyncio.Semaphore(max_concurrent)

    async def _complete(self, messages: list[dict[str, str]]) -> str | None:
        async with self._limiter:
            try:
                completion = await self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=0
                )
            except Exception:
                return None
        return completion.choices[0].message.content or ""

    async def _item(self, rubric_item: str, response: str) -> tuple[float | None, str]:
        reply = await self._complete([{"role": "user", "content": rubric_prompt(rubric_item, response)}])
        parsed = rubric_score(reply) if reply is not None else None
        return parsed if parsed is not None else (None, "")

    async def _equivalent(self, first: str, second: str) -> bool:
        user = _EQUIVALENCE_USER.format(first_paragraph=first, second_paragraph=second)
        reply = await self._complete(
            [{"role": "system", "content": _EQUIVALENCE_SYSTEM}, {"role": "user", "content": user}]
        )
        return "yes" in (reply or "").lower()

    async def _align(self, reference: list[str], system: list[str]) -> list[str]:
        """Upstream's ``align_with_llm``: each answer line takes the first unused rubric item it matches.

        Sequential, as upstream: which items are already used depends on every earlier match.
        """
        used: set[int] = set()
        aligned = []
        for line in system:
            match = None
            for index, item in enumerate(reference):
                if index not in used and await self._equivalent(item, line):
                    match = index
                    break
            if match is None:
                aligned.append(line)
            else:
                aligned.append(reference[match])
                used.add(match)
        return aligned

    async def judge(self, ability: str, rubric: list[str], response: str) -> BeamVerdict:
        """Score one answer. A rubric item whose call failed is unjudged; any unjudged item leaves the score None."""
        if not rubric:
            return BeamVerdict(score=None, llm_judge_score=None)
        items = await asyncio.gather(*(self._item(item, response) for item in rubric))
        scores = [score for score, _ in items]
        judged = [score for score in scores if score is not None]
        llm_judge_score = sum(judged) / len(judged) if len(judged) == len(scores) else None
        ordering = None
        score = llm_judge_score
        if ability == "event_ordering":
            ordering = ordering_score(rubric, await self._align(rubric, response.split("\n")))
            score = ordering["tau_norm"]
        return BeamVerdict(
            score=score,
            llm_judge_score=llm_judge_score,
            item_scores=scores,
            item_reasons=[reason for _, reason in items],
            ordering=ordering,
        )
