"""Convert LongMemEval v1 records into deepeval Goldens.

LongMemEval (https://github.com/xiaowu0162/LongMemEval, MIT) supplies Tier 1 of
the eval corpus -- questions that already carry gold answers, so they are
converted rather than authored. See docs/research/2026-08-memory-benchmarks.md.
"""

from dataclasses import dataclass

from deepeval.dataset import Golden

SOURCE = "longmemeval-v1"

#: LongMemEval's own name for questions whose correct answer is "not in memory".
ABSTENTION_QUESTION_TYPE = "abstention"


def to_golden(record: dict) -> Golden:
    """Convert one LongMemEval question record into a Golden."""
    question_type = record["question_type"]
    return Golden(
        input=record["question"],
        expected_output=record["answer"],
        context=_evidence_turns(record),
        name=record["question_id"],
        source_file=SOURCE,
        additional_metadata={
            # Tier 1 is the adopted corpus, scored separately from the authored
            # Tier 2 so an organizational-recall regression cannot hide behind a
            # personal-memory gain.
            "tier": 1,
            "question_type": question_type,
            "question_date": record["question_date"],
            "abstention": question_type == ABSTENTION_QUESTION_TYPE,
        },
    )


@dataclass(frozen=True)
class SessionFixture:
    """One haystack session, ready to be injected into an eval database.

    ``holds_evidence`` is bookkeeping, not a scoring input -- retrieval must
    never get to see which sessions carry the answer.
    """

    session_id: str
    date: str
    turns: list[str]
    holds_evidence: bool


def to_session_fixtures(record: dict) -> list[SessionFixture]:
    """Convert a record's haystack into injectable session fixtures.

    Distractor sessions are kept deliberately. They are what give retrieval
    precision -- and so the payload-size efficiency metric -- something to
    measure; a haystack of evidence alone would score well by construction.
    """
    evidence_ids = set(record["answer_session_ids"])
    return [
        SessionFixture(
            session_id=session_id,
            date=date,
            turns=[_format_turn(turn) for turn in session],
            holds_evidence=session_id in evidence_ids,
        )
        # strict: these three are parallel arrays upstream. If they ever
        # disagree, fail loudly -- silently truncating would drop haystack
        # sessions and corrupt the corpus with no visible error.
        for session_id, date, session in zip(
            record["haystack_session_ids"],
            record["haystack_dates"],
            record["haystack_sessions"],
            strict=True,
        )
    ]


def _format_turn(turn: dict) -> str:
    return f"{turn['role']}: {turn['content']}"


def _evidence_turns(record: dict) -> list[str]:
    """The turns a correct answer actually depends on.

    Only turns flagged ``has_answer`` count. Whole evidence sessions would drag
    in surrounding chatter, and ContextualRecall would then score retrieval
    against facts the answer never needed.
    """
    return [_format_turn(turn) for session in record["haystack_sessions"] for turn in session if turn.get("has_answer")]
