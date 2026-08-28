"""Convert LongMemEval v1 records into deepeval Goldens.

LongMemEval (https://github.com/xiaowu0162/LongMemEval, MIT) supplies Tier 1 of
the eval corpus -- questions that already carry gold answers, so they are
converted rather than authored. See docs/research/2026-08-memory-benchmarks.md.
"""

import itertools
import json
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from deepeval.dataset import Golden

SOURCE = "longmemeval-v1"

#: How the dataset marks a question whose correct answer is "that isn't in
#: memory". Note this is a ``question_id`` suffix, *not* a ``question_type``:
#: abstention records keep their original type. The upstream README lists
#: "abstention" among the question types, but the published data does not use
#: it -- verified against the real dataset, where 30 of 500 records carry this
#: suffix and none has that type.
ABSTENTION_ID_SUFFIX = "_abs"

_REPO = "xiaowu0162/longmemeval-cleaned"

#: Pinned upstream revision. A moving ref (``main``) would silently change the
#: corpus between runs, which would break the cross-version score comparison the
#: committed corpus exists to make possible -- the same reason #304 pins the
#: judge model. Bump deliberately; treat a bump as invalidating prior baselines.
DEFAULT_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"

_VARIANT_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


def download_url(variant: str = "s", revision: str = DEFAULT_REVISION) -> str:
    """URL of a pinned LongMemEval variant.

    ``oracle`` is refused: it ships evidence sessions only, so retrieval faces
    no distractors and both precision and the payload-size efficiency metric
    would score well by construction.
    """
    if variant == "oracle":
        raise ValueError(
            "the oracle variant has no distractor sessions, so retrieval precision "
            "and payload-size efficiency would score well by construction; use 's' or 'm'"
        )
    if variant not in _VARIANT_FILES:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_VARIANT_FILES)}")
    return f"https://huggingface.co/datasets/{_REPO}/resolve/{revision}/{_VARIANT_FILES[variant]}"


def fetch(variant: str = "s", revision: str = DEFAULT_REVISION, *, dest: Path) -> Path:
    """Download a pinned LongMemEval variant to ``dest``.

    Only the *converted* output is committed, never this file -- see #302.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(download_url(variant, revision)) as response:
        dest.write_bytes(response.read())
    return dest


def build_corpus(records: Iterable[dict], limit: int | None = None) -> list[Golden]:
    """Convert upstream records into Goldens, optionally sampling down to ``limit``.

    Sampling is deterministic, stratified by ``(question_type, abstention)``,
    and proportional to upstream with a small floor per stratum.

    *Deterministic* so a regenerated corpus produces no spurious diff and two
    runs stay comparable.

    *Stratified on both dimensions* because they are independent and upstream
    clusters each: taking the first N would skew the question-type mix, and
    stratifying on type alone samples zero abstention questions, since those sit
    in contiguous runs at the end of each type block.

    *Proportional* so the aggregate score reflects the real distribution --
    equal-weighting strata over-samples rare ones badly (it turned an upstream
    6% abstention rate into 40% of a 60-question sample, enough for abstention
    behaviour to dominate the headline).

    *With a floor* so a rare stratum cannot round to zero at small limits and
    vanish silently.
    """
    records = list(records)
    if limit is None or limit >= len(records):
        return [to_golden(record) for record in records]

    strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for record in records:
        strata[_stratum(record)].append(record)

    quotas = _quotas({key: len(group) for key, group in strata.items()}, limit)

    # Round-robin across strata up to each one's quota, so a truncated sample
    # stays balanced rather than front-loading whichever stratum sorts first.
    taken = [strata[key][: quotas[key]] for key in sorted(strata)]
    sampled = [record for group in itertools.zip_longest(*taken) for record in group if record is not None]
    return [to_golden(record) for record in sampled[:limit]]


def _quotas(sizes: dict[tuple[str, bool], int], limit: int, floor: int = 2) -> dict[tuple[str, bool], int]:
    """How many records to take from each stratum.

    Proportional to stratum size, but at least ``floor`` (and never more than
    the stratum holds). Largest strata absorb the rounding so the quotas sum to
    ``limit``.
    """
    total = sum(sizes.values())
    quotas = {key: min(size, max(floor, round(limit * size / total))) for key, size in sizes.items()}

    # Reconcile rounding drift against the largest strata first -- they absorb a
    # record with the least proportional distortion -- but rotate through them
    # rather than repeatedly hitting the same one, which would leave equally
    # sized categories with visibly unequal shares.
    by_size = sorted(sizes, key=lambda key: (-sizes[key], key))
    rotation = itertools.cycle(by_size)
    stalled = 0
    while (drift := sum(quotas.values()) - limit) != 0 and stalled <= len(by_size):
        key = next(rotation)
        if drift > 0 and quotas[key] > floor:
            quotas[key] -= 1
            stalled = 0
        elif drift < 0 and quotas[key] < sizes[key]:
            quotas[key] += 1
            stalled = 0
        else:
            stalled += 1
    return quotas


def _stratum(record: dict) -> tuple[str, bool]:
    return (
        record["question_type"],
        str(record["question_id"]).endswith(ABSTENTION_ID_SUFFIX),
    )


def load_raw(path: Path) -> list[dict]:
    """Read a downloaded LongMemEval file."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def to_golden(record: dict) -> Golden:
    """Convert one LongMemEval question record into a Golden."""
    question_id = record["question_id"]
    return Golden(
        input=record["question"],
        # Coerced: counting questions upstream are answered with a bare int,
        # and Golden.expected_output is typed str.
        expected_output=str(record["answer"]),
        context=_evidence_turns(record),
        name=question_id,
        source_file=SOURCE,
        additional_metadata={
            # Tier 1 is the adopted corpus, scored separately from the authored
            # Tier 2 so an organizational-recall regression cannot hide behind a
            # personal-memory gain.
            "tier": 1,
            "question_type": record["question_type"],
            "question_date": record["question_date"],
            # Scored apart: for these, a confident answer is the failure.
            "abstention": str(question_id).endswith(ABSTENTION_ID_SUFFIX),
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
    turns: list["Turn"]
    holds_evidence: bool


@dataclass(frozen=True)
class Turn:
    """One conversational turn.

    Role and content stay separate rather than pre-formatted into one string:
    injection records each turn as a Message with its own role, and flattening
    early would throw that away.
    """

    role: str
    content: str


def to_session_fixtures(record: dict, max_sessions: int | None = None) -> list[SessionFixture]:
    """Convert a record's haystack into injectable session fixtures.

    Distractor sessions are kept deliberately. They are what give retrieval
    precision -- and so the payload-size efficiency metric -- something to
    measure; a haystack of evidence alone would score well by construction.

    ``max_sessions`` trims the haystack, evidence first. It exists because
    reconciliation cost scales with *sessions* while coverage needs *questions*,
    and upstream couples them at roughly 47:1 -- so a full-pipeline run at an
    affordable session count would otherwise be reduced to two questions, at
    which point one flip is 50pp and coverage says nothing.

    Trimming makes retrieval easier, so any score measured this way is an upper
    bound and is not comparable to a full-haystack run. Off by default: the full
    haystack is the honest difficulty, and flattering it must be asked for.

    Session ids are kept verbatim. Upstream draws distractors from a shared pool
    and reuses them across questions -- 3,942 of 23,867 haystack ids in the real
    dataset repeat -- but *zero* of those repeats carry differing content, so a
    repeated id genuinely is the same session. Letting it become one node
    matches how a real organizational graph would hold it; namespacing per
    question would store byte-identical copies and pay to reconcile each. The
    duplicate-turns hazard that suggests is handled by deduplicating at
    injection instead (see ``inject.inject_batch``).
    """
    evidence_ids = set(record["answer_session_ids"])
    fixtures = [
        SessionFixture(
            session_id=session_id,
            date=date,
            turns=[Turn(role=turn["role"], content=turn["content"]) for turn in session],
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

    if max_sessions is None or len(fixtures) <= max_sessions:
        return fixtures

    # Evidence first, then distractors in upstream order. Dropping an evidence
    # session would make the question unanswerable for a reason unrelated to
    # recall, and the resulting miss would be indistinguishable from a real
    # failure. Upstream order rather than a random sample keeps the choice
    # deterministic: the subsample is part of what a run measured, so two runs
    # of the same corpus must inject the same graph.
    evidence = [fixture for fixture in fixtures if fixture.holds_evidence]
    distractors = [fixture for fixture in fixtures if not fixture.holds_evidence]
    return (evidence + distractors)[:max_sessions]


def _format_turn(turn: dict) -> str:
    return f"{turn['role']}: {turn['content']}"


def _evidence_turns(record: dict) -> list[str]:
    """The turns a correct answer actually depends on.

    Only turns flagged ``has_answer`` count. Whole evidence sessions would drag
    in surrounding chatter, and ContextualRecall would then score retrieval
    against facts the answer never needed.
    """
    return [_format_turn(turn) for session in record["haystack_sessions"] for turn in session if turn.get("has_answer")]
