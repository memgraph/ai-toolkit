"""Convert LongMemEval v1 records into deepeval Goldens.

LongMemEval (https://github.com/xiaowu0162/LongMemEval, MIT) supplies Tier 1 of
the eval corpus -- questions that already carry gold answers, so they are
converted rather than authored. See docs/research/2026-08-memory-benchmarks.md.
"""

import itertools
import json
import os
import random
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


#: Chunk size for streaming the haystack to disk. The 's' variant is ~277MB,
#: which is not something to hold in memory in one buffer.
_CHUNK_BYTES = 1 << 20

#: How many times to re-attempt a truncated or failed download.
_DOWNLOAD_ATTEMPTS = 3


def cache_root() -> Path:
    """The directory every pinned benchmark download is cached under (``$XDG_CACHE_HOME`` or ``~/.cache``)."""
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "context-graph-eval"


def haystack_path(variant: str = "s", revision: str = DEFAULT_REVISION, *, cache_dir: Path | None = None) -> Path:
    """Where a pinned variant is cached between runs; keyed by variant and commit sha, so immutable."""
    root = cache_dir or cache_root()
    return Path(root) / f"longmemeval-{variant}-{revision[:12]}.json"


def fetch(
    variant: str = "s",
    revision: str = DEFAULT_REVISION,
    *,
    dest: Path,
    attempts: int = _DOWNLOAD_ATTEMPTS,
    opener: Any = None,
) -> Path:
    """Download a pinned LongMemEval variant to ``dest``, or reuse it if cached.

    Written to a ``.part`` file and renamed only once the whole body arrived
    and matches ``Content-Length``, so ``dest`` existing means it is complete.

    Raises:
        OSError: every attempt failed or was truncated.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    url = download_url(variant, revision)
    open_url = opener or urllib.request.urlopen

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with open_url(url) as response:
                expected = response.headers.get("Content-Length") if hasattr(response, "headers") else None
                written = 0
                with partial.open("wb") as handle:
                    while chunk := response.read(_CHUNK_BYTES):
                        handle.write(chunk)
                        written += len(chunk)
            if expected is not None and written != int(expected):
                raise OSError(f"truncated download: got {written} bytes, expected {expected}")
            partial.replace(dest)
            return dest
        except Exception as error:  # retried below; re-raised if the last attempt
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                break
    raise OSError(f"could not download {url} after {attempts} attempts: {last_error}") from last_error


def build_corpus(records: Iterable[dict], limit: int | None = None) -> list[Golden]:
    """Convert upstream records into Goldens, optionally sampling down to ``limit``.

    Sampling is deterministic (no spurious diff on regeneration), stratified
    by ``(question_type, abstention)`` since upstream clusters both,
    proportional to upstream so rare strata are not over-weighted, and floored
    so none rounds to zero. The sample is shuffled with a fixed seed because
    ``run --limit`` takes a prefix of the committed corpus, and round-robin
    order would make every prefix floor-uniform.
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
    sampled = sampled[:limit]
    random.Random(_SHUFFLE_SEED).shuffle(sampled)
    return [to_golden(record) for record in sampled]


#: Fixed so a regenerated corpus is byte-identical (no spurious diff) rather
#: than reshuffled every build.
_SHUFFLE_SEED = 297


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

    ``user_id`` is the person whose history the session is: the question it
    came from. ``holds_evidence`` is bookkeeping, not a scoring input --
    retrieval must never get to see which sessions carry the answer.
    """

    session_id: str
    date: str
    turns: list["Turn"]
    holds_evidence: bool
    user_id: str


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
    """Convert a record's haystack, distractors included, into injectable session fixtures.

    ``max_sessions`` trims distractors only -- evidence is always kept, even
    past the cap -- and makes the score an upper bound. Off by default.

    Each fixture belongs to the question's user, with its session id
    namespaced by the question: upstream reuses distractors across questions,
    and reconciliation resolves a single user per session.
    """
    question_id = record["question_id"]
    evidence_ids = set(record["answer_session_ids"])
    fixtures = [
        SessionFixture(
            session_id=f"{question_id}--{session_id}",
            date=date,
            turns=[Turn(role=turn["role"], content=turn["content"]) for turn in session],
            holds_evidence=session_id in evidence_ids,
            user_id=question_id,
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

    # The cap is exceeded rather than cutting evidence: over budget is a cost
    # problem, an unanswerable question a corrupt measurement. Distractors stay
    # in upstream order so two runs inject the same graph.
    evidence = [fixture for fixture in fixtures if fixture.holds_evidence]
    distractors = [fixture for fixture in fixtures if not fixture.holds_evidence]
    room = max(0, max_sessions - len(evidence))
    return evidence + distractors[:room]


def _format_turn(turn: dict) -> str:
    return f"{turn['role']}: {turn['content']}"


def _evidence_turns(record: dict) -> list[str]:
    """The turns a correct answer actually depends on.

    Only turns flagged ``has_answer`` count. Whole evidence sessions would drag
    in surrounding chatter, and ContextualRecall would then score retrieval
    against facts the answer never needed.
    """
    return [_format_turn(turn) for session in record["haystack_sessions"] for turn in session if turn.get("has_answer")]
