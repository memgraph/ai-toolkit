"""Convert BEAM conversations and probing questions into eval goldens and fixtures.

BEAM (https://github.com/mohammadtavakoli78/BEAM, code MIT, data CC BY-SA 4.0)
is a second adopted benchmark beside LongMemEval: long multi-domain chats --
coding, math, writing, advice -- each probed by questions across ten memory
abilities, judged per rubric item rather than yes/no.

Its shape differs from LongMemEval's in the way that matters for injection:
one chat is one user's whole history, and all of that chat's questions are
asked of it. So a chat is the unit of fetching and of ``user_id``, and each of
its time-anchored batches becomes one dated session.

Nothing converted is committed. LongMemEval is MIT, so its converted corpus
lives in git; BEAM's data is share-alike, so it is fetched from a pinned
upstream commit instead, and that pin is what proves two runs asked the same
questions.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from deepeval.dataset import Golden

from .longmemeval import SessionFixture, Turn

SOURCE = "beam"

_REPO = "mohammadtavakoli78/BEAM"

#: Pinned upstream commit; bumping it invalidates prior BEAM baselines.
DEFAULT_REVISION = "b2da22eac88bb0874c64665f13457eb99835774a"

#: Chat sizes and how many chats upstream ships for each.
CHAT_COUNTS = {"100K": 20, "500K": 35, "1M": 35}

#: Upstream's ten abilities, in its report's column order.
ABILITIES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

#: The field each ability keeps its reference answer in; upstream names it per ability.
_REFERENCE_FIELDS = ("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance")

#: Upstream's batch time anchors, e.g. 'March-15-2024'.
_ANCHOR_FORMAT = "%B-%d-%Y"

#: inject.CORPUS_DATE_FORMAT, which session dates must be in to be stamped on turns.
_SESSION_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


@dataclass(frozen=True)
class BeamChat:
    """One upstream chat: its conversation and the questions asked of it."""

    size: str
    chat_id: int
    #: Upstream's batches, each ``{"turns": [[message, ...], ...], ...}``.
    batches: list[dict[str, Any]]
    probing_questions: dict[str, list[dict[str, Any]]]

    @property
    def user_id(self) -> str:
        """The user whose whole history this chat is."""
        return f"beam-{self.size}-{self.chat_id}"


def cache_dir(revision: str = DEFAULT_REVISION) -> Path:
    """Where a pinned revision's files are cached between runs."""
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "context-graph-eval"
    return root / f"beam-{revision[:12]}"


def _raw_url(path: str, revision: str) -> str:
    return f"https://raw.githubusercontent.com/{_REPO}/{revision}/{path}"


def _fetch_json(path: str, revision: str, *, cache: Path, opener: Any = None) -> Any:
    """Download one upstream JSON file, or reuse it from the cache. None when upstream has no such file."""
    local = cache / path
    if not local.exists():
        open_url = opener or urllib.request.urlopen
        try:
            with open_url(_raw_url(path, revision)) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        # Parsed before caching, so a truncated body raises here instead of
        # being adopted as the conversation.
        json.loads(body)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".part")
        partial.write_bytes(body)
        partial.replace(local)
    return json.loads(local.read_text(encoding="utf-8"))


def fetch_chat(size: str, chat_id: int, *, revision: str = DEFAULT_REVISION, opener: Any = None) -> BeamChat:
    """Fetch one chat and its probing questions from the pinned revision.

    Upstream answers from ``chat_trunecated.json`` whenever a chat has one
    (``answer_generation.py``), so it is preferred here too: the full chat runs
    past the size bucket, and questions were checked against the truncated one.
    """
    if size not in CHAT_COUNTS:
        raise ValueError(f"unknown BEAM chat size {size!r}; expected one of {sorted(CHAT_COUNTS)}")
    cache = cache_dir(revision)
    base = f"chats/{size}/{chat_id}"
    batches = _fetch_json(f"{base}/chat_trunecated.json", revision, cache=cache, opener=opener)
    if batches is None:
        batches = _fetch_json(f"{base}/chat.json", revision, cache=cache, opener=opener)
    questions = _fetch_json(f"{base}/probing_questions/probing_questions.json", revision, cache=cache, opener=opener)
    if batches is None or questions is None:
        raise FileNotFoundError(f"BEAM {size} chat {chat_id} is missing at {revision[:12]}")
    return BeamChat(size=size, chat_id=chat_id, batches=batches, probing_questions=questions)


def _messages(batch: dict[str, Any]) -> list[dict[str, Any]]:
    return [message for turn in batch["turns"] for message in turn]


def _batch_date(batch: dict[str, Any]) -> str:
    """The batch's time anchor in the injected session-date format.

    Upstream sets the anchor on the batch's first message, and leaves the
    batch-level field None for the first batch.
    """
    anchor = batch.get("time_anchor") or next(
        (message["time_anchor"] for message in _messages(batch) if message.get("time_anchor")), None
    )
    if anchor is None:
        raise ValueError(f"BEAM batch {batch.get('batch_number')} has no time anchor")
    return datetime.strptime(anchor, _ANCHOR_FORMAT).strftime(_SESSION_DATE_FORMAT)


def to_session_fixtures(chat: BeamChat) -> list[SessionFixture]:
    """One dated session per upstream batch, all belonging to the chat's user.

    ``holds_evidence`` is always False: every question is asked of the whole
    chat, so no session is a distractor, and the field is never written to
    the graph either way.
    """
    return [
        SessionFixture(
            session_id=f"{chat.user_id}--b{index}",
            date=_batch_date(batch),
            turns=[Turn(role=message["role"], content=message["content"]) for message in _messages(batch)],
            holds_evidence=False,
            user_id=chat.user_id,
        )
        for index, batch in enumerate(chat.batches, start=1)
    ]


def _source_ids(question: dict[str, Any]) -> list[int]:
    """Message ids a question cites: a flat list, or a dict of lists keyed by role in the question."""
    cited = question.get("source_chat_ids") or []
    if isinstance(cited, dict):
        cited = [value for values in cited.values() for value in (values if isinstance(values, list) else [values])]
    return [int(value) for value in cited if isinstance(value, int) or str(value).isdigit()]


def _reference(question: dict[str, Any]) -> str:
    for key in _REFERENCE_FIELDS:
        if question.get(key):
            return str(question[key])
    return "; ".join(question.get("rubric", []))


def to_goldens(chat: BeamChat) -> list[Golden]:
    """The chat's probing questions as goldens, in upstream's ability order.

    ``context`` holds the cited messages as ``"role: content"`` strings, the
    shape ``scoring.evidence_recall`` matches against retrieval. The judge's
    rubric rides in metadata: it, not ``expected_output``, is what BEAM scores.
    """
    by_id = {message["id"]: message for batch in chat.batches for message in _messages(batch)}
    goldens = []
    for ability in ABILITIES:
        for index, question in enumerate(chat.probing_questions.get(ability, [])):
            cited = [by_id[i] for i in _source_ids(question) if i in by_id]
            goldens.append(
                Golden(
                    input=question["question"],
                    expected_output=_reference(question),
                    context=[f"{message['role']}: {message['content']}" for message in cited],
                    name=f"{chat.user_id}-{ability}-{index}",
                    source_file=f"{SOURCE}-{chat.size}",
                    additional_metadata={
                        "tier": 1,
                        "benchmark": SOURCE,
                        "question_type": ability,
                        "abstention": ability == "abstention",
                        "rubric": list(question.get("rubric", [])),
                        "user_id": chat.user_id,
                        "difficulty": question.get("difficulty"),
                    },
                )
            )
    return goldens
