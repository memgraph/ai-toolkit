"""How correct the extracted graph is, measured apart from the benchmark (#411).

The benchmark answers from turn text, so a wrong fact rarely costs a point;
the graph is still what a reader queries and what the answerer is shown. This
measures it directly:

- **Semantic labels** on a seeded sample of extracted edges, half from user
  turns and half from assistant turns, stratified by relation type. A judge
  from another provider than the extractor and the answerer reads each edge
  against the sentence it came from (``r.text``), its speaker and its turn
  date, and gives one LABELS entry.
- **Deterministic counts** over the whole graph, no judge: repeats, relative-
  time tails, the ``prefers`` share, ontology non-conformance, generic mentions
  merged across owners, and unresolved dates.

The judge is calibrated once against hand labels (``agreement``) before its
numbers are trusted.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

#: One label per edge. ``correct`` is the only good one; the rest name what is wrong.
LABELS = ("correct", "wrong_type", "wrong_endpoint", "advice_as_event", "wrong_date", "not_supported")
Label = Literal["correct", "wrong_type", "wrong_endpoint", "advice_as_event", "wrong_date", "not_supported"]

DEFAULT_SAMPLE = 200
DEFAULT_SEED = 0

#: A tail naming a point relative to when it was said, which belongs on a Date, not as an entity.
RELATIVE_TIME = re.compile(
    r"\b(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|a few|few|several|couple of)\s+"
    r"(?:day|week|month|year|hour)s?\s+ago\b"
    r"|\b(?:last|next|this|past)\s+(?:week|month|year|weekend|night|morning|evening|summer|winter|spring|fall"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(?:yesterday|tomorrow|tonight)\b",
    re.IGNORECASE,
)

_EDGES = (
    "MATCH (a)-[r]->(b) WHERE r.chunk IS NOT NULL "
    "RETURN type(r) AS type, coalesce(a.text, 'user') AS head, "
    "coalesce(a.entity_type, CASE WHEN a:User THEN 'User' END) AS head_type, "
    "coalesce(b.text, 'user') AS tail, "
    "coalesce(b.entity_type, CASE WHEN b:User THEN 'User' END) AS tail_type, "
    "r.text AS sentence, r.role AS role, toString(r.valid_at) AS said_on, r.source_id AS source, "
    "r.ontology_conformant AS conformant"
)


@dataclass(frozen=True)
class Edge:
    """One extracted edge, as the judge sees it."""

    type: str
    head: str
    head_type: str | None
    tail: str
    tail_type: str | None
    sentence: str
    role: str
    said_on: str
    source: str

    @property
    def key(self) -> str:
        """Stable across rebuilds of the same extraction, unlike Memgraph's internal ids."""
        return f"{self.source}|{self.type}|{self.head}|{self.tail}"


@dataclass(frozen=True)
class Labelled:
    edge: Edge
    label: str | None
    reason: str = ""


@dataclass
class QualityReport:
    """What one measurement found; ``labels`` is empty when no judge ran."""

    counts: dict[str, Any]
    labels: list[Labelled] = field(default_factory=list)

    def label_shares(self) -> dict[str, dict[str, float]]:
        """Per speaker and overall: each label's share of the judged edges."""
        out: dict[str, dict[str, float]] = {}
        groups: dict[str, list[str]] = defaultdict(list)
        for item in self.labels:
            if item.label is not None:
                groups[item.edge.role].append(item.label)
                groups["all"].append(item.label)
        for role, labels in groups.items():
            tally = Counter(labels)
            out[role] = {label: tally[label] / len(labels) for label in LABELS}
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "label_shares": self.label_shares(),
            "labels": [{**asdict(item.edge), "label": item.label, "reason": item.reason} for item in self.labels],
        }


def edges(db: Any) -> list[Edge]:
    """Every extracted edge in the graph."""
    return [
        Edge(
            type=row["type"],
            head=str(row["head"]),
            head_type=row["head_type"],
            tail=str(row["tail"]),
            tail_type=row["tail_type"],
            sentence=row["sentence"] or "",
            role=row["role"] or "unknown",
            said_on=(row["said_on"] or "")[:10],
            source=row["source"] or "",
        )
        for row in db.query(_EDGES)
    ]


def _rank(seed: int, edge: Edge) -> str:
    return hashlib.sha256(f"{seed}:{edge.key}".encode()).hexdigest()


def sample(all_edges: list[Edge], n: int = DEFAULT_SAMPLE, seed: int = DEFAULT_SEED) -> list[Edge]:
    """``n`` edges, half from user turns and half from assistant turns, each half spread over relation types.

    Seeded and keyed on the edge's content, so the same graph always gives the
    same sample. Each relation type gets a share of its speaker's half in
    proportion to its count, at least one; the edges within a type are the
    lowest-ranked by seed.
    """
    out: list[Edge] = []
    for role, quota in (("user", n // 2), ("assistant", n - n // 2)):
        by_type: dict[str, list[Edge]] = defaultdict(list)
        for edge in all_edges:
            if edge.role == role:
                by_type[edge.type].append(edge)
        total = sum(len(group) for group in by_type.values())
        if not total:
            continue
        quota = min(quota, total)
        share = {t: max(1, round(quota * len(group) / total)) for t, group in by_type.items()}
        # Trim or top up the rounding, largest types first, to hit the quota exactly.
        order = sorted(by_type, key=lambda t: (-len(by_type[t]), t))
        while sum(share.values()) > quota:
            t = next(t for t in order if share[t] > 1)
            share[t] -= 1
        while sum(share.values()) < quota:
            t = next(t for t in order if share[t] < len(by_type[t]))
            share[t] += 1
        for t in order:
            out += sorted(by_type[t], key=lambda e: _rank(seed, e))[: share[t]]
    return out


def counts(db: Any, all_edges: list[Edge]) -> dict[str, Any]:
    """The deterministic half of the measure: no judge, over the whole graph."""
    user_headed = [e for e in all_edges if e.head_type == "User"]
    # Joined in Python: Action.action_id carries no index, so matching a turn
    # per mention source in Cypher scans every turn for each one.
    owner = {
        row["id"]: row["user"]
        for row in db.query(
            "MATCH (u:User)-[:HAD_SESSION]->(:Session)-[:HAS_ACTION]->(a:Action) RETURN a.action_id AS id, u.user_id AS user"
        )
    }
    generic = []
    for row in db.query(
        "MATCH (n:gliner2)-[m:MENTIONED_IN]->() "
        "RETURN n.text AS text, n.entity_type AS type, collect(coalesce(m.sources, [])) AS sources"
    ):
        owners = {owner[s] for sources in row["sources"] for s in sources if s in owner}
        if len(owners) > 1 and row["text"] and not any(c.isupper() for c in row["text"]):
            generic.append(row)
    dates = db.query(
        "MATCH (d:gliner2) WHERE d.entity_type = 'Date' RETURN count(d) AS total, count(d.value) AS resolved"
    )[0]
    return {
        "edges": len(all_edges),
        "edges_by_role": dict(Counter(e.role for e in all_edges)),
        # Per pair of nodes, not per text: two people's "wedding" nodes are different facts.
        "repeats": db.query(
            "MATCH (a)-[r]->(b) WHERE r.chunk IS NOT NULL WITH a, type(r) AS t, b, count(r) AS k WHERE k > 1 "
            "RETURN coalesce(sum(k - 1), 0) AS n"
        )[0]["n"],
        "relative_time_tails": sum(1 for e in all_edges if RELATIVE_TIME.search(e.tail)),
        "prefers_share_of_user_edges": (
            sum(1 for e in user_headed if e.type == "prefers") / len(user_headed) if user_headed else None
        ),
        "non_conformant_edges": db.query(
            "MATCH ()-[r]->() WHERE r.chunk IS NOT NULL AND r.ontology_conformant = false RETURN count(r) AS n"
        )[0]["n"],
        "generic_mentions_merged_across_owners": len(generic),
        "generic_persons_merged_across_owners": sum(1 for row in generic if row["type"] == "Person"),
        "dates": dates["total"],
        "unresolved_dates": dates["total"] - dates["resolved"],
    }


PROMPT = """You check one fact extracted from a conversation against the sentence it was extracted from.

Fact: ({head}: {head_type}) -[{type}]-> ({tail}: {tail_type})
Said by: {role}, on {said_on}
Sentence: {sentence}

Give exactly one label:
- correct: the sentence states this relation between these two things, as something that is or happened.
- wrong_type: the two things are right, but the sentence relates them differently than "{type}".
- wrong_endpoint: one end is not what the sentence relates (a different thing, or a fragment of one).
- advice_as_event: the sentence suggests, recommends or speaks generally; the fact presents it as something that \
is or happened.
- wrong_date: the sentence dates the event, and the fact's time ({said_on}, when it was said) is not that date.
- not_supported: the sentence does not say this at all.
Pick the first label that applies, in this order: not_supported, wrong_endpoint, wrong_type, advice_as_event, \
wrong_date, correct.

Reply with JSON only: {{"label": "<label>", "reason": "<one short sentence>"}}"""


def prompt_for(edge: Edge) -> str:
    return PROMPT.format(**{**asdict(edge), "head_type": edge.head_type or "?", "tail_type": edge.tail_type or "?"})


async def label(sampled: list[Edge], judge: Any, *, max_concurrent: int = 8) -> list[Labelled]:
    """The judge's label for each edge; None where it failed or answered off-rubric -- unjudged, not wrong."""
    from pydantic import BaseModel

    class _Verdict(BaseModel):
        label: str
        reason: str = ""

    limiter = asyncio.Semaphore(max_concurrent)

    async def one(edge: Edge) -> Labelled:
        async with limiter:
            try:
                verdict, _ = await judge.a_generate(prompt_for(edge), schema=_Verdict)
            except Exception:
                return Labelled(edge, None)
            return Labelled(edge, verdict.label if verdict.label in LABELS else None, verdict.reason)

    return list(await asyncio.gather(*(one(edge) for edge in sampled)))


#: How many sampled edges are hand-labelled to calibrate the judge, and the agreement that trusts it.
CALIBRATION_SIZE = 30
CALIBRATION_AGREEMENT = 27


def calibration_set(sampled: list[Edge], size: int = CALIBRATION_SIZE) -> list[Edge]:
    """The edges to hand-label: an even spread through the sample, so both speakers and many types appear."""
    if len(sampled) <= size:
        return list(sampled)
    step = len(sampled) / size
    return [sampled[int(i * step)] for i in range(size)]


def agreement(judged: list[Labelled], hand: dict[str, str]) -> tuple[int, int, list[tuple[str, str, str | None]]]:
    """(agreeing, compared, disagreements as (edge key, hand label, judge label)) over the hand-labelled edges."""
    by_key = {item.edge.key: item.label for item in judged}
    compared = [(key, label, by_key.get(key)) for key, label in hand.items() if key in by_key]
    disagreements = [row for row in compared if row[1] != row[2]]
    return len(compared) - len(disagreements), len(compared), disagreements
