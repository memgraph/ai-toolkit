"""Learning a user's ontology from their sessions: when a derivation runs, and what it adopts.

The decisions this implements (map #431):

- **Schedule (#433).** It counts the user's reconciled sessions past a
  content floor and runs at 2, 4, 8, … 128 of them, then every 128. Each run
  reads the delta since the last adopted run, sampled down to a budget, with
  the adopted model N as its starting point. The delta is snapshotted by a
  watermark at the start, so a session finishing mid-run belongs to the next.
- **Claim (#433).** A run holds an expiring claim on ``(:User)``, so a second
  run exits and a crashed one's claim lapses.
- **Gate (#435).** From the 16-session run on, a quarter of the delta (at
  most 8 sessions) is held out; earlier runs gate in-sample. Up to three
  candidates (different seeds) are derived and each is extracted over the
  held-out sessions; one passes when it is no worse than N, within a small
  tolerance, on catch-all share and on user-turn coverage. The best passing
  candidate is adopted. If none passes, N stays, the candidates are stored as
  rejected, and the watermark holds so the delta rolls into the next run.
- **Adoption (#434).** Merged types are relabelled in the graph straight
  away; retired ones are left as they are. If the run added anything, its
  delta is re-extracted under the new model. Older sessions never are.

What a run spends, the gate's numbers, the agreement with N and the
changelog are written on every version node, adopted or rejected (#435).
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Protocol

from hygm import CATCH_ALL_LABELS, DerivationError, LlmRecommendationStrategy, require_valid_identifier

from .ontology import OntologyVersion, adopt, adopted, next_version, reject

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from actions_graph import ActionsGraph
    from hygm import Derivation, HygmModel, Llm
    from unstructured2graph import Document
    from unstructured2graph.gliner2_observer import Measurement

    from .core import SessionsGraph

#: Run when this many qualifying sessions exist, then every EVERY after the last.
MILESTONES = (2, 4, 8, 16, 32, 64, 128)
EVERY = 128
#: A session counts toward the schedule only past this much of the user's own words.
MIN_USER_TURNS, MIN_USER_CHARS = 2, 1000
#: The most delta sessions one run reads; above it the delta is sampled.
DELTA_BUDGET = 64
#: Held-out slice: this share of the delta, at most HOLDOUT_MAX, once the delta has HOLDOUT_FROM.
HOLDOUT_SHARE, HOLDOUT_MAX, HOLDOUT_FROM = 0.25, 8, 8
SEEDS = 3
#: How much worse than N a candidate may score and still pass, per gate number.
TOLERANCE = 0.01
CLAIM = timedelta(hours=3)

_QUALIFYING = """
MATCH (:User {user_id: $user_id})-[:HAD_SESSION]->(s:Session {reconciliation_status: 'completed'})
MATCH (s)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:UserMessage)
WHERE a.text IS NOT NULL
WITH s, count(DISTINCT a) AS turns, sum(size(a.text)) AS chars
WHERE turns >= $min_turns AND chars >= $min_chars
RETURN s.session_id AS session_id, s.reconciled_at AS reconciled_at
ORDER BY reconciled_at, session_id
"""


class GateObserver(Protocol):
    """``hygm.Observer`` that can also score a model for the gate, like ``GLiNER2Observer``."""

    def observe(self, model: HygmModel, sample: Sequence[Any]) -> Mapping[str, Any]: ...

    def measure(self, model: HygmModel, sample: Sequence[Document], catch_alls: Sequence[str]) -> Measurement: ...


def next_milestone(after: int) -> int:
    """The qualifying-session count the next run waits for, after one at `after`."""
    return next((m for m in MILESTONES if m > after), after - after % EVERY + EVERY)


def qualifying_sessions(db: Any, user_id: str) -> list[tuple[str, str]]:
    """(session_id, reconciled_at) of the user's sessions that count toward the schedule, oldest first."""
    rows = db.query(_QUALIFYING, params={"user_id": user_id, "min_turns": MIN_USER_TURNS, "min_chars": MIN_USER_CHARS})
    return [(row["session_id"], row["reconciled_at"]) for row in rows]


def due(db: Any, user_id: str) -> int | None:
    """The milestone a run would be for, or None when the user hasn't reached the next one.

    Never due while the adopted version has ``derive = "off"``.
    """
    if adopted(db, user_id).derive == "off":
        return None
    milestone = next_milestone(_state(db, user_id)["milestone"])
    return milestone if len(qualifying_sessions(db, user_id)) >= milestone else None


@dataclass(frozen=True)
class GateScore:
    """One model's gate numbers over the held-out sessions."""

    catch_all_share: float
    coverage: float

    @classmethod
    def of(cls, measured: Measurement) -> GateScore:
        return cls(round(measured.catch_all_share, 4), round(measured.coverage, 4))

    def passes_against(self, base: GateScore) -> bool:
        return self.catch_all_share <= base.catch_all_share + TOLERANCE and self.coverage >= base.coverage - TOLERANCE


@dataclass
class CandidateReport:
    seed: int
    score: GateScore | None = None
    passed: bool = False
    agreement: float | None = None
    added: int = 0
    merged: int = 0
    retired: int = 0
    error: str | None = None
    #: Seconds deriving (LLM calls and the observe pass), and extracting the gate sessions.
    derive_seconds: float = 0.0
    gate_seconds: float = 0.0


@dataclass
class RunReport:
    """What one run did; printed by the CLI and written on the version nodes."""

    user_id: str
    milestone: int | None
    outcome: str  # "adopted", "rejected", "nothing to do", "claimed elsewhere", "derive off"
    delta: int = 0
    sampled: int = 0
    held_out: int = 0
    in_sample: bool = False
    base: GateScore | None = None
    candidates: list[CandidateReport] = field(default_factory=list)
    adopted_version: int | None = None
    reextracted: int = 0
    llm: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    #: Seconds extracting the gate sessions under N, and re-extracting the delta after adoption.
    base_seconds: float = 0.0
    reextract_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run(
    graph: SessionsGraph,
    user_id: str,
    *,
    llm: Llm,
    observer: GateObserver,
    actions_graph: ActionsGraph | None = None,
    force: bool = False,
    seeds: int = SEEDS,
    usage: Any = None,
) -> RunReport:
    """Derive, gate and maybe adopt the user's next ontology version.

    Args:
        graph: Where the user's sessions and versions live.
        llm: ``hygm.Llm`` for the propose, consolidate and prune calls.
        observer: Observes candidates and measures them for the gate.
        force: Run even if no milestone was reached (a backfill, or by hand).
        seeds: How many candidates to derive.
        usage: The LLM's usage tally, copied into the report when given.

    Returns:
        What happened; never raises for a candidate that fails, only records it.
    """
    started = time.monotonic()
    db = graph._db
    current = adopted(db, user_id)
    if current.derive == "off":
        return RunReport(user_id, None, "derive off")
    state = _state(db, user_id)
    milestone = next_milestone(state["milestone"])
    sessions = qualifying_sessions(db, user_id)
    if len(sessions) < milestone and not force:
        return RunReport(user_id, milestone, "nothing to do")
    if not _claim(db, user_id):
        return RunReport(user_id, milestone, "claimed elsewhere")
    try:
        watermark = max(reconciled_at for _, reconciled_at in sessions) if sessions else None
        delta = [
            sid for sid, reconciled_at in sessions if state["watermark"] is None or reconciled_at > state["watermark"]
        ]
        report = RunReport(user_id, milestone, "nothing to do", delta=len(delta))
        if not delta:
            return report
        actions_graph = graph._default_actions_graph(actions_graph, "derive")
        rng = random.Random(f"{user_id}:{milestone}")
        sample = sorted(rng.sample(delta, DELTA_BUDGET)) if len(delta) > DELTA_BUDGET else delta
        documents = {sid: graph._prepare_session(sid, actions_graph).document(user_id) for sid in sample}
        held_out = _hold_out(sample, rng)
        training = [sid for sid in sample if sid not in held_out] if held_out else sample
        gate_on = held_out or sample
        report.sampled, report.held_out, report.in_sample = len(sample), len(held_out), not held_out

        gate_docs = [documents[sid] for sid in gate_on]
        clock = time.monotonic()
        base_measured = observer.measure(current.model, gate_docs, CATCH_ALL_LABELS)
        report.base = GateScore.of(base_measured)
        report.base_seconds = round(time.monotonic() - clock, 1)

        strategy = LlmRecommendationStrategy(llm, observer)
        best: tuple[GateScore, Derivation, CandidateReport] | None = None
        candidates: list[tuple[Derivation, CandidateReport]] = []
        for seed in range(seeds):
            candidate = CandidateReport(seed)
            report.candidates.append(candidate)
            clock = time.monotonic()
            try:
                derivation = strategy.derive(
                    current.model,
                    conversations=[_conversation(documents[sid]) for sid in training],
                    observe_sample=[documents[sid] for sid in training],
                    pinned=current.pinned,
                    counts=current.counts,
                    pool=current.pool,
                    seed=seed,
                )
            except DerivationError as exc:
                candidate.error = str(exc)
                continue
            except Exception as exc:  # a provider or observer failure fails this seed, not the run
                candidate.error = f"{type(exc).__name__}: {exc}"
                continue
            candidate.derive_seconds = round(time.monotonic() - clock, 1)
            clock = time.monotonic()
            measured = observer.measure(derivation.model, gate_docs, CATCH_ALL_LABELS)
            candidate.gate_seconds = round(time.monotonic() - clock, 1)
            candidate.score = GateScore.of(measured)
            candidate.passed = candidate.score.passes_against(report.base)
            candidate.agreement = _agreement(base_measured, measured, derivation)
            candidate.added = sum(c.op in {"add", "restore"} for c in derivation.changelog)
            candidate.merged = sum(c.op == "merge" for c in derivation.changelog)
            candidate.retired = sum(c.op == "retire" for c in derivation.changelog)
            candidates.append((derivation, candidate))
            if candidate.passed and (best is None or _better(candidate.score, best[0])):
                best = (candidate.score, derivation, candidate)

        if usage is not None:
            report.llm = asdict(usage)
        report.seconds = round(time.monotonic() - started, 1)
        if best is None:
            report.outcome = "rejected"
            for derivation, candidate in candidates:
                reject(db, _version(current, derivation, next_version(db, user_id), report, candidate))
            _set_state(db, user_id, milestone=milestone, watermark=state["watermark"])
            return report

        _, derivation, chosen = best
        version = adopt(db, _version(current, derivation, next_version(db, user_id), report, chosen))
        report.outcome, report.adopted_version = "adopted", version.version
        for other, candidate in candidates:
            if other is not derivation:
                reject(db, _version(current, other, next_version(db, user_id), report, candidate))
        _relabel_merges(graph, derivation)
        clock = time.monotonic()
        if chosen.added:
            for sid in delta:
                summary = await graph.reconcile_session(
                    sid, lightrag_wrapper=None, actions_graph=actions_graph, enforce_ontology=True, summarize=False
                )
                report.reextracted += summary.status == "completed"
        report.reextract_seconds = round(time.monotonic() - clock, 1)
        _set_state(db, user_id, milestone=milestone, watermark=watermark)
        report.seconds = round(time.monotonic() - started, 1)
        return report
    finally:
        _release(db, user_id)


def _hold_out(sample: list[str], rng: random.Random) -> list[str]:
    """The held-out sessions, or none when the delta is too small and the gate runs in-sample."""
    if len(sample) < HOLDOUT_FROM:
        return []
    return sorted(rng.sample(sample, min(HOLDOUT_MAX, max(1, round(len(sample) * HOLDOUT_SHARE)))))


def _conversation(document: Document) -> str:
    """A session as the propose call reads it: one line per turn, by speaker."""
    lines = [
        f"{segment.role or 'tool'}: {document.text[segment.start : segment.end].strip()}"
        for segment in document.segments
    ]
    return "=== conversation ===\n" + "\n".join(lines)


def _better(a: GateScore, b: GateScore) -> bool:
    """Ranks passing candidates: more coverage gained than catch-all share kept."""
    return a.coverage - a.catch_all_share > b.coverage - b.catch_all_share


def _agreement(base: Measurement, candidate: Measurement, derivation: Derivation) -> float | None:
    """Share of spans both models typed that the candidate types as N did, merges resolved (#435's churn signal)."""
    renames = {
        change.label: change.into for change in derivation.changelog if change.op == "merge" and change.kind == "node"
    }
    shared = base.spans.keys() & candidate.spans.keys()
    if not shared:
        return None
    same = sum(renames.get(base.spans[key], base.spans[key]) == candidate.spans[key] for key in shared)
    return round(same / len(shared), 4)


def _version(
    current: OntologyVersion, derivation: Derivation, number: int, report: RunReport, candidate: CandidateReport
) -> OntologyVersion:
    return OntologyVersion(
        user_id=current.user_id,
        version=number,
        model=derivation.model,
        source="derived",
        derive=current.derive,
        pinned=current.pinned,
        source_hash=current.source_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        counts=derivation.counts,
        pool=derivation.pool,
        changelog=tuple(asdict(change) for change in derivation.changelog),
        report={
            **{key: value for key, value in report.as_dict().items() if key != "candidates"},
            "candidate": asdict(candidate),
            "based_on": current.version,
            "unanchored_values": list(derivation.unanchored_values),
            "saturation": _saturation(derivation),
        },
    )


def _saturation(derivation: Derivation) -> dict[str, int]:
    """Observed mentions that fit types the model already had vs. ones it needed new types for (#433)."""
    added = {change.label for change in derivation.changelog if change.op in {"add", "restore"}}
    types = derivation.observation.get("types", {})
    fitting = sum(stats.get("mentions", 0) for label, stats in types.items() if label not in added)
    return {"fitting_existing": fitting, "needing_new": sum(types.get(label, {}).get("mentions", 0) for label in added)}


def _relabel_merges(graph: SessionsGraph, derivation: Derivation) -> None:
    """Move entities and relationships of a merged-away type onto its survivor, in place (#434).

    Possible since a global entity's key no longer holds its type (#442):
    after relabelling, new mentions of the name land on the same node.
    """
    workspace = graph._extraction_backend_for(derivation.model).workspace_label
    for change in derivation.changelog:
        if change.op != "merge" or change.into is None:
            continue
        source = require_valid_identifier(change.label, "merged label")
        target = require_valid_identifier(change.into, "surviving label")
        if change.kind == "node":
            # Labels can't be parameters; both are validated identifiers above.
            graph._db.query(
                f"MATCH (n:{workspace} {{entity_type: $source}}) SET n.entity_type = $target "
                f"REMOVE n:{source} SET n:{target}",
                params={"source": source, "target": target},
            )
        else:
            graph._db.query(
                f"MATCH (h)-[r:{source}]->(t) CREATE (h)-[moved:{target}]->(t) SET moved = properties(r) DELETE r"
            )


def _state(db: Any, user_id: str) -> dict[str, Any]:
    rows = db.query(
        "MATCH (u:User {user_id: $user_id}) RETURN u.derive_milestone AS milestone, u.derive_watermark AS watermark",
        params={"user_id": user_id},
    )
    row = rows[0] if rows else {}
    return {"milestone": row.get("milestone") or 0, "watermark": row.get("watermark")}


def _set_state(db: Any, user_id: str, *, milestone: int, watermark: str | None) -> None:
    db.query(
        "MATCH (u:User {user_id: $user_id}) SET u.derive_milestone = $milestone, u.derive_watermark = $watermark",
        params={"user_id": user_id, "milestone": milestone, "watermark": watermark},
    )


def _claim(db: Any, user_id: str) -> bool:
    # One query, one transaction: of two runs checking and setting at once, one sees the other's claim.
    now = datetime.now(timezone.utc)
    rows = db.query(
        """
        MATCH (u:User {user_id: $user_id})
        WHERE u.derive_claimed_until IS NULL OR u.derive_claimed_until < $now
        SET u.derive_claimed_until = $until
        RETURN u.user_id AS user_id
        """,
        params={"user_id": user_id, "now": now.isoformat(), "until": (now + CLAIM).isoformat()},
    )
    return bool(rows)


def _release(db: Any, user_id: str) -> None:
    db.query("MATCH (u:User {user_id: $user_id}) REMOVE u.derive_claimed_until", params={"user_id": user_id})
