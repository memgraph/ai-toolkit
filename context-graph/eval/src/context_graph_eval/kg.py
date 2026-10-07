"""How the default knowledge-graph construction behaves on BEAM's long chats.

`context-graph-eval kg` runs what reconcile runs by default -- GLiNER2 under
hygm's default model, its value-only pass and the user resolver -- over a
fixed, evenly spread sample of windows from pinned BEAM chats, with no
Memgraph and no LLM. It reports what a change to the defaults moves:

- speed: seconds per window, and with `--observe` the open-endpoint pass a
  derivation pays;
- volume and shape: mentions and relations per window, by type;
- fit: the share of mentions in the catch-alls, in value types, User-headed
  edges, and user turns with at least one typed relation (the gate's
  coverage);
- known facts (probes), when the LongMemEval corpus is cached locally;
- stability: how much of a saved baseline's mentions and edges a run
  reproduces.

Mentions and edges are fingerprinted from session, offsets and type, never
text, so a baseline can be committed: BEAM is CC BY-SA, and nothing converted
from it is (see the README). A change that moves the defaults reruns this
against the committed baseline and, if the move is intended, updates it.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hygm import HygmModel
    from unstructured2graph import Document

    from .convert.beam import BeamChat

#: The committed baseline the default run is compared against.
BASELINE_PATH = Path(__file__).resolve().parents[2] / "baselines" / "kg-beam-100k.json"
DEFAULT_WINDOWS = 200

#: Known facts from LongMemEval turns: (name, a regex finding the user's turn, what must be extracted).
#: Checked only when the pinned LongMemEval corpus is cached; nothing from it is committed.
PROBES = (
    ("25:50 personal best", r"personal best time of 25:50", ("entity", "Duration", "25:50")),
    (
        "MoMA visit",
        r"got back from a guided tour at the Museum of Modern Art",
        ("relation", "User", "Museum of Modern Art"),
    ),
)


@dataclass
class KgRun:
    """One run: what it ran on, what it measured, and the fingerprints a later run compares against."""

    meta: dict[str, Any]
    metrics: dict[str, Any]
    probes: dict[str, bool | None] = field(default_factory=dict)
    fingerprints: dict[str, list[str]] = field(default_factory=dict)


def documents(chats: list[BeamChat]) -> list[tuple[str, Document]]:
    """Each BEAM session as reconcile hands it to extraction: every turn cut to reconcile's
    limit, repeats dropped, joined by blank lines, one segment per turn."""
    from sessions_graph.reconciliation import MAX_RECONCILABLE_CHARS

    from unstructured2graph import Document, Segment

    from .convert.beam import to_session_fixtures

    built = []
    for chat in chats:
        for fixture in to_session_fixtures(chat):
            texts = list(dict.fromkeys(turn.content[:MAX_RECONCILABLE_CHARS] for turn in fixture.turns if turn.content))
            roles = {}
            for turn in fixture.turns:
                roles.setdefault(turn.content[:MAX_RECONCILABLE_CHARS], turn.role)
            segments, cursor = [], 0
            for text in texts:
                segments.append(Segment(cursor, cursor + len(text), roles[text]))
                cursor += len(text) + 2
            built.append((fixture.session_id, Document("\n\n".join(texts), tuple(segments), fixture.user_id)))
    return built


def run_kg(
    chats: list[BeamChat],
    *,
    windows: int = DEFAULT_WINDOWS,
    model: HygmModel | None = None,
    observe: bool = False,
    meta: dict[str, Any] | None = None,
) -> KgRun:
    """Extract a spread sample of `windows` windows from `chats` and measure it.

    Args:
        model: The graph model to extract under; hygm's default when None.
        observe: Also time the open-endpoint pass a derivation runs.
    """
    from dataclasses import replace

    from hygm import CATCH_ALL_LABELS, USER_LABEL, VALUE_LABELS, default_model
    from unstructured2graph import gliner2_backend
    from unstructured2graph.gliner2_observer import GLiNER2Observer

    model = model or default_model()
    built = documents(chats)
    sample = [document for _, document in built]
    observer = GLiNER2Observer(window_budget=windows)
    observer.extract(model, sample[:1])  # load the checkpoint before timing

    started = time.perf_counter()
    extraction = observer.extract(model, sample)
    seconds = time.perf_counter() - started
    mentions, edges = extraction.mentions, extraction.edges
    session_of = [session_id for session_id, _ in built]

    observe_seconds = None
    if observe:
        opened = replace(
            model, relation_types=tuple(replace(r, start_labels=(), end_labels=()) for r in model.relation_types)
        )
        started = time.perf_counter()
        observer.extract(opened, sample)
        observe_seconds = (time.perf_counter() - started) / max(extraction.windows, 1)

    with_fact = {(h.session, h.turn) for _, h, _ in edges if h.role == "user" and h.turn is not None}
    count = max(len(mentions), 1)
    metrics = {
        "windows": extraction.windows,
        "seconds_per_window": round(seconds / max(extraction.windows, 1), 3),
        "observe_seconds_per_window": round(observe_seconds, 3) if observe_seconds is not None else None,
        "mentions": len(mentions),
        "relations": len(edges),
        "mentions_per_window": round(len(mentions) / max(extraction.windows, 1), 2),
        "relations_per_window": round(len(edges) / max(extraction.windows, 1), 2),
        "catch_all_share": round(sum(m.label in CATCH_ALL_LABELS for m in mentions) / count, 4),
        "value_share": round(sum(m.label in VALUE_LABELS for m in mentions) / count, 4),
        "user_headed_relations": sum(h.label == USER_LABEL for _, h, _ in edges),
        "coverage": round(len(with_fact) / max(len(extraction.user_turns), 1), 4),
        "entity_types": dict(Counter(m.label for m in mentions).most_common()),
        "relation_types": dict(Counter(relation for relation, _, _ in edges).most_common()),
    }
    fingerprints = {
        "mentions": sorted({_print(session_of[m.session], m.start, m.end, m.label) for m in mentions}),
        "relations": sorted({_print(session_of[h.session], relation, h.start, t.start) for relation, h, t in edges}),
    }
    run_meta = {
        "model_checkpoint": "fastino/gliner2.5-base-v1",
        "decoder": gliner2_backend.DECODER,
        "candidate_cap": gliner2_backend.DEFAULT_CANDIDATE_CAP,
        "graph_model": "hygm default" if model == default_model() else "custom",
        "commit": _commit(),
        **(meta or {}),
    }
    return KgRun(meta=run_meta, metrics=metrics, probes=probe(model), fingerprints=fingerprints)


def probe(model: HygmModel) -> dict[str, bool | None]:
    """Whether each known fact is extracted under `model`; None for all when LongMemEval isn't cached."""
    from unstructured2graph import Ontology
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    from .convert.longmemeval import haystack_path

    path = haystack_path()
    if not path.exists():
        return {name: None for name, _, _ in PROBES}
    turns = [
        turn["content"]
        for record in json.loads(path.read_text())
        for session in record["haystack_sessions"]
        for turn in session
        if turn["role"] == "user"
    ]
    backend = GLiNER2Backend(ontology=Ontology.from_model(model))
    found: dict[str, bool | None] = {}
    for name, pattern, (kind, label, text) in PROBES:
        turn = next((t for t in turns if re.search(pattern, t)), None)
        if turn is None:
            found[name] = None
            continue
        result = backend.engine.extract(turn, backend._schema, config=backend._config)
        by_id = {e.id: e for e in result.entities}
        if kind == "entity":
            found[name] = any(e.type == label and text in e.text for e in result.entities)
        else:
            found[name] = any(
                by_id[r.head].type == label and text in by_id[r.tail].text
                for r in result.relations
                if r.head in by_id and r.tail in by_id
            )
    return found


def compare(run: KgRun, baseline: KgRun) -> list[str]:
    """How `run` differs from `baseline`: the headline metrics side by side, then what it kept."""
    lines = [f"{'':28}{'baseline':>12}{'this run':>12}"]
    for key in (
        "seconds_per_window",
        "observe_seconds_per_window",
        "mentions_per_window",
        "relations_per_window",
        "catch_all_share",
        "value_share",
        "user_headed_relations",
        "coverage",
    ):
        lines.append(f"{key:28}{_cell(baseline.metrics.get(key)):>12}{_cell(run.metrics.get(key)):>12}")
    for kind in ("mentions", "relations"):
        old, new = set(baseline.fingerprints.get(kind, [])), set(run.fingerprints.get(kind, []))
        kept = len(old & new) / max(len(old), 1)
        lines.append(f"{kind} kept from baseline: {kept:.0%} ({len(old & new)}/{len(old)}), new: {len(new - old)}")
    for name, value in run.probes.items():
        lines.append(f"probe {name}: baseline {baseline.probes.get(name)}, this run {value}")
    if run.meta.get("windows_requested") != baseline.meta.get("windows_requested") or run.meta.get(
        "chats"
    ) != baseline.meta.get("chats"):
        lines.append("WARNING: different chats or window budget than the baseline; fingerprints don't compare")
    return lines


def render(run: KgRun) -> str:
    m = run.metrics
    lines = [
        f"{m['windows']} windows | {m['seconds_per_window']}s/window"
        + (f", open observe {m['observe_seconds_per_window']}s/window" if m["observe_seconds_per_window"] else ""),
        f"mentions {m['mentions']} ({m['mentions_per_window']}/window), relations {m['relations']} "
        f"({m['relations_per_window']}/window), User-headed {m['user_headed_relations']}",
        f"catch-all share {m['catch_all_share']}, value share {m['value_share']}, coverage {m['coverage']}",
        f"entity types {m['entity_types']}",
        f"relation types {m['relation_types']}",
        f"probes {run.probes}",
    ]
    return "\n".join(lines)


def save(run: KgRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(run), indent=1) + "\n", encoding="utf-8")
    return path


def load(path: Path) -> KgRun:
    return KgRun(**json.loads(path.read_text(encoding="utf-8")))


def _print(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _cell(value: Any) -> str:
    return "-" if value is None else str(value)


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
