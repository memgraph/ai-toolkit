"""hygm's Observer over GLiNER2: what a candidate model actually extracts from a sample.

Derivation (``hygm.LlmRecommendationStrategy``) observes its candidate before
pruning it: every type and relation is run over a sample of the user's
sessions, and the relations' real endpoints are chosen from the type pairs
they fired on. This runs the same extraction a GLiNER2Backend ingests with --
one window per turn, the value-only pass, the user-mention resolver -- and
tallies the result instead of writing it.

Both read a sample of windows, not every one: an open-endpoint pass costs
~5 s a window on CPU (relation decoding over every type pair), so a run over
whole sessions took hours, and the tables and the gate's two numbers are
statistics a spread sample estimates. The sample depends only on the
documents, never the schema, so two models are measured on the same windows.

Not imported by unstructured2graph/__init__.py, like gliner2_backend: importing
it loads nothing heavy, constructing a backend does.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hygm import USER_LABEL, HygmModel

from .gliner2_backend import DEFAULT_CANDIDATE_CAP, GLiNER2Backend, _normalize_text
from .loaders import Chunk
from .ontology import Ontology

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .loaders import Document

#: Per relation, the endpoint pairs reported, and per pair, the example edges.
_PAIRS, _EXAMPLES = 8, 4
#: Per type, the most frequent surfaces reported.
_TOP_TEXTS = 10
#: Windows one observe() or measure() call extracts at most.
DEFAULT_WINDOW_BUDGET = 100


@dataclass(frozen=True)
class ResolvedMention:
    """One mention as the backend would write it: in which session and turn, typed as resolved."""

    session: int
    turn: int | None
    role: str | None
    label: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Extraction:
    """What a model extracted from a sample's windows, resolved as the backend would write it.

    Attributes:
        mentions: Every kept mention.
        edges: Every relation between two kept, distinct mentions, as (type, head, tail).
        user_turns: The user turns the windows cover, as (session, turn).
        windows: How many windows were extracted.
    """

    mentions: list[ResolvedMention]
    edges: list[tuple[str, ResolvedMention, ResolvedMention]]
    user_turns: set[tuple[int, int]]
    windows: int


@dataclass(frozen=True)
class Measurement:
    """What the adoption gate compares between two models over one held-out sample (#435).

    Attributes:
        catch_all_share: Mentions typed into a catch-all, over all mentions:
            how much the model fails to name.
        coverage: User turns with at least one typed relation, over all user
            turns: guards against a model that trades facts for tidy labels.
        spans: Each mention's type by (session, start, end), for agreement
            between two models' typings.
        mentions: Mentions counted.
        user_turns: User turns the sampled windows cover.
    """

    catch_all_share: float
    coverage: float
    spans: dict[tuple[int, int, int], str]
    mentions: int
    user_turns: int


class GLiNER2Observer:
    """Implements ``hygm.Observer`` for GLiNER2 over ``Document`` samples.

    The checkpoint loads on the first observe() and is reused by every later
    one; each call compiles the candidate's own schema.

    Args:
        model_name: A GLiNER2 checkpoint. Ignored if `model` is given.
        model: A pre-loaded engine or model, as for GLiNER2Backend.
        candidate_cap: gliner2's relation_pair_cap and max_edges_per_type. A
            permissive schema needs the high default most: every span ties
            once per type, and the cut at the library default drops every
            User edge (#371).
        window_budget: The most windows one call extracts, evenly spaced over
            the sample's sessions in order; None reads every window.
    """

    def __init__(
        self,
        model_name: str = "fastino/gliner2.5-base-v1",
        model: Any | None = None,
        candidate_cap: int = DEFAULT_CANDIDATE_CAP,
        window_budget: int | None = DEFAULT_WINDOW_BUDGET,
    ) -> None:
        self._model_name = model_name
        self._model = model
        self._candidate_cap = candidate_cap
        self._window_budget = window_budget

    def observe(self, model: HygmModel, sample: Sequence[Document]) -> dict[str, Any]:
        """Observation tables for `model` over `sample`, one session per Document.

        Returns:
            ``{"relations": {name: {"edges", "pairs"}}, "types": {label: {...}}}``
            for every relation and type `model` declares, in the shape
            ``hygm.Observer`` documents. Mentions are counted under the type
            the user resolver settles on (the user's own mentions as User,
            third parties as Person); dropped mentions aren't counted.
        """
        extraction = self.extract(model, sample)
        mentions, edges = extraction.mentions, extraction.edges
        return {
            "relations": _relation_tables(model, [(r, h.label, h.text, t.label, t.text) for r, h, t in edges]),
            "types": _type_tables(model, [(m.session, m.label, m.text) for m in mentions]),
        }

    def measure(self, model: HygmModel, sample: Sequence[Document], catch_alls: Sequence[str]) -> Measurement:
        """Extract `sample` under `model` and score it for the adoption gate.

        Args:
            catch_alls: The labels counted as catch-alls (``hygm.CATCH_ALL_LABELS``).
        """
        extraction = self.extract(model, sample)
        mentions, edges, user_turns = extraction.mentions, extraction.edges, extraction.user_turns
        with_fact = {(h.session, h.turn) for _, h, _ in edges if h.role == "user" and h.turn is not None}
        return Measurement(
            catch_all_share=sum(m.label in catch_alls for m in mentions) / max(len(mentions), 1),
            coverage=len(with_fact) / max(len(user_turns), 1),
            spans={(m.session, m.start, m.end): m.label for m in mentions},
            mentions=len(mentions),
            user_turns=len(user_turns),
        )

    def extract(self, model: HygmModel, sample: Sequence[Document]) -> Extraction:
        """Extract the sampled windows of `sample` under `model`, resolved as the backend writes them.

        The windows are at most `window_budget`, evenly spaced over the
        sessions in order; a session is the document at that index.
        """
        backend = GLiNER2Backend(
            model_name=self._model_name,
            ontology=Ontology.from_model(model),
            model=self._model,
            candidate_cap=self._candidate_cap,
        )
        self._model = backend.engine
        mentions: list[ResolvedMention] = []
        edges: list[tuple[str, ResolvedMention, ResolvedMention]] = []
        chunks = [
            Chunk(
                text=document.text,
                hash=hashlib.sha256(document.text.encode()).hexdigest(),
                segments=document.segments,
                user_id=document.user_id,
            )
            for document in sample
        ]
        chosen = _spread(
            [(session, window) for session, chunk in enumerate(chunks) for window in backend._windows(chunk)],
            self._window_budget,
        )
        user_turns = {
            (session, sample[session].segments.index(window.segment))
            for session, window in chosen
            if window.segment is not None and window.segment.role == "user"
        }
        for session, document in enumerate(sample):
            chunk = chunks[session]
            windows = [window for at, window in chosen if at == session]
            if not windows:
                continue
            extracted = backend._extract_sync(chunk, windows)
            resolved: list[ResolvedMention | None] = []
            for mention, segment in extracted.mentions:
                resolution = backend._resolve(mention, segment, chunk)
                if resolution.action == "drop":
                    resolved.append(None)
                    continue
                label = (
                    USER_LABEL if resolution.action == "bind_user" else resolution.entity_type or mention.entity_type
                )
                turn = document.segments.index(segment) if segment is not None else None
                role = segment.role if segment is not None else None
                found = ResolvedMention(session, turn, role, label, mention.text, mention.start, mention.end)
                resolved.append(found)
                mentions.append(found)
            for relation, head, tail, _ in extracted.relations:
                head_mention, tail_mention = resolved[head], resolved[tail]
                if head_mention is not None and tail_mention is not None and head != tail:
                    edges.append((relation, head_mention, tail_mention))
        return Extraction(mentions=mentions, edges=edges, user_turns=user_turns, windows=len(chosen))


def _spread(items: list[Any], budget: int | None) -> list[Any]:
    """At most `budget` of `items`, evenly spaced from first to last, in order."""
    if budget is None or len(items) <= budget:
        return items
    if budget < 2:
        return items[:budget]
    return [items[round(i * (len(items) - 1) / (budget - 1))] for i in range(budget)]


def _relation_tables(model: HygmModel, edges: list[tuple[str, str, str, str, str]]) -> dict[str, Any]:
    tables = {}
    for name in model.relation_labels():
        mine = [edge for edge in edges if edge[0] == name]
        pairs = Counter((head_type, tail_type) for _, head_type, _, tail_type, _ in mine)
        examples: dict[tuple[str, str], list[str]] = defaultdict(list)
        for _, head_type, head_text, tail_type, tail_text in mine:
            bucket = examples[(head_type, tail_type)]
            example = f"{head_text} -> {tail_text}"
            if example not in bucket and len(bucket) < _EXAMPLES:
                bucket.append(example)
        tables[name] = {
            "edges": len(mine),
            "pairs": [
                {"head": head, "tail": tail, "count": count, "examples": examples[(head, tail)]}
                for (head, tail), count in pairs.most_common(_PAIRS)
            ],
        }
    return tables


def _type_tables(model: HygmModel, mentions: list[tuple[int, str, str]]) -> dict[str, Any]:
    """Per type, the statistics prune judges identity from (#366): how often a name recurs across sessions."""
    tables = {}
    for label in model.node_labels():
        mine = [(session, text) for session, kind, text in mentions if kind == label]
        texts = Counter(_normalize_text(text) for _, text in mine)
        sessions_of: dict[str, set[int]] = defaultdict(set)
        for session, text in mine:
            sessions_of[_normalize_text(text)].add(session)
        tables[label] = {
            "mentions": len(mine),
            "distinct_texts": len(texts),
            "sessions": len({session for session, _ in mine}),
            "texts_recurring_across_sessions": round(
                sum(len(found) > 1 for found in sessions_of.values()) / max(len(texts), 1), 2
            ),
            "capitalized_share": round(sum(text[:1].isupper() for _, text in mine) / max(len(mine), 1), 2),
            "top_texts": [text for text, _ in texts.most_common(_TOP_TEXTS)],
        }
    return tables
