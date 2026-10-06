"""GLiNER2-based ExtractionBackend: local, LLM-free entity/relation extraction.

Uses GLiNER2 (https://github.com/fastino-ai/GLiNER2, pip package `gliner2`),
a small open-source model that runs entirely locally (no GPU required, no
network calls, no API key) and does joint entity+relation extraction. It is
the project's default extraction backend. `gliner2` is imported only when a
GLiNER2Backend is constructed, so `import unstructured2graph` stays light.

gliner2[local] pins transformers<5, below the CVE-2026-1839 floor the
workspace used to hold; restoring it is tracked in #443.

The typed relation model (map #344) this backend implements:

- Extraction runs on gliner2's joint path (`gliner2.joint_ie`), where a
  relation's start/end labels are enforced during decoding (#345). The coarse
  `Schema.relations()` builder writes blank head/tail types and was never
  constrained at all.
- One window per segment (a conversation turn), split only past
  `chunk_size` words (#352): a window spanning turns puts a tail from one
  speaker on a head from the other.
- Every mention goes through a mention resolver before it gets an identity
  (#358): the user's own mentions bind to (:User {user_id}), a third party
  typed User is re-typed Person, and first person in the assistant's mouth
  is dropped.
- Identity is per entity type (#346, #361): `global`, `chunk` or `span`.
- A relationship carries the source chunk, the source turn's timestamp as
  `valid_at` (#364) and the model's confidence. Self-loops left after identity
  resolution are dropped (#355).
- Provenance is exact, from the spans the model returns (#392): a segment's
  `source_id` (the turn) is recorded in `MENTIONED_IN.sources` for every
  entity it mentions, and each relationship extracted from it is its own edge
  carrying `source_id`, the speaker as `role`, and the sentence covering both
  endpoints as `text`.
"""

import asyncio
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from hygm import PERSON_LABEL, USER_LABEL, VALUE_LABELS, require_valid_identifier
from memgraph_toolbox.api.memgraph import Memgraph

from .memgraph import Endpoint, create_nodes_from_list, link_mentions, upsert_extracted_relationships
from .ontology import DEFAULT_ONTOLOGY, Ontology

if TYPE_CHECKING:
    from .loaders import Chunk, Segment

logger = logging.getLogger(__name__)

#: gliner2 2.0.0 cuts relation candidate pairs to `relation_pair_cap` (default
#: 128) and then `max_edges_per_type` (256), breaking ties between equally
#: scored pairs alphabetically by entity type. Every typed copy of a span pair
#: ties, so at the defaults `User`, sorting last, loses edges: ~4% on a
#: constrained schema, all of them on a permissive one past 11 types (#371).
DEFAULT_CANDIDATE_CAP = 4096

#: Surfaces that are the speaker themself. `user` is here because every user
#: turn carries a "user: " role prefix, which the model types User.
FIRST_PERSON = frozenset({"i", "me", "my", "myself", "mine", "user", "i'm", "i've"})

#: gliner2's own word pattern (gliner2.processing.word_splitter.WhitespaceTokenSplitter,
#: copied because it is private). Windows are sized in the words the model
#: counts -- punctuation is a word of its own -- so `chunk_size` means what
#: gliner2's chunk_size means; sized in whitespace words, a long turn's windows
#: run past the encoder's 512 positions and lose mentions (#352).
_WORD = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass(frozen=True)
class Mention:
    """One entity span GLiNER2 returned, in chunk coordinates."""

    entity_type: str
    text: str
    start: int
    end: int
    confidence: float | None


@dataclass(frozen=True)
class Resolution:
    """What a mention resolver decided for one mention.

    action:
        keep       -- an ordinary entity of `entity_type`
        bind_user  -- the chunk's own user: (:User {user_id: chunk.user_id})
        drop       -- not written, and neither is any relationship touching it
    """

    action: Literal["keep", "bind_user", "drop"]
    entity_type: str | None = None


KEEP = Resolution("keep")
BIND_USER = Resolution("bind_user")
DROP = Resolution("drop")

#: (mention, the segment it sits in or None, the chunk) -> Resolution.
MentionResolver = Callable[[Mention, "Segment | None", "Chunk"], Resolution]


def resolve_user_mentions(mention: Mention, segment: "Segment | None", chunk: "Chunk") -> Resolution:
    """The default mention resolver: decide which `User` mentions are the user (#358).

    A User mention is the user iff its surface is first person AND it sits in
    a user turn -- or it is "you" in an assistant turn. Measured over #350's
    sessions the two halves catch disjoint errors, so only the conjunction
    scores zero both ways. Otherwise:

    - a first-person mention in any other turn, "you" outside an assistant
      turn, and the literal "assistant" are dropped: a first-person mention in
      the assistant's mouth is not the user asserting anything;
    - anything else typed User is a third party, re-typed Person.

    A `Person` mention whose surface is a speaker pronoun ("I", "you",
    "assistant") gets the same rule: the model types pronouns `Person` as well
    as `User`, and left alone they merged, under global identity, into
    `Person:'you'` / `Person:'I'` hubs spanning a hundred unrelated sessions,
    heading the user's own facts.

    Mentions of other types, and every mention in a chunk with no `user_id`,
    are kept as extracted.
    """
    if chunk.user_id is None or mention.entity_type not in (USER_LABEL, PERSON_LABEL):
        return KEEP
    surface = _normalize_text(mention.text)
    speaker_pronoun = surface in FIRST_PERSON or surface in {"you", "assistant"}
    if mention.entity_type == PERSON_LABEL and not speaker_pronoun:
        return KEEP
    role = segment.role if segment is not None else None
    if (role == "user" and surface in FIRST_PERSON) or (role == "assistant" and surface == "you"):
        return BIND_USER
    if speaker_pronoun:
        return DROP
    return Resolution("keep", PERSON_LABEL)


@dataclass
class GLiNER2Stats:
    """Running counts across every chunk this backend ingested. Reporting only."""

    windows: int = 0
    #: A window whose decoding found no feasible solution. Unreachable under
    #: domain/range and cardinality constraints (#350, 0/109), so nonzero is a bug.
    infeasible_windows: int = 0
    #: Main-pass mentions dropped because the value-only pass typed an
    #: overlapping span as a value type.
    value_spans_claimed: int = 0
    mentions: int = 0
    mentions_bound_to_user: int = 0
    mentions_retyped: int = 0
    mentions_dropped: int = 0
    relations_written: int = 0
    self_loops_dropped: int = 0


@dataclass(frozen=True)
class _Window:
    start: int
    end: int
    segment: "Segment | None"


@dataclass
class _Extracted:
    mentions: list[tuple[Mention, "Segment | None"]] = field(default_factory=list)
    # (relation type, head mention index, tail mention index, confidence)
    relations: list[tuple[str, int, int, float | None]] = field(default_factory=list)
    infeasible: int = 0
    windows: int = 0
    value_spans_claimed: int = 0


@dataclass(frozen=True)
class _Span:
    """One entity of a window, in window coordinates, whichever pass found it."""

    id: str
    type: str
    text: str
    start: int
    end: int
    confidence: float | None


@dataclass(frozen=True)
class _Edge:
    type: str
    head: str
    tail: str
    confidence: float | None


def _with_value_pass(main: Any, values: Any) -> tuple[list[_Span], list[_Edge], int]:
    """A window's main-pass result with the value-only pass's value spans taking precedence (#386).

    In one pass every label competes for a span, and in a large learned
    vocabulary the value types lose theirs to domain types (`25:50` typed
    Food). So a span the value pass types as a value type wins: a main-pass
    mention overlapping it with another type is dropped, with its relations,
    and the value pass's relations into it are added. A value-pass head on the
    exact span of a kept main mention is that mention.

    Returns:
        The window's spans and edges, and how many main mentions were claimed.
    """
    claims = [e for e in values.entities if e.type in VALUE_LABELS]

    def claimed(e: Any) -> bool:
        return any(e.start < v.end and v.start < e.end and e.type != v.type for v in claims)

    spans = [_Span(e.id, e.type, e.text, e.start, e.end, e.confidence) for e in main.entities if not claimed(e)]
    dropped = len(main.entities) - len(spans)
    kept = {span.id for span in spans}
    edges = [_Edge(r.type, r.head, r.tail, r.confidence) for r in main.relations if r.head in kept and r.tail in kept]

    at = {(span.start, span.end): span for span in spans}
    alias: dict[str, str] = {}
    for e in values.entities:
        same = at.get((e.start, e.end))
        if same is not None and (same.type == e.type or e.type not in VALUE_LABELS):
            alias[e.id] = same.id
            continue
        span = _Span(f"value:{e.id}", e.type, e.text, e.start, e.end, e.confidence)
        spans.append(span)
        at.setdefault((e.start, e.end), span)
        alias[e.id] = span.id
    seen = {(edge.type, edge.head, edge.tail) for edge in edges}
    for r in values.relations:
        if r.head in alias and r.tail in alias and (r.type, alias[r.head], alias[r.tail]) not in seen:
            edges.append(_Edge(r.type, alias[r.head], alias[r.tail], r.confidence))
            seen.add((r.type, alias[r.head], alias[r.tail]))
    return spans, edges, dropped


def _entity_id(chunk_hash: str, entity_type: str, normalized_text: str, identity: str, span: tuple[int, int]) -> str:
    """
    A stable node key for one mention, scoped by its type's identity (#346, #361):

    - global: (normalized text) -- one node across every chunk, so a name
      mentioned in two sessions is one node. The type is left out so that
      merging two types in a learned model (#434) only relabels: a name keeps
      its node whichever type it is extracted as later, and the node keeps the
      type it was first seen with (ON CREATE SET), as it keeps the first
      chunk's hash as `file_path`; MENTIONED_IN, written per chunk at ingest,
      carries the rest.
    - chunk: (chunk hash, type, normalized text) -- one node per chunk.
    - span: (chunk hash, type, span offsets) -- one node per mention, for
      values, where every "3" in a session is a different fact.
    """
    if identity == "global":
        key = f"global|{normalized_text}"
    elif identity == "span":
        key = f"{chunk_hash}|{entity_type}|{span[0]}:{span[1]}"
    else:
        key = f"{chunk_hash}|{entity_type}|{normalized_text}"
    return hashlib.sha256(key.encode()).hexdigest()


def _word_windows(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character ranges of `size`-word windows over text[start:end], consecutive ones sharing `overlap` words."""
    words = [(m.start(), m.end()) for m in _WORD.finditer(text, start, end)]  # already absolute offsets
    if len(words) <= size:
        return [(start, end)]
    step = max(size - overlap, 1)
    ranges = []
    for first in range(0, len(words), step):
        last = min(first + size, len(words)) - 1
        ranges.append((words[first][0], words[last][1]))
        if last == len(words) - 1:
            break
    return ranges


#: A sentence ends at terminal punctuation followed by whitespace, or at a line break.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")

#: The longest source text a relationship carries; the full turn is one hop
#: away through its source_id.
SOURCE_TEXT_CAP = 300


def _source_text(text: str, bounds: tuple[int, int], head: tuple[int, int], tail: tuple[int, int]) -> str:
    """The sentence or sentences of text[bounds] covering both endpoint spans, at most SOURCE_TEXT_CAP characters.

    Past the cap the text is trimmed around the spans, keeping both whole.
    """
    lo, hi = min(head[0], tail[0]), max(head[1], tail[1])
    start, end = bounds
    for boundary in _SENTENCE_END.finditer(text, bounds[0], bounds[1]):
        if boundary.end() <= lo:
            start = boundary.end()
        elif boundary.start() >= hi:
            end = boundary.start()
            break
    if end - start > SOURCE_TEXT_CAP:
        slack = max(SOURCE_TEXT_CAP - (hi - lo), 0)
        start = max(start, lo - slack // 2)
        end = min(end, max(hi, start + SOURCE_TEXT_CAP))
    return text[start:end].strip()


class GLiNER2Backend:
    """Local, LLM-free ExtractionBackend backed by a GLiNER2 model.

    Entity types come from `ontology.entity_types`; relation types, with the
    entity types each may connect, from `ontology.relation_types`, written as
    per-relation-type Cypher relationship types (e.g. `:works_for`). The same
    start/end labels constrain decoding here and are checked again post hoc by
    memgraph.enforce_relation_domain_range (#348): one specification, two
    compilations, so a post-hoc violation on this backend's output is a bug.

    The schema is compiled once, here, and held: gliner2 caches compiled
    schemas under the schema object's memory address, so a schema rebuilt per
    call can collide with a dead one's cache entry and silently extract
    against the wrong vocabulary (#365).

    When the ontology has relations into value types, every window is also
    run through a second, value-only schema whose value spans take precedence
    (#386; see `_with_value_pass`).

    Relationships whose endpoint is the user are written onto
    (:User {user_id}), which this backend never creates: the caller (e.g.
    sessions-graph, which owns (:User)) must MERGE it before ingesting.
    """

    def __init__(
        self,
        model_name: str = "fastino/gliner2.5-base-v1",
        ontology: Ontology | None = None,
        workspace: str = "gliner2",
        model: Any | None = None,
        entity_confidence_threshold: float | None = None,
        relation_confidence_threshold: float | None = None,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        mention_resolver: MentionResolver = resolve_user_mentions,
        candidate_cap: int = DEFAULT_CANDIDATE_CAP,
    ) -> None:
        """
        Args:
            model_name: A GLiNER2 checkpoint. Ignored if `model` is given.
            ontology: Entity and relation vocabulary to extract against.
                Defaults to DEFAULT_ONTOLOGY, which is entity-only -- pass one
                with `relation_types` to also extract relations.
            workspace: Memgraph label entity nodes are written under.
                f-string-interpolated into every Cypher query this backend's
                data touches, so it must be a valid identifier.
            model: A pre-loaded gliner2 model (an AutoExtractor, wrapped in a
                `gliner2.joint_ie.JointIEEngine` here), or an object exposing
                the engine's own create_schema()/compile_schema()/extract()
                -- a JointIEEngine, or a test fake, which needs no `gliner2`
                installed at all.
            entity_confidence_threshold: Drop mentions below this confidence.
                None (default) keeps everything the model returns.
            relation_confidence_threshold: Drop relations below this confidence.
                None (default) keeps everything the model returns.
            chunk_size: Maximum words per extraction window. A segment (turn)
                longer than this is split into overlapping windows; past the
                encoder's 512 positions mentions and distinct claims halve (#352).
            chunk_overlap: Words shared by consecutive windows of one long segment.
            mention_resolver: Decides, per mention, keep / re-type / bind to
                the chunk's user / drop, before identity. Defaults to
                resolve_user_mentions (#358).
            candidate_cap: gliner2's relation_pair_cap and max_edges_per_type;
                see DEFAULT_CANDIDATE_CAP.

        Raises:
            ValueError: if `workspace` isn't a valid Cypher identifier.
        """
        require_valid_identifier(workspace, "workspace")
        self._workspace = workspace
        self.ontology = ontology if ontology is not None else DEFAULT_ONTOLOGY
        self.engine = self._engine(model, model_name)
        self.entity_confidence_threshold = entity_confidence_threshold
        self.relation_confidence_threshold = relation_confidence_threshold
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._resolve = mention_resolver
        self._identity = {t.label: t.identity for t in self.ontology.entity_types}
        self._config = self._make_config(candidate_cap)
        self._schema = self.engine.compile_schema(self._build_schema())
        value_schema = self._build_value_schema()
        self._value_schema = self.engine.compile_schema(value_schema) if value_schema is not None else None
        self.stats = GLiNER2Stats()

    @staticmethod
    def _engine(model: Any | None, model_name: str) -> Any:
        if model is not None and all(hasattr(model, a) for a in ("create_schema", "compile_schema", "extract")):
            return model
        from gliner2 import AutoExtractor
        from gliner2.joint_ie import JointIEEngine

        return JointIEEngine(model if model is not None else AutoExtractor.from_pretrained(model_name))

    @staticmethod
    def _make_config(candidate_cap: int) -> Any:
        from gliner2.joint_ie import JointIEConfig

        return JointIEConfig(
            include_spans=True,
            include_confidence=True,
            relation_pair_cap=candidate_cap,
            max_edges_per_type=candidate_cap,
        )

    def _build_schema(self) -> Any:
        """The ontology as a JointSchema. An unconstrained endpoint becomes every declared
        entity type, since JointSchema rejects an empty head/tail (#345)."""
        model = self.ontology.model
        schema = self.engine.create_schema()
        for entity_type in self.ontology.entity_types:
            schema = schema.entity(entity_type.label, entity_type.description or None)
        for relation in self.ontology.relation_types:
            # No description: the joint compiler drops relation descriptions, so
            # the name is the only steering surface (#360).
            schema = schema.relation(
                relation.label, model.endpoint_labels(relation, "start"), model.endpoint_labels(relation, "end")
            )
        return schema

    def _build_value_schema(self) -> Any | None:
        """The value-only schema, or None when no relation points only into value types.

        It holds the value types, User and Person, the value relations, and
        the head types those relations declare; a relation with an
        unconstrained head starts at User or Person here, so the value types
        never compete with the whole vocabulary again.
        """
        declared = {t.label: t for t in self.ontology.entity_types}
        relations = [r for r in self.ontology.relation_types if r.end_labels and set(r.end_labels) <= set(VALUE_LABELS)]
        if not relations:
            return None
        people = tuple(label for label in (USER_LABEL, PERSON_LABEL) if label in declared)
        heads = {r.label: tuple(r.start_labels) or people for r in relations}
        labels = dict.fromkeys(
            [*people, *(label for r in relations for label in r.end_labels), *(h for hs in heads.values() for h in hs)]
        )
        schema = self.engine.create_schema()
        for label in labels:
            if label in declared:
                schema = schema.entity(label, declared[label].description or None)
        for relation in relations:
            if heads[relation.label]:
                schema = schema.relation(relation.label, heads[relation.label], tuple(relation.end_labels))
        return schema

    @property
    def workspace_label(self) -> str:
        return self._workspace

    def _windows(self, chunk: "Chunk") -> list[_Window]:
        spans = [(s.start, s.end, s) for s in chunk.segments] or [(0, len(chunk.text), None)]
        windows = []
        for start, end, segment in spans:
            for lo, hi in _word_windows(chunk.text, start, end, self._chunk_size, self._chunk_overlap):
                if chunk.text[lo:hi].strip():
                    windows.append(_Window(lo, hi, segment))
        return windows

    def _extract_sync(self, chunk: "Chunk") -> _Extracted:
        """
        Run the joint extractor over each window of `chunk` and map everything
        back to chunk coordinates. Synchronous and CPU/GPU-bound (GLiNER2 has no
        async API), so aingest_chunk() runs it via asyncio.to_thread.

        A mention found by two overlapping windows of one long segment is one
        mention. Relations never cross windows: their endpoints are the
        window's own entity ids.
        """
        extracted = _Extracted()
        index_of: dict[tuple[str, int, int], int] = {}
        for window in self._windows(chunk):
            text = chunk.text[window.start : window.end]
            result = self.engine.extract(text, self._schema, config=self._config)
            extracted.windows += 1
            if not getattr(result, "feasible", True):
                extracted.infeasible += 1
                logger.error(f"GLiNER2 decoding was infeasible for a window of chunk {chunk.hash[:12]}: a bug (#355)")
            entities: list[Any] = list(result.entities)
            relations: list[Any] = list(result.relations)
            if self._value_schema is not None:
                values = self.engine.extract(text, self._value_schema, config=self._config)
                entities, relations, claimed = _with_value_pass(result, values)
                extracted.value_spans_claimed += claimed
            local: dict[str, int] = {}
            for entity in entities:
                start, end = entity.start + window.start, entity.end + window.start
                key = (entity.type, start, end)
                if key not in index_of:
                    index_of[key] = len(extracted.mentions)
                    extracted.mentions.append(
                        (Mention(entity.type, entity.text, start, end, entity.confidence), window.segment)
                    )
                local[entity.id] = index_of[key]
            for relation in relations:
                if relation.head in local and relation.tail in local:
                    extracted.relations.append(
                        (relation.type, local[relation.head], local[relation.tail], relation.confidence)
                    )
        return extracted

    def _endpoint(
        self, mention: Mention, resolution: Resolution, chunk: "Chunk"
    ) -> tuple[Endpoint, dict | None] | None:
        """The node a mention resolves to, and the entity node to write for it (None for the user)."""
        if resolution.action == "drop":
            return None
        if resolution.action == "bind_user":
            return Endpoint(USER_LABEL, "user_id", str(chunk.user_id)), None
        entity_type = resolution.entity_type or mention.entity_type
        if entity_type not in self._identity:
            return None
        normalized = _normalize_text(mention.text)
        if not normalized:
            return None
        identity = self._identity[entity_type]
        if identity == "global" and mention.text == mention.text.lower():
            # Global identity is for names. An all-lowercase mention of a
            # global type is a generic noun ("home", "area", "city"), and
            # merging those across sessions built hubs joining ~50 unrelated
            # conversations each, so it stays per chunk.
            identity = "chunk"
        entity_id = _entity_id(chunk.hash, entity_type, normalized, identity, (mention.start, mention.end))
        node = {"entity_id": entity_id, "entity_type": entity_type, "text": mention.text, "file_path": chunk.hash}
        return Endpoint(self._workspace, "entity_id", entity_id), node

    async def aingest_chunk(self, memgraph: Memgraph, chunk: "Chunk") -> None:
        extracted = await asyncio.to_thread(self._extract_sync, chunk)
        self.stats.windows += extracted.windows
        self.stats.infeasible_windows += extracted.infeasible
        self.stats.value_spans_claimed += extracted.value_spans_claimed

        endpoints: list[Endpoint | None] = []
        nodes: dict[str, dict[str, Any]] = {}
        sources: dict[str, set[str]] = {}
        for mention, segment in extracted.mentions:
            self.stats.mentions += 1
            if (
                self.entity_confidence_threshold is not None
                and mention.confidence is not None
                and mention.confidence < self.entity_confidence_threshold
            ):
                endpoints.append(None)
                continue
            resolution = self._resolve(mention, segment, chunk)
            if resolution.action == "drop":
                self.stats.mentions_dropped += 1
            elif resolution.action == "bind_user":
                self.stats.mentions_bound_to_user += 1
            elif resolution.entity_type not in (None, mention.entity_type):
                self.stats.mentions_retyped += 1
            resolved = self._endpoint(mention, resolution, chunk)
            endpoints.append(resolved[0] if resolved else None)
            if resolved and resolved[1] is not None:
                entity_id = resolved[1]["entity_id"]
                nodes.setdefault(entity_id, resolved[1])
                mentioned_in = sources.setdefault(entity_id, set())
                if segment is not None and segment.source_id is not None:
                    mentioned_in.add(segment.source_id)

        if nodes:
            create_nodes_from_list(memgraph, list(nodes.values()), self._workspace, 100, merge_key="entity_id")
            link_mentions(
                memgraph,
                self._workspace,
                "entity_id",
                chunk.hash,
                {entity_id: list(ids) for entity_id, ids in sources.items()},
            )

        merged: dict[tuple[str, Endpoint, Endpoint, str | None], dict[str, Any]] = {}
        for relation_type, head_index, tail_index, confidence in extracted.relations:
            head, tail = endpoints[head_index], endpoints[tail_index]
            if head is None or tail is None:
                continue
            if (
                self.relation_confidence_threshold is not None
                and confidence is not None
                and confidence < self.relation_confidence_threshold
            ):
                continue
            if head == tail:
                # allow_self can't see this: decoding sees two spans, identity
                # resolution then makes them one node (#355).
                self.stats.self_loops_dropped += 1
                continue
            head_mention, segment = extracted.mentions[head_index]
            tail_mention = extracted.mentions[tail_index][0]
            # Relations never cross windows, so both endpoints share the head's segment.
            when = segment.valid_at if segment is not None else None
            source_id = segment.source_id if segment is not None else None
            key = (relation_type, head, tail, source_id)
            text = _source_text(
                chunk.text,
                (segment.start, segment.end) if segment is not None else (0, len(chunk.text)),
                (head_mention.start, head_mention.end),
                (tail_mention.start, tail_mention.end),
            )
            previous = merged.get(key)
            if previous is None:
                merged[key] = {
                    "type": relation_type,
                    "head": head,
                    "tail": tail,
                    "chunk": chunk.hash,
                    "source_id": source_id,
                    "valid_at": when,
                    "confidence": confidence,
                    "text": text,
                    "role": segment.role if segment is not None else None,
                }
                continue
            # The same fact twice under one key: when and where it was first said, how sure the model ever was.
            if when is not None and (previous["valid_at"] is None or when < previous["valid_at"]):
                previous.update(valid_at=when, text=text, role=segment.role if segment is not None else None)
            if confidence is not None and (previous["confidence"] is None or confidence > previous["confidence"]):
                previous["confidence"] = confidence

        if merged:
            upsert_extracted_relationships(memgraph, list(merged.values()))
            self.stats.relations_written += len(merged)
