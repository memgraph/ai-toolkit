"""GLiNER2-based ExtractionBackend: local, LLM-free entity/relation extraction.

Uses GLiNER2 (https://github.com/fastino-ai/GLiNER2, pip package `gliner2`),
a small open-source model that runs entirely locally (no GPU required, no
network calls, no API key) and does joint entity+relation extraction. This
module is not imported by unstructured2graph/__init__.py and never imports
`gliner2` at module scope, so `import unstructured2graph` never requires the
optional `gliner2` dependency -- only constructing a GLiNER2Backend does.

gliner2 is deliberately NOT declared as a pyproject.toml extra of this
package -- gliner2[local] hard-pins transformers<5, which conflicts with
this workspace's transformers>=5.0.0rc3 floor (the fix for CVE-2026-1839,
an RCE in transformers.Trainer's checkpoint loading -- see
GHSA-69w3-r845-3855). Install it manually in your own environment:
`pip install 'gliner2[local]>=2.0.0'`. Doing so accepts that CVE's exposure
for your environment only; it never affects the workspace-managed lock or
anyone who doesn't opt in.
"""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memgraph_toolbox.api.memgraph import Memgraph

from .memgraph import create_nodes_from_list, upsert_typed_relationships
from .ontology import DEFAULT_ONTOLOGY, Ontology

if TYPE_CHECKING:
    from .loaders import Chunk

logger = logging.getLogger(__name__)

# workspace gets f-string-interpolated into every Cypher query this backend's
# entities/relations touch (create_nodes_from_list, connect_chunks_to_entities,
# promote_*_to_labels, upsert_typed_relationships) -- same restriction, and
# same duplicated-with-a-comment convention, as ontology.py's and memgraph.py's
# own _VALID_LABEL_PATTERN, since workspace is a caller-supplied constructor
# argument rather than a compile-time literal.
_VALID_LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass(frozen=True)
class ExtractedEntity:
    """One entity span GLiNER2 returned, before ontology/confidence filtering."""

    entity_type: str
    text: str
    start: int | None
    end: int | None
    confidence: float | None


@dataclass(frozen=True)
class ExtractedRelation:
    """One relation triple GLiNER2 returned, head/tail as raw spans -- not yet
    resolved to a specific extracted entity's id (see GLiNER2Backend._resolve_entity_id)."""

    relation_type: str
    head_text: str
    head_span: tuple[int, int] | None
    head_confidence: float | None
    tail_text: str
    tail_span: tuple[int, int] | None
    tail_confidence: float | None


def _entity_id(chunk_hash: str, entity_type: str, normalized_text: str) -> str:
    """
    GLiNER2 gives only raw text spans, no canonical entity id the way
    LightRAG's LLM-assigned entity_id does -- so identity is derived from
    (chunk hash, entity type, normalized text) instead. Deliberately scoped
    to the chunk, not global: connect_chunks_to_entities() joins on exact
    scalar file_path == hash equality, so a globally-merged identity would
    make file_path ambiguous across chunks. One consequence: unlike
    LightRAG, this backend does no cross-chunk coreference -- the same
    real-world entity mentioned in two different chunks becomes two
    separate Memgraph nodes.
    """
    return hashlib.sha256(f"{chunk_hash}|{entity_type}|{normalized_text}".encode()).hexdigest()


class GLiNER2Backend:
    """Local, LLM-free ExtractionBackend backed by a GLiNER2 model.

    Entity types come from `ontology.entity_types`; if `ontology` also
    carries `relation_types`, relations are extracted too and written as
    per-relation-type Cypher relationship types (e.g. `:works_for`), not a
    single generic edge type -- unlike LightRAG's free-form LLM-assigned
    entity_type (which needs a later label-promotion pass because nothing
    constrains what the LLM returns), GLiNER2's relation labels come from
    this closed, pre-validated vocabulary, so the safe Cypher identifier is
    already known at write time.

    See _entity_id() for the cross-chunk coreference limitation this
    backend has relative to LightRAGBackend.
    """

    def __init__(
        self,
        model_name: str = "fastino/gliner2.5-base-v1",
        ontology: Ontology | None = None,
        workspace: str = "gliner2",
        model: Any | None = None,
        entity_confidence_threshold: float | None = None,
        relation_confidence_threshold: float | None = None,
    ) -> None:
        """
        Args:
            model_name: A GLiNER2 model checkpoint (e.g. the 74M "small",
                200M "base", or 300M multilingual variant). Ignored if
                `model` is given.
            ontology: Entity (and optionally relation) vocabulary to extract
                against. Defaults to DEFAULT_ONTOLOGY, which is entity-only
                (there's no LightRAG-equivalent default relation vocabulary
                to mirror) -- pass one with `relation_types` to also extract
                relations.
            workspace: Memgraph label entity nodes are written under.
                Distinct from LightRAGBackend's default ("base") so the two
                backends' output doesn't collide if run against the same
                database. f-string-interpolated into every Cypher query this
                backend's data touches, so it must be a valid identifier
                (letters, digits, underscore, not starting with a digit).
            model: Pre-loaded GLiNER2 extractor (e.g. an AutoExtractor
                instance), or a test fake exposing the same
                create_schema()/extract() methods. Bypasses loading
                `model_name` and importing `gliner2` entirely -- the only
                supported way to use this class without the `gliner2`
                package installed.
            entity_confidence_threshold: Drop extracted entities below this
                confidence. None (default) keeps everything the model returns.
            relation_confidence_threshold: Drop extracted relations whose
                head/tail confidence falls below this (relations carry no
                confidence of their own -- see _extract_sync). None (default)
                keeps everything the model returns.

        Raises:
            ValueError: if `workspace` isn't a valid Cypher identifier.
            ImportError: if `model` is omitted and `gliner2` isn't installed
                (see the module docstring for why it's a manual install, not
                a pyproject.toml extra).
        """
        if not _VALID_LABEL_PATTERN.match(workspace):
            raise ValueError(
                f"Invalid workspace {workspace!r}: must be a valid identifier (letters, digits, "
                "underscore, not starting with a digit) since it's used directly as a Memgraph label"
            )

        if model is not None:
            self.model = model
        else:
            try:
                # ty can't resolve this: gliner2 is deliberately not part of
                # the workspace's managed dependency graph at all (see this
                # module's docstring), so `uv sync --all-extras` never
                # installs it -- unlike a normal optional extra, there's no
                # sync flag that would make this resolvable in CI.
                from gliner2 import AutoExtractor  # ty: ignore[unresolved-import]
            except ImportError as e:
                raise ImportError(
                    "gliner2 is required for GLiNER2Backend; install it manually with "
                    "`pip install 'gliner2[local]>=2.0.0'` (see this module's docstring for why "
                    "it's not a pyproject.toml extra)"
                ) from e
            self.model = AutoExtractor.from_pretrained(model_name)

        resolved_ontology = ontology if ontology is not None else DEFAULT_ONTOLOGY
        self._entity_schema = {t.label: t.description for t in resolved_ontology.entity_types}
        self._relation_schema = {t.label: t.description for t in resolved_ontology.relation_types}
        self._workspace = workspace
        self.entity_confidence_threshold = entity_confidence_threshold
        self.relation_confidence_threshold = relation_confidence_threshold

    @property
    def workspace_label(self) -> str:
        return self._workspace

    def _extract_sync(self, text: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """
        Synchronous, CPU/GPU-bound model call (GLiNER2 has no async API,
        unlike LightRAG's network-bound LLM call) -- run via asyncio.to_thread
        from aingest_chunk() so it doesn't block the event loop.

        One combined `model.extract()` call covers both entities and
        relations in the same forward pass: `model.create_schema().entities(
        self._entity_schema)`, `.relations(self._relation_schema)` when
        relations are configured, then `model.extract(text, schema,
        include_spans=True, include_confidence=True)`. Confirmed against
        gliner2==2.0.0 that this returns `{"entities": {...},
        "relation_extraction": {...}}` -- the same per-key shape
        extract_entities()/extract_relations() each return individually,
        just merged into one result.

        This used to be two independent calls (extract_entities() then
        extract_relations()). A relation's head/tail span only ever carries
        a text span, not a reference to a specific already-extracted entity,
        so matching them back together (see aingest_chunk()) is required
        either way -- but running two independent inference passes could
        return a relation's head/tail with a subtly different span/text for
        the same real entity than the entities pass returned, dropping a
        real relation at the matching step for no reason beyond the two
        calls disagreeing. Confirmed live: for entities the model recognizes
        in *both* its entities and relations output, a joint call returns
        identical spans for each (e.g. "Alice Johnson" at the same (0, 13)
        in both), which two independent calls did not reliably guarantee.

        This does not make every relation matchable, and callers should not
        expect it to: GLiNER2's relation task can name a head/tail span its
        entities task never surfaces at all under the configured entity
        schema (observed live: a `located_in` relation between "townhouse"
        and "Brookside neighborhood" where the entities result for that
        chunk was empty) -- joint or not, there's no extracted entity_type
        to write such an endpoint under, so aingest_chunk() still correctly
        skips-and-logs it rather than fabricating one.

        Returns:
            (entities, relations) extracted from `text`, unfiltered --
            confidence-threshold and ontology filtering happen in
            aingest_chunk().
        """
        schema = self.model.create_schema().entities(self._entity_schema)
        if self._relation_schema:
            schema = schema.relations(self._relation_schema)
        raw = self.model.extract(text, schema, include_spans=True, include_confidence=True)

        entities: list[ExtractedEntity] = []
        for entity_type, spans in raw.get("entities", {}).items():
            for span in spans:
                if isinstance(span, dict):
                    entities.append(
                        ExtractedEntity(
                            entity_type=entity_type,
                            text=span.get("text", ""),
                            start=span.get("start"),
                            end=span.get("end"),
                            confidence=span.get("confidence"),
                        )
                    )
                else:
                    entities.append(
                        ExtractedEntity(entity_type=entity_type, text=str(span), start=None, end=None, confidence=None)
                    )

        relations: list[ExtractedRelation] = []
        for relation_type, pairs in raw.get("relation_extraction", {}).items():
            for pair in pairs:
                if isinstance(pair, dict):
                    head, tail = pair.get("head"), pair.get("tail")
                elif isinstance(pair, (tuple, list)) and len(pair) == 2:
                    head, tail = {"text": pair[0]}, {"text": pair[1]}
                else:
                    continue
                head = head if isinstance(head, dict) else {"text": str(head)}
                tail = tail if isinstance(tail, dict) else {"text": str(tail)}
                relations.append(
                    ExtractedRelation(
                        relation_type=relation_type,
                        head_text=head.get("text", ""),
                        head_span=(head["start"], head["end"]) if head.get("start") is not None else None,
                        head_confidence=head.get("confidence"),
                        tail_text=tail.get("text", ""),
                        tail_span=(tail["start"], tail["end"]) if tail.get("start") is not None else None,
                        tail_confidence=tail.get("confidence"),
                    )
                )
        return entities, relations

    @staticmethod
    def _resolve_entity_id(
        span: tuple[int, int] | None,
        text: str,
        by_span: dict[tuple[int, int], str],
        by_text: dict[str, str],
    ) -> str | None:
        if span is not None and span in by_span:
            return by_span[span]
        return by_text.get(_normalize_text(text))

    async def aingest_chunk(self, memgraph: Memgraph, chunk: "Chunk") -> None:
        entities, relations = await asyncio.to_thread(self._extract_sync, chunk.text)

        entity_id_by_span: dict[tuple[int, int], str] = {}
        entity_id_by_text: dict[str, str] = {}
        node_dicts: list[dict[str, Any]] = []
        for entity in entities:
            if (
                self.entity_confidence_threshold is not None
                and entity.confidence is not None
                and entity.confidence < self.entity_confidence_threshold
            ):
                continue
            normalized = _normalize_text(entity.text)
            if not normalized:
                continue
            entity_id = _entity_id(chunk.hash, entity.entity_type, normalized)
            node_dicts.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity.entity_type,
                    "text": entity.text,
                    "file_path": chunk.hash,
                }
            )
            if entity.start is not None and entity.end is not None:
                entity_id_by_span[(entity.start, entity.end)] = entity_id
            entity_id_by_text.setdefault(normalized, entity_id)

        if node_dicts:
            create_nodes_from_list(memgraph, node_dicts, self._workspace, 100, merge_key="entity_id")

        if not relations:
            return

        relationships_by_type: dict[str, list[dict[str, Any]]] = {}
        for relation in relations:
            confidences = [c for c in (relation.head_confidence, relation.tail_confidence) if c is not None]
            if (
                self.relation_confidence_threshold is not None
                and confidences
                and min(confidences) < self.relation_confidence_threshold
            ):
                continue
            head_id = self._resolve_entity_id(
                relation.head_span, relation.head_text, entity_id_by_span, entity_id_by_text
            )
            tail_id = self._resolve_entity_id(
                relation.tail_span, relation.tail_text, entity_id_by_span, entity_id_by_text
            )
            if head_id is None or tail_id is None:
                logger.warning(
                    f"Skipping {relation.relation_type!r} relation: could not match head/tail "
                    f"({relation.head_text!r} -> {relation.tail_text!r}) to an extracted entity"
                )
                continue
            relationships_by_type.setdefault(relation.relation_type, []).append({"from": head_id, "to": tail_id})

        if relationships_by_type:
            upsert_typed_relationships(memgraph, self._workspace, "entity_id", relationships_by_type)
