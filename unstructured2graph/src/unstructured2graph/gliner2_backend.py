"""GLiNER2-based ExtractionBackend: local, LLM-free entity/relation extraction.

Uses GLiNER2 (https://github.com/fastino-ai/GLiNER2, pip package `gliner2`),
a small open-source model that runs entirely locally (no GPU required, no
network calls, no API key) and does joint entity+relation extraction. This
module is not imported by unstructured2graph/__init__.py and never imports
`gliner2` at module scope, so `import unstructured2graph` never requires the
optional `gliner2` dependency -- only constructing a GLiNER2Backend does.
Install it with `pip install unstructured2graph[gliner2]`.
"""

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any

from memgraph_toolbox.api.memgraph import Memgraph

from .memgraph import create_nodes_from_list, upsert_typed_relationships
from .ontology import DEFAULT_ONTOLOGY, Ontology

if TYPE_CHECKING:
    from .loaders import Chunk

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


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
                database.
            model: Pre-loaded GLiNER2 extractor (e.g. an AutoExtractor
                instance), or a test fake exposing the same
                extract_entities()/extract_relations() methods. Bypasses
                loading `model_name` and importing `gliner2` entirely --
                the only supported way to use this class without the
                `gliner2` package installed.
            entity_confidence_threshold: Drop extracted entities below this
                confidence. None (default) keeps everything the model returns.
            relation_confidence_threshold: Drop extracted relations whose
                head/tail confidence falls below this (relations carry no
                confidence of their own -- see _extract_sync). None (default)
                keeps everything the model returns.
        """
        if model is not None:
            self.model = model
        else:
            try:
                from gliner2 import AutoExtractor
            except ImportError as e:
                raise ImportError("gliner2 is required for GLiNER2Backend; install unstructured2graph[gliner2]") from e
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

    def _extract_sync(self, text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Synchronous, CPU/GPU-bound model call (GLiNER2 has no async API,
        unlike LightRAG's network-bound LLM call) -- run via asyncio.to_thread
        from aingest_chunk() so it doesn't block the event loop.

        Returns (entities, relations):
            entities: [{"entity_type", "text", "start", "end", "confidence"}, ...]
            relations: [{"relation_type", "head_text", "head_span", "head_confidence",
                         "tail_text", "tail_span", "tail_confidence"}, ...]

        Entities and relations are extracted via two separate calls rather
        than one combined multi-task schema, since extract_entities()/
        extract_relations() each have an unambiguous, individually
        documented output shape; a relation's head/tail only ever carry a
        text span anyway (not a reference to a specific typed entity), so
        matching them back to already-extracted entities (see
        aingest_chunk()) is required either way.
        """
        entities: list[dict[str, Any]] = []
        raw_entities = self.model.extract_entities(
            text, self._entity_schema, include_spans=True, include_confidence=True
        )
        for entity_type, spans in raw_entities.get("entities", {}).items():
            for span in spans:
                if isinstance(span, dict):
                    entities.append(
                        {
                            "entity_type": entity_type,
                            "text": span.get("text", ""),
                            "start": span.get("start"),
                            "end": span.get("end"),
                            "confidence": span.get("confidence"),
                        }
                    )
                else:
                    entities.append(
                        {"entity_type": entity_type, "text": str(span), "start": None, "end": None, "confidence": None}
                    )

        relations: list[dict[str, Any]] = []
        if self._relation_schema:
            raw_relations = self.model.extract_relations(
                text, self._relation_schema, include_spans=True, include_confidence=True
            )
            for relation_type, pairs in raw_relations.get("relation_extraction", {}).items():
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
                        {
                            "relation_type": relation_type,
                            "head_text": head.get("text", ""),
                            "head_span": (head["start"], head["end"]) if head.get("start") is not None else None,
                            "head_confidence": head.get("confidence"),
                            "tail_text": tail.get("text", ""),
                            "tail_span": (tail["start"], tail["end"]) if tail.get("start") is not None else None,
                            "tail_confidence": tail.get("confidence"),
                        }
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
            confidence = entity.get("confidence")
            if (
                self.entity_confidence_threshold is not None
                and confidence is not None
                and confidence < self.entity_confidence_threshold
            ):
                continue
            normalized = _normalize_text(entity["text"])
            if not normalized:
                continue
            entity_id = _entity_id(chunk.hash, entity["entity_type"], normalized)
            node_dicts.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity["entity_type"],
                    "text": entity["text"],
                    "file_path": chunk.hash,
                }
            )
            if entity["start"] is not None and entity["end"] is not None:
                entity_id_by_span[(entity["start"], entity["end"])] = entity_id
            entity_id_by_text.setdefault(normalized, entity_id)

        if node_dicts:
            create_nodes_from_list(memgraph, node_dicts, self._workspace, 100, merge_key="entity_id")

        if not relations:
            return

        relationships_by_type: dict[str, list[dict[str, Any]]] = {}
        for relation in relations:
            confidences = [c for c in (relation["head_confidence"], relation["tail_confidence"]) if c is not None]
            if (
                self.relation_confidence_threshold is not None
                and confidences
                and min(confidences) < self.relation_confidence_threshold
            ):
                continue
            head_id = self._resolve_entity_id(
                relation["head_span"], relation["head_text"], entity_id_by_span, entity_id_by_text
            )
            tail_id = self._resolve_entity_id(
                relation["tail_span"], relation["tail_text"], entity_id_by_span, entity_id_by_text
            )
            if head_id is None or tail_id is None:
                logger.warning(
                    f"Skipping {relation['relation_type']!r} relation: could not match head/tail "
                    f"({relation['head_text']!r} -> {relation['tail_text']!r}) to an extracted entity"
                )
                continue
            relationships_by_type.setdefault(relation["relation_type"], []).append({"from": head_id, "to": tail_id})

        if relationships_by_type:
            upsert_typed_relationships(memgraph, self._workspace, "entity_id", relationships_by_type)
