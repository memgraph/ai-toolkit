"""Direct extraction-quality comparison: LightRAGBackend vs GLiNER2Backend.

Runs both backends over the same gold-labeled chunks (gold_corpus.jsonl),
reads back what each wrote to a dedicated eval Memgraph instance, and scores
against gold with deterministic precision/recall/F1 -- no LLM judge, since
entity/relation matching is a countable metric, not a quality judgment (see
evals/README.md).

    uv run --package unstructured2graph python evals/extraction_quality.py \
        --memgraph-url bolt://localhost:7691 --backend both

Requires a live, DEDICATED Memgraph instance (wiped before each backend's
pass -- never point this at a shared or development database):

    docker run -d --name ai-toolkit-eval-memgraph-extraction -p 7691:7687 \
        memgraph/memgraph-mage:latest

LightRAGBackend additionally needs OPENAI_API_KEY (or another configured
LLM); GLiNER2Backend needs `gliner2` installed manually -- not a
pyproject.toml extra, see gliner2_backend.py's module docstring for why
(`pip install 'gliner2[local]>=2.0.0'`). Missing either skips that backend
with a clear message rather than failing the whole run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memgraph_toolbox.api.memgraph import Memgraph
from unstructured2graph import DEFAULT_ONTOLOGY, Ontology, RelationType, from_texts
from unstructured2graph.extraction_backend import ExtractionBackend, LightRAGBackend
from unstructured2graph.gliner2_backend import _normalize_text
from unstructured2graph.memgraph import _require_valid_identifier

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CORPUS_PATH = SCRIPT_DIR / "gold_corpus.jsonl"
DEFAULT_MEMGRAPH_URL = "bolt://localhost:7691"  # distinct from context-graph-eval's own 7689

logger = logging.getLogger(__name__)

#: USD per 1M tokens, (input, output) -- OpenAI's published list price at the
#: time this was written. Point-in-time, not fetched live: check current
#: pricing before trusting this for a real budget decision, and update here
#: if LightRAG's default llm_model_func (gpt_4o_mini_complete) ever changes.
MODEL_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
}
DEFAULT_LIGHTRAG_MODEL = "gpt-4o-mini"  # LightRAG's own default llm_model_func


def estimated_cost_usd(prompt_tokens: int, completion_tokens: int, model: str) -> float | None:
    """None (not 0.0) when the model isn't in MODEL_PRICING_PER_1M_TOKENS --
    an unpriced model should read as "unknown", not "free"."""
    pricing = MODEL_PRICING_PER_1M_TOKENS.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price


#: Same entity vocabulary as DEFAULT_ONTOLOGY (the one real callers get by
#: default) plus a small relation vocabulary this eval needs to exercise
#: relation scoring at all -- see build_gold_corpus.py's module docstring for
#: why the adopted LongMemEval text alone doesn't carry named-entity
#: relations. Passed to both backends so LightRAG's entity_types_guidance and
#: GLiNER2's schema are steered toward the same vocabulary as the gold labels.
EVAL_ONTOLOGY = Ontology(
    entity_types=DEFAULT_ONTOLOGY.entity_types,
    relation_types=(
        RelationType(label="works_for", description="Employment relationship between a person and an organization"),
        RelationType(label="founded", description="A person founded an organization"),
        RelationType(label="located_in", description="A place or organization is located within another place"),
    ),
)


# ------------------------------------------------------------------
# Gold corpus
# ------------------------------------------------------------------


@dataclass(frozen=True)
class GoldEntity:
    """One gold entity span. `text` is scored as a normalized set member
    (see _entity_agnostic_set/_entity_sensitive_set below), never by its
    character offset -- gold_corpus.jsonl also carries `start`/`end` for
    each entity, but those exist only so build_gold_corpus.py can assert
    the text occurs verbatim in the chunk; this eval never reads them back.
    Matching by normalized text instead of offset is deliberate (see
    README.md's "Reading the report" section): LightRAG's LLM-normalized
    entity text doesn't reliably char-align with the source span the way
    GLiNER2's does, so offset matching would penalize LightRAG for a
    difference that has nothing to do with extraction quality. The
    consequence: two distinct gold mentions of the same normalized text
    within one chunk collapse into a single scored element (a set, not a
    multiset) -- this measures unique-normalized-entity-text coverage per
    chunk, not exhaustive mention-level recall.

    Attributes:
        text: The gold entity's literal text, as it appears in the chunk.
        type: Ontology entity-type label (e.g. "Person").
    """

    text: str
    type: str


@dataclass(frozen=True)
class GoldRelation:
    """One gold relation triple, matched the same way GoldEntity's text is:
    by normalized text, not by span or by reference to a specific
    GoldEntity -- see GoldEntity's docstring for why.

    Attributes:
        type: Ontology relation-type label (e.g. "works_for").
        head: The relation's subject, as literal text.
        tail: The relation's object, as literal text.
    """

    type: str
    head: str
    tail: str


@dataclass(frozen=True)
class GoldRecord:
    """One gold-labeled chunk from gold_corpus.jsonl.

    Attributes:
        chunk_id: Stable id from build_gold_corpus.py (e.g. "lme-03", "auth-02").
        source: Provenance -- "longmemeval" (sampled real text) or "authored"
            (written for this eval). See build_gold_corpus.py's module docstring.
        text: The chunk's literal text, fed to from_texts() unmodified.
        entities: Gold entities for this chunk.
        relations: Gold relations for this chunk (empty for most --
            see build_gold_corpus.py for why relation coverage is limited).
    """

    chunk_id: str
    source: str
    text: str
    entities: tuple[GoldEntity, ...]
    relations: tuple[GoldRelation, ...]


def load_gold_corpus(path: Path = DEFAULT_CORPUS_PATH) -> list[GoldRecord]:
    """Parse gold_corpus.jsonl (one GoldRecord per line) in file order."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            records.append(
                GoldRecord(
                    chunk_id=raw["chunk_id"],
                    source=raw["source"],
                    text=raw["text"],
                    entities=tuple(GoldEntity(e["text"], e["type"]) for e in raw["entities"]),
                    relations=tuple(GoldRelation(r["type"], r["head"], r["tail"]) for r in raw["relations"]),
                )
            )
    return records


def _normalize_type(entity_type: str) -> str:
    """Case-insensitive type compare: LightRAG's raw entity_type casing is
    whatever the LLM returned (typically lowercase), while GLiNER2's schema
    keys -- and this eval's gold labels -- are the PascalCase
    default_ontology.yaml vocabulary. Comparing case-folded on both sides is
    the fair basis, not exact-string equality."""
    return entity_type.strip().lower()


# ------------------------------------------------------------------
# Cost instrumentation (LightRAG only -- GLiNER2 makes no LLM calls at all)
# ------------------------------------------------------------------


@dataclass
class LLMCallCounters:
    calls: int = 0
    #: Approximate, via tiktoken's cl100k_base over prompt/system/history/
    #: completion text -- LightRAG's llm_model_func implementations return a
    #: plain string, not a response object with native usage stats, so exact
    #: provider-billed token counts aren't available without swapping in a
    #: different llm_model_func. Reuses the same tokenizer context-graph-eval's
    #: own efficiency metric is pinned to (scoring.DEFAULT_TOKENIZER).
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _history_message_text(message: Any) -> str:
    """LightRAG's history_messages entries are typically {"role": ..., "content": ...}
    dicts; fall back to str() for anything else rather than skipping it silently."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(message)


def _counting_llm_wrapper(llm_func: Any, counters: LLMCallCounters) -> Any:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    async def wrapped(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs):
        counters.calls += 1
        counters.prompt_tokens += len(encoding.encode(prompt or ""))
        if system_prompt:
            counters.prompt_tokens += len(encoding.encode(system_prompt))
        # history_messages is real provider-billed input too -- LightRAG's
        # gleaning/merge calls pass prior conversation turns through it, so
        # omitting it here understated every such call's token/USD cost.
        for message in history_messages or []:
            counters.prompt_tokens += len(encoding.encode(_history_message_text(message)))
        result = await llm_func(prompt, system_prompt=system_prompt, history_messages=history_messages or [], **kwargs)
        counters.completion_tokens += len(encoding.encode(result if isinstance(result, str) else str(result)))
        return result

    return wrapped


# ------------------------------------------------------------------
# Reading back what a backend wrote
# ------------------------------------------------------------------


def _read_entities(memgraph: Memgraph, workspace: str, chunk_hash: str, text_property: str) -> list[tuple[str, str]]:
    """(text, entity_type) pairs written under this chunk's file_path.

    Raises:
        ValueError: if workspace or text_property isn't a valid Cypher
            identifier -- both are f-string-interpolated below, and workspace
            in particular is an ExtractionBackend's caller-configured
            workspace_label, not a compile-time literal.
    """
    _require_valid_identifier(workspace, "workspace")
    _require_valid_identifier(text_property, "text_property")
    rows = memgraph.query(
        f"MATCH (n:{workspace}) WHERE n.file_path = $hash RETURN n.{text_property} AS text, "
        "n.entity_type AS entity_type",
        params={"hash": chunk_hash},
    )
    return [(row["text"] or "", row["entity_type"] or "") for row in rows]


def _read_ontology_conformance(memgraph: Memgraph, workspace: str, chunk_hash: str) -> tuple[int, int]:
    """(conformant_count, nonconformant_count) among this chunk's entities,
    per enforce_ontology's ontology_conformant flag -- a free bonus signal
    from running with enforce_ontology=True, not something scored against
    gold.

    Raises:
        ValueError: if workspace isn't a valid Cypher identifier.
    """
    _require_valid_identifier(workspace, "workspace")
    rows = memgraph.query(
        f"MATCH (n:{workspace}) WHERE n.file_path = $hash RETURN n.ontology_conformant AS conformant",
        params={"hash": chunk_hash},
    )
    nonconformant = sum(1 for row in rows if row["conformant"] is False)
    return len(rows) - nonconformant, nonconformant


def _read_relations(
    memgraph: Memgraph, workspace: str, chunk_hash: str, text_property: str, relation_types: tuple[str, ...]
) -> list[tuple[str, str, str]]:
    """(relation_type, head_text, tail_text) triples among entities touching
    this chunk. Only meaningful for a backend whose ontology declared
    relation_types -- LightRAG writes generic :DIRECTED edges with no typed
    label to match against, so callers pass relation_types=() for it and get
    back an empty list rather than a meaningless comparison.

    Raises:
        ValueError: if workspace, text_property, or any of relation_types
            isn't a valid Cypher identifier.
    """
    if not relation_types:
        return []
    _require_valid_identifier(workspace, "workspace")
    _require_valid_identifier(text_property, "text_property")
    for relation_type in relation_types:
        _require_valid_identifier(relation_type, "relation type")
    rows = memgraph.query(
        f"""
        MATCH (a:{workspace})-[r]->(b:{workspace})
        WHERE type(r) IN $relation_types AND (a.file_path = $hash OR b.file_path = $hash)
        RETURN type(r) AS rel_type, a.{text_property} AS head, b.{text_property} AS tail
        """,
        params={"hash": chunk_hash, "relation_types": list(relation_types)},
    )
    return [(row["rel_type"], row["head"] or "", row["tail"] or "") for row in rows]


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


@dataclass
class PRF1:
    """Accumulates precision/recall/F1 over repeated set-vs-set comparisons
    (one `add()` call per chunk), via true/false positive/negative counts
    rather than per-chunk P/R/F1 averaged afterward -- micro-averaging, so a
    chunk with more gold entities counts proportionally more, not the same
    as a chunk with one.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, gold: set, predicted: set) -> None:
        """Accumulate one chunk's set comparison into the running totals."""
        self.tp += len(gold & predicted)
        self.fp += len(predicted - gold)
        self.fn += len(gold - predicted)

    @property
    def precision(self) -> float | None:
        """None when nothing was predicted at all (undefined, not 0.0)."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        """None when gold has nothing to find at all (undefined, not 0.0)."""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def f1(self) -> float | None:
        """None only when precision or recall is itself None (nothing to
        score). A real, computed 0.0 for either must still produce F1 = 0.0,
        not None -- `if not p or not r` treated a genuine zero the same as
        "undefined", silently reporting a real "found nothing right" result
        as `n/a` in the printed table."""
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class BackendReport:
    """One backend run's scoring + cost/latency summary, accumulated by
    run_backend() across every chunk in the gold corpus.

    Attributes:
        backend_name: "lightrag" or "gliner2".
        entity_type_agnostic: Entity P/R/F1 by normalized text only, type ignored.
        entity_type_sensitive: Entity P/R/F1 by normalized text AND type.
        relation_scoring: Relation P/R/F1, or None if this backend has no
            relation vocabulary at all (LightRAG) -- distinct from a PRF1
            with all-zero counts, which would mean "had a vocabulary but
            found nothing."
        latencies_seconds: Wall-clock seconds per chunk's aingest_chunk() call.
        ontology_conformant / ontology_nonconformant: Count of entities whose
            entity_type did/didn't match default_ontology.yaml's vocabulary
            (enforce_ontology's own signal, not scored against gold).
        llm_calls / llm_prompt_tokens / llm_completion_tokens: None for a
            backend with no LLM calls at all (GLiNER2); otherwise LightRAG's
            approximate call/token counts (see LLMCallCounters).
        llm_model: Which MODEL_PRICING_PER_1M_TOKENS key prices this run's
            cost -- None when there's no LLM cost to price.
    """

    backend_name: str
    entity_type_agnostic: PRF1 = field(default_factory=PRF1)
    entity_type_sensitive: PRF1 = field(default_factory=PRF1)
    relation_scoring: PRF1 | None = None
    latencies_seconds: list[float] = field(default_factory=list)
    ontology_conformant: int = 0
    ontology_nonconformant: int = 0
    llm_calls: int | None = None
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_model: str | None = None

    def latency_summary(self) -> tuple[float, float]:
        """(median, mean) wall-clock seconds per chunk."""
        return (statistics.median(self.latencies_seconds), statistics.mean(self.latencies_seconds))

    @property
    def estimated_cost_usd(self) -> float | None:
        """None when there's no LLM cost to estimate at all (llm_calls is
        None) or the model has no entry in MODEL_PRICING_PER_1M_TOKENS --
        see estimated_cost_usd()'s own None-vs-0.0 contract."""
        if self.llm_calls is None or self.llm_model is None:
            return None
        return estimated_cost_usd(self.llm_prompt_tokens or 0, self.llm_completion_tokens or 0, self.llm_model)


def _entity_agnostic_set(entities: list[tuple[str, str]]) -> set[str]:
    return {_normalize_text(text) for text, _ in entities if _normalize_text(text)}


def _entity_sensitive_set(entities: list[tuple[str, str]]) -> set[tuple[str, str]]:
    return {(_normalize_text(text), _normalize_type(etype)) for text, etype in entities if _normalize_text(text)}


def _relation_set(relations: list[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    return {
        (_normalize_type(rel_type), _normalize_text(head), _normalize_text(tail))
        for rel_type, head, tail in relations
        if head and tail
    }


# ------------------------------------------------------------------
# Backend configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class BackendSpec:
    """Everything run_backend() and print_report() need to know about a
    backend TYPE (as opposed to one particular built instance): its name,
    which node property holds the literal extracted text ("entity_id" for
    LightRAG -- see LightRAG's own upsert_node() convention -- vs "text" for
    GLiNER2), its relation-type vocabulary (empty for a backend with none),
    and which LLM model (if any) prices its cost. These used to travel
    run_backend()'s call chain as four separate, loosely related parameters;
    bundling them here is what let LightRAGBackend's and GLiNER2Backend's
    repeat loops in run() collapse into one shared _run_repeated() instead
    of two near-identical copies.

    Attributes:
        name: "lightrag" or "gliner2".
        text_property: Node property holding the literal extracted entity text.
        relation_types: This backend's relation-type vocabulary, or () if none.
        llm_model: MODEL_PRICING_PER_1M_TOKENS key for this backend's LLM
            cost, or None if it makes no LLM calls at all.
    """

    name: str
    text_property: str
    relation_types: tuple[str, ...] = ()
    llm_model: str | None = None


LIGHTRAG_SPEC = BackendSpec(
    name="lightrag",
    text_property="entity_id",
    relation_types=(),  # LightRAG has no relation_types concept -- see README
    llm_model=DEFAULT_LIGHTRAG_MODEL,
)
GLINER2_SPEC = BackendSpec(
    name="gliner2", text_property="text", relation_types=tuple(t.label for t in EVAL_ONTOLOGY.relation_types)
)


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

#: Builds one fresh (backend, llm_counters) pair. llm_counters is None for a
#: backend with no LLM cost to track (GLiNER2). Async so LightRAG's
#: OPENAI_API_KEY-dependent MemgraphLightRAGWrapper.initialize() and
#: GLiNER2's model-loading stay lazy until actually needed.
BackendBuilder = Callable[[], Awaitable[tuple[ExtractionBackend, LLMCallCounters | None]]]


async def _build_lightrag() -> tuple[LightRAGBackend, LLMCallCounters]:
    """Fresh MemgraphLightRAGWrapper + LightRAGBackend + LLMCallCounters
    every call -- unlike GLiNER2's reused-instance builder, LightRAG is
    rebuilt (and its counters reset) for each repeat so per-run cost/latency
    numbers aren't cumulative across repeats, and so its shared process-
    global state gets a clean afinalize()/reinit cycle (see
    MemgraphLightRAGWrapper.afinalize()'s own docstring)."""
    from lightrag.llm.openai import gpt_4o_mini_complete

    from lightrag_memgraph import MemgraphLightRAGWrapper

    counters = LLMCallCounters()
    wrapper = MemgraphLightRAGWrapper(log_level="WARNING")
    await wrapper.initialize(
        working_dir="./lightrag_storage.extraction_quality_eval",
        llm_model_func=_counting_llm_wrapper(gpt_4o_mini_complete, counters),
        addon_params=EVAL_ONTOLOGY.addon_params(),
    )
    return LightRAGBackend(wrapper), counters


def _reuse_gliner2_builder(gliner2_model: str) -> BackendBuilder:
    """Returns a builder that loads the GLiNER2 model once and hands back
    the same instance on every call -- unlike LightRAG, GLiNER2 is a fixed
    local model with no per-run state to reset, and reloading it per repeat
    would just re-pay an expensive, pointless model load."""
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    backend = GLiNER2Backend(model_name=gliner2_model, ontology=EVAL_ONTOLOGY)

    async def build() -> tuple[GLiNER2Backend, None]:
        return backend, None

    return build


def _pin_memgraph_env(memgraph_url: str) -> None:
    """Force every consumer of Memgraph connection env vars -- this eval's
    own memgraph_toolbox client (reads MEMGRAPH_URL) AND LightRAG's
    separately-read storage backends (read MEMGRAPH_URI/MEMGRAPH_USERNAME,
    per lightrag_memgraph.core) -- onto the same instance.

    lightrag_memgraph.core's own _bridge_lightrag_env_names() mirrors
    MEMGRAPH_URL onto MEMGRAPH_URI, but only when MEMGRAPH_URI isn't already
    set. An ambient MEMGRAPH_URI left over from something else on this
    machine would silently win over --memgraph-url under that conditional
    bridge, so LightRAG would write into a different -- possibly shared --
    database than the one this eval reads back from and wipes. Setting both
    unconditionally here, before any backend is built, closes that gap
    rather than relying on the bridge's fallback behavior.
    """
    os.environ["MEMGRAPH_URL"] = memgraph_url
    os.environ["MEMGRAPH_URI"] = memgraph_url
    os.environ.setdefault("MEMGRAPH_USER", "")
    os.environ.setdefault("MEMGRAPH_USERNAME", "")
    os.environ.setdefault("MEMGRAPH_PASSWORD", "")


async def run_backend(
    spec: BackendSpec,
    backend: ExtractionBackend,
    memgraph: Memgraph,
    gold: list[GoldRecord],
    *,
    llm_counters: LLMCallCounters | None = None,
) -> BackendReport:
    """Run one already-built backend instance over the whole gold corpus
    once: wipes `memgraph`, ingests every chunk via from_texts(), reads back
    what got written, and scores it against gold.

    Args:
        spec: Static config for this backend type (see BackendSpec).
        backend: The backend instance to run.
        memgraph: Wiped at the start of this call -- must be a dedicated eval
            instance (see this module's docstring), never shared.
        gold: The corpus to ingest and score against.
        llm_counters: If given, folded into the returned report's
            llm_calls/llm_prompt_tokens/llm_completion_tokens.

    Returns:
        A BackendReport summarizing this one run.
    """
    memgraph.query("MATCH (n) DETACH DELETE n")
    report = BackendReport(backend_name=spec.name)
    workspace = backend.workspace_label

    for record in gold:
        started = time.perf_counter()
        grouped = await from_texts([record.text], memgraph, backend, enforce_ontology=True)
        report.latencies_seconds.append(time.perf_counter() - started)

        chunk_hash = grouped[0][0].hash
        predicted_entities = _read_entities(memgraph, workspace, chunk_hash, spec.text_property)
        conformant, nonconformant = _read_ontology_conformance(memgraph, workspace, chunk_hash)
        report.ontology_conformant += conformant
        report.ontology_nonconformant += nonconformant

        report.entity_type_agnostic.add(
            _entity_agnostic_set([(e.text, e.type) for e in record.entities]),
            _entity_agnostic_set(predicted_entities),
        )
        report.entity_type_sensitive.add(
            _entity_sensitive_set([(e.text, e.type) for e in record.entities]),
            _entity_sensitive_set(predicted_entities),
        )

        if spec.relation_types:
            predicted_relations = _read_relations(
                memgraph, workspace, chunk_hash, spec.text_property, spec.relation_types
            )
            if report.relation_scoring is None:
                report.relation_scoring = PRF1()
            report.relation_scoring.add(
                _relation_set([(r.type, r.head, r.tail) for r in record.relations]),
                _relation_set(predicted_relations),
            )

    if llm_counters is not None:
        report.llm_calls = llm_counters.calls
        report.llm_prompt_tokens = llm_counters.prompt_tokens
        report.llm_completion_tokens = llm_counters.completion_tokens
        report.llm_model = spec.llm_model

    return report


async def _run_repeated(
    spec: BackendSpec,
    build: BackendBuilder,
    memgraph: Memgraph,
    gold: list[GoldRecord],
    *,
    repeat: int,
    finalize: Callable[[Any], Awaitable[None]] | None = None,
) -> list[BackendReport]:
    """Run one backend `repeat` times, collecting one BackendReport per run.

    Shared by both backends in run() -- the only difference between them is
    how to build (and optionally finalize) an instance, threaded through
    `build`/`finalize` rather than duplicating this loop once per backend.

    Args:
        spec: Static config for this backend type.
        build: Builds one fresh (backend, llm_counters) pair -- called once
            per repeat, so a backend that wants a fresh instance per run
            (LightRAG) and one that wants to reuse a single instance
            (GLiNER2, via _reuse_gliner2_builder) both fit this same loop.
        memgraph: Passed through to run_backend() each iteration.
        gold: The corpus each run scores against.
        repeat: How many times to run.
        finalize: If given, awaited on the built backend after each run
            (e.g. LightRAGBackend.afinalize, to reset LightRAG's shared
            process-global state between repeats). Typed loosely (Any, not
            ExtractionBackend) since this is inherently backend-specific --
            afinalize() isn't part of the ExtractionBackend protocol at all
            (GLiNER2Backend has no equivalent), so a caller only ever pairs
            this with a `build` that returns a matching concrete type.

    Returns:
        One BackendReport per repeat, in run order.
    """
    reports = []
    for i in range(repeat):
        backend, llm_counters = await build()
        reports.append(await run_backend(spec, backend, memgraph, gold, llm_counters=llm_counters))
        if finalize is not None:
            await finalize(backend)
        print(f"  {spec.name} run {i + 1}/{repeat} done")
    return reports


async def run(
    *,
    memgraph_url: str,
    backends: list[str],
    corpus_path: Path,
    gliner2_model: str,
    repeat: int,
) -> dict[str, list[BackendReport]]:
    """Run the requested backends over the gold corpus and return their
    reports, keyed by backend name. A backend whose dependency is missing
    (OPENAI_API_KEY for lightrag, `gliner2` installed for gliner2) is
    skipped with a printed message and simply absent from the result,
    rather than raising."""
    _pin_memgraph_env(memgraph_url)
    gold = load_gold_corpus(corpus_path)
    memgraph = Memgraph(user_agent="unstructured2graph-extraction-eval")

    results: dict[str, list[BackendReport]] = {}

    if "lightrag" in backends:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Skipping lightrag: OPENAI_API_KEY not set.")
        else:
            results["lightrag"] = await _run_repeated(
                LIGHTRAG_SPEC,
                _build_lightrag,
                memgraph,
                gold,
                repeat=repeat,
                finalize=lambda backend: backend.afinalize(),
            )

    if "gliner2" in backends:
        try:
            build_gliner2 = _reuse_gliner2_builder(gliner2_model)
        except ImportError as e:
            print(f"Skipping gliner2: {e}")
        else:
            results["gliner2"] = await _run_repeated(GLINER2_SPEC, build_gliner2, memgraph, gold, repeat=repeat)

    memgraph.close()
    return results


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _fmt_cost(cost: float | None, llm_calls: int | None) -> str:
    if llm_calls is None:
        return "n/a"  # not an LLM-backed backend at all -- not the same as "$0"
    if cost is None:
        return "unpriced"  # LLM-backed, but MODEL_PRICING_PER_1M_TOKENS has no entry for this model
    return f"${cost:.4f}"


def print_report(results: dict[str, list[BackendReport]]) -> None:
    """Print a plain table of every BackendReport in `results` (as returned
    by run()), plus fixed caveats about how to read it. No automatic
    verdict, by design -- see the caveats themselves."""
    print()
    print(
        f"{'backend':<10} {'run':>4} {'ent P':>8} {'ent R':>8} {'ent F1':>8}   "
        f"{'typed P':>8} {'typed R':>8} {'typed F1':>8}   {'rel P':>8} {'rel R':>8} {'rel F1':>8}   "
        f"{'latency med/s':>14} {'calls':>7} {'cost':>10}"
    )
    for backend_name, reports in results.items():
        for i, r in enumerate(reports, start=1):
            median_latency, _mean_latency = r.latency_summary()
            rel = r.relation_scoring
            print(
                f"{backend_name:<10} {i:>4} "
                f"{_fmt_pct(r.entity_type_agnostic.precision):>8} {_fmt_pct(r.entity_type_agnostic.recall):>8} "
                f"{_fmt_pct(r.entity_type_agnostic.f1):>8}   "
                f"{_fmt_pct(r.entity_type_sensitive.precision):>8} {_fmt_pct(r.entity_type_sensitive.recall):>8} "
                f"{_fmt_pct(r.entity_type_sensitive.f1):>8}   "
                f"{_fmt_pct(rel.precision if rel else None):>8} {_fmt_pct(rel.recall if rel else None):>8} "
                f"{_fmt_pct(rel.f1 if rel else None):>8}   "
                f"{median_latency:>14.2f} "
                f"{r.llm_calls if r.llm_calls is not None else 'n/a':>7} "
                f"{_fmt_cost(r.estimated_cost_usd, r.llm_calls):>10}"
            )
            if r.ontology_conformant + r.ontology_nonconformant:
                total = r.ontology_conformant + r.ontology_nonconformant
                print(
                    f"{'':<10} {'':>4} ontology_conformant: {r.ontology_conformant}/{total} "
                    f"({_fmt_pct(r.ontology_conformant / total)})"
                )
            if r.llm_calls is not None:
                print(
                    f"{'':<10} {'':>4} tokens: {r.llm_prompt_tokens or 0} prompt + "
                    f"{r.llm_completion_tokens or 0} completion, model={r.llm_model} "
                    f"({'unpriced -- add it to MODEL_PRICING_PER_1M_TOKENS' if r.estimated_cost_usd is None else 'approximate, point-in-time pricing'})"
                )
    print()
    print("Entity/relation P/R/F1 are unique-normalized-text set matches per chunk, not span-position matches")
    print(
        "(see GoldEntity's docstring) -- repeated identical mentions within one chunk count once, not per-occurrence."
    )
    print("No automatic verdict -- read the numbers, don't let this pick for you.")
    print("relation columns are only meaningful for backends with a relation vocabulary (gliner2 here).")
    print(
        "cost is a point-in-time estimate (see MODEL_PRICING_PER_1M_TOKENS) from approximate token counts, not billed usage."
    )
    if any(len(reports) > 1 for reports in results.values()):
        print("Multiple runs shown per backend -- compare their spread before trusting any single number.")


def main() -> None:
    try:
        from dotenv import load_dotenv

        # Bare call: python-dotenv's own find_dotenv() searches upward from
        # cwd, so this picks up the repo-root .env (OPENAI_API_KEY etc.)
        # whether this script is run from unstructured2graph/ or the repo
        # root -- same bare-call convention integrations/langchain-memgraph
        # and integrations/mcp-memgraph already use in their own tests.
        # Soft dependency (unstructured2graph[test]): running without it
        # installed just means OPENAI_API_KEY has to be exported manually.
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memgraph-url", default=DEFAULT_MEMGRAPH_URL)
    parser.add_argument("--backend", choices=["lightrag", "gliner2", "both"], default="both")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--gliner2-model", default="fastino/gliner2.5-base-v1")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each backend N times to see run-to-run spread")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    backends = ["lightrag", "gliner2"] if args.backend == "both" else [args.backend]
    results = asyncio.run(
        run(
            memgraph_url=args.memgraph_url,
            backends=backends,
            corpus_path=args.corpus,
            gliner2_model=args.gliner2_model,
            repeat=args.repeat,
        )
    )
    print_report(results)


if __name__ == "__main__":
    main()
