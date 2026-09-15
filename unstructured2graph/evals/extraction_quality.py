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
LLM); GLiNER2Backend needs the optional `gliner2` dependency
(`pip install unstructured2graph[gliner2]`). Missing either skips that
backend with a clear message rather than failing the whole run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memgraph_toolbox.api.memgraph import Memgraph
from unstructured2graph import DEFAULT_ONTOLOGY, Ontology, RelationType, from_texts
from unstructured2graph.extraction_backend import ExtractionBackend, LightRAGBackend
from unstructured2graph.gliner2_backend import _normalize_text

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
    text: str
    type: str


@dataclass(frozen=True)
class GoldRelation:
    type: str
    head: str
    tail: str


@dataclass(frozen=True)
class GoldRecord:
    chunk_id: str
    source: str
    text: str
    entities: tuple[GoldEntity, ...]
    relations: tuple[GoldRelation, ...]


def load_gold_corpus(path: Path = DEFAULT_CORPUS_PATH) -> list[GoldRecord]:
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
    #: Approximate, via tiktoken's cl100k_base over prompt/completion text --
    #: LightRAG's llm_model_func implementations return a plain string, not a
    #: response object with native usage stats, so exact provider-billed
    #: token counts aren't available without swapping in a different
    #: llm_model_func. Reuses the same tokenizer context-graph-eval's own
    #: efficiency metric is pinned to (scoring.DEFAULT_TOKENIZER).
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _counting_llm_wrapper(llm_func: Any, counters: LLMCallCounters) -> Any:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    async def wrapped(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs):
        counters.calls += 1
        counters.prompt_tokens += len(encoding.encode(prompt or ""))
        if system_prompt:
            counters.prompt_tokens += len(encoding.encode(system_prompt))
        result = await llm_func(prompt, system_prompt=system_prompt, history_messages=history_messages or [], **kwargs)
        counters.completion_tokens += len(encoding.encode(result if isinstance(result, str) else str(result)))
        return result

    return wrapped


# ------------------------------------------------------------------
# Reading back what a backend wrote
# ------------------------------------------------------------------


def _read_entities(memgraph: Memgraph, workspace: str, chunk_hash: str, text_property: str) -> list[tuple[str, str]]:
    """(text, entity_type) pairs written under this chunk's file_path."""
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
    gold."""
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
    back an empty list rather than a meaningless comparison."""
    if not relation_types:
        return []
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
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, gold: set, predicted: set) -> None:
        self.tp += len(gold & predicted)
        self.fp += len(predicted - gold)
        self.fn += len(gold - predicted)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return None
        return 2 * p * r / (p + r)


@dataclass
class BackendReport:
    backend_name: str
    entity_type_agnostic: PRF1 = field(default_factory=PRF1)
    entity_type_sensitive: PRF1 = field(default_factory=PRF1)
    relation_scoring: PRF1 | None = None  # None when this backend has no relation vocabulary at all
    latencies_seconds: list[float] = field(default_factory=list)
    ontology_conformant: int = 0
    ontology_nonconformant: int = 0
    llm_calls: int | None = None
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_model: str | None = None  # which MODEL_PRICING_PER_1M_TOKENS key priced this run

    def latency_summary(self) -> tuple[float, float]:
        return (statistics.median(self.latencies_seconds), statistics.mean(self.latencies_seconds))

    @property
    def estimated_cost_usd(self) -> float | None:
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
# Runner
# ------------------------------------------------------------------


async def _build_lightrag_backend(counters: LLMCallCounters) -> LightRAGBackend:
    from lightrag.llm.openai import gpt_4o_mini_complete

    from lightrag_memgraph import MemgraphLightRAGWrapper

    wrapper = MemgraphLightRAGWrapper(log_level="WARNING")
    await wrapper.initialize(
        working_dir="./lightrag_storage.extraction_quality_eval",
        llm_model_func=_counting_llm_wrapper(gpt_4o_mini_complete, counters),
        addon_params=EVAL_ONTOLOGY.addon_params(),
    )
    return LightRAGBackend(wrapper)


def _build_gliner2_backend(model_name: str):
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    return GLiNER2Backend(model_name=model_name, ontology=EVAL_ONTOLOGY)


async def run_backend(
    backend_name: str,
    backend: ExtractionBackend,
    memgraph: Memgraph,
    gold: list[GoldRecord],
    *,
    text_property: str,
    relation_types: tuple[str, ...],
    llm_counters: LLMCallCounters | None = None,
    llm_model: str | None = None,
) -> BackendReport:
    memgraph.query("MATCH (n) DETACH DELETE n")
    report = BackendReport(backend_name=backend_name)
    workspace = backend.workspace_label

    for record in gold:
        started = time.perf_counter()
        grouped = await from_texts([record.text], memgraph, backend, enforce_ontology=True)
        report.latencies_seconds.append(time.perf_counter() - started)

        chunk_hash = grouped[0][0].hash
        predicted_entities = _read_entities(memgraph, workspace, chunk_hash, text_property)
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

        if relation_types:
            predicted_relations = _read_relations(memgraph, workspace, chunk_hash, text_property, relation_types)
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
        report.llm_model = llm_model

    return report


async def run(
    *,
    memgraph_url: str,
    backends: list[str],
    corpus_path: Path,
    gliner2_model: str,
    repeat: int,
) -> dict[str, list[BackendReport]]:
    os.environ["MEMGRAPH_URL"] = memgraph_url
    gold = load_gold_corpus(corpus_path)
    memgraph = Memgraph(user_agent="unstructured2graph-extraction-eval")

    results: dict[str, list[BackendReport]] = {}

    if "lightrag" in backends:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Skipping lightrag: OPENAI_API_KEY not set.")
        else:
            reports = []
            for i in range(repeat):
                counters = LLMCallCounters()
                wrapper_backend = await _build_lightrag_backend(counters)
                reports.append(
                    await run_backend(
                        "lightrag",
                        wrapper_backend,
                        memgraph,
                        gold,
                        text_property="entity_id",
                        relation_types=(),  # LightRAG has no relation_types concept -- see README
                        llm_counters=counters,
                        llm_model=DEFAULT_LIGHTRAG_MODEL,
                    )
                )
                await wrapper_backend.wrapper.afinalize()
                print(f"  lightrag run {i + 1}/{repeat} done")
            results["lightrag"] = reports

    if "gliner2" in backends:
        try:
            gliner2_backend = _build_gliner2_backend(gliner2_model)
        except ImportError as e:
            print(f"Skipping gliner2: {e}")
        else:
            reports = []
            for i in range(repeat):
                reports.append(
                    await run_backend(
                        "gliner2",
                        gliner2_backend,
                        memgraph,
                        gold,
                        text_property="text",
                        relation_types=tuple(t.label for t in EVAL_ONTOLOGY.relation_types),
                    )
                )
                print(f"  gliner2 run {i + 1}/{repeat} done")
            results["gliner2"] = reports

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
