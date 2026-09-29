"""Where does the typed GLiNER2 graph lose each eval answer?

Runs against a reconciled eval graph (PR #373's typed model, hand vocabulary:
100 questions, 5 sessions per question), with the eval's own answer prompt
and judge. For every question it traces the gold answer through its evidence
sessions:

    literal    the answer string occurs in the evidence session text
    entity     ...and GLiNER2 extracted it as an entity mentioned there
    edge       ...and that entity is an endpoint of an edge extracted there
    user edge  ...and that edge hangs off the user

Then it answers each question from two oracle retrievals, which skip the
retrieval agent entirely:

    edges      every typed edge extracted from the evidence sessions,
               with valid_at (the graph model's ceiling)
    text       the evidence sessions' full text (chunk-and-read's ceiling)

    python eval_diagnosis.py --memgraph-url bolt://localhost:7721 --out eval_diagnosis.json

Needs the typed-relation-model worktree on PYTHONPATH and OPENAI_API_KEY and
ANTHROPIC_API_KEY in the environment.
"""

import argparse
import asyncio
import json
import re
from collections import Counter

from context_graph_eval.cli import DEFAULT_AGENT_PROVIDER, _build_model, _parse_model_spec
from context_graph_eval.convert.longmemeval import DEFAULT_REVISION, fetch, haystack_path, load_raw
from context_graph_eval.corpus import read_corpus
from context_graph_eval.retrieval import DeepEvalLLM, Retrieved, answer_prompt
from context_graph_eval.runner import RunPlan, _score

from memgraph_toolbox.api.memgraph import Memgraph

CORPUS = "context-graph/eval/corpus/tier1-longmemeval.jsonl"
JUDGE = "anthropic:claude-sonnet-4-5-20250929"


def norm(text):
    return " ".join(re.sub(r"[^\w:$.,%-]+", " ", str(text).lower()).split())


def evidence(db, session_ids):
    """Chunks, entities and extracted edges of the evidence sessions."""
    chunks = db.query(
        """
        MATCH (s:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk)
        WHERE s.session_id IN $ids
        RETURN DISTINCT s.session_id AS sid, c.hash AS hash, c.text AS text
        """,
        params={"ids": session_ids},
    )
    hashes = [c["hash"] for c in chunks]
    entities = db.query(
        "MATCH (n:gliner2)-[:MENTIONED_IN]->(c:Chunk) WHERE c.hash IN $h RETURN DISTINCT n.text AS text, n.entity_type AS type",
        params={"h": hashes},
    )
    edges = db.query(
        """
        MATCH (a)-[r]->(b) WHERE r.chunk IN $h
        RETURN type(r) AS type, 'User' IN labels(a) AS head_is_user, coalesce(a.text, 'user') AS head,
               coalesce(b.text, 'user') AS tail, toString(r.valid_at) AS valid_at
        """,
        params={"h": hashes},
    )
    return chunks, entities, edges


def matches(entity_text, answer):
    e, a = norm(entity_text), norm(answer)
    return len(e) >= 3 and (e == a or (e in a and len(e) >= 0.5 * len(a)) or (a in e and len(a) >= 3))


def trace(answer, chunks, entities, edges):
    text = norm(" ".join(c["text"] for c in chunks))
    a = norm(answer)
    literal = len(a) <= 60 and a in text
    hit_entities = {e["text"] for e in entities if matches(e["text"], answer)}
    hit_edges = [e for e in edges if e["head"] in hit_entities or e["tail"] in hit_entities]
    return {
        "literal": literal,
        "entity": literal and bool(hit_entities),
        "edge": literal and bool(hit_edges),
        "user_edge": literal and any(e["head_is_user"] or e["tail"] == "user" for e in hit_edges),
        "matched_entities": sorted(hit_entities)[:5],
        "matched_edges": [f"{e['head']} -[{e['type']}]-> {e['tail']}" for e in hit_edges][:5],
    }


def edge_rows(edges):
    return [f"{e['head']} -[{e['type']} @ {e['valid_at']}]-> {e['tail']}" for e in edges]


async def answer_all(llm, goldens, contexts, concurrency):
    # Full evidence sessions run ~12k chars each; 8 at once exceeded the
    # agent model's tokens-per-minute limit.
    semaphore = asyncio.Semaphore(concurrency)

    async def one(golden, rows):
        async with semaphore:
            reply = await llm.complete(answer_prompt(golden.input, rows))
        return Retrieved(answer=reply.strip(), retrieval_context=rows)

    return await asyncio.gather(*(one(g, contexts[g.name]) for g in goldens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memgraph-url", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    db = Memgraph(url=args.memgraph_url, username="", password="")
    goldens = read_corpus(CORPUS)[:100]
    records = {
        r["question_id"]: r for r in load_raw(fetch("s", DEFAULT_REVISION, dest=haystack_path("s", DEFAULT_REVISION)))
    }

    traced, edge_ctx, text_ctx = {}, {}, {}
    for golden in goldens:
        record = records[golden.name]
        chunks, entities, edges = evidence(db, record["answer_session_ids"])
        traced[golden.name] = {
            "question_type": record["question_type"],
            "abstention": golden.name.endswith("_abs"),
            "answer": str(record["answer"]),
            "evidence_sessions": len(record["answer_session_ids"]),
            "evidence_chunks": len(chunks),
            "evidence_edges": len(edges),
            **trace(record["answer"], chunks, entities, edges),
        }
        edge_ctx[golden.name] = edge_rows(edges)
        text_ctx[golden.name] = [f"[session {c['sid']}] {c['text']}" for c in chunks]

    agent = _build_model(*_parse_model_spec(None, default_provider=DEFAULT_AGENT_PROVIDER))
    judge = _build_model(*_parse_model_spec(JUDGE, default_provider="anthropic"))
    plan = RunPlan(judge=judge)
    llm = DeepEvalLLM(agent)
    scored = {}
    for name, contexts, concurrency in (("edges", edge_ctx, 8), ("text", text_ctx, 2)):
        retrieved = asyncio.run(answer_all(llm, goldens, contexts, concurrency))
        for s in _score(goldens, retrieved, plan):
            traced[s.name][f"oracle_{name}_covered"] = s.covered
            traced[s.name][f"oracle_{name}_answer"] = s.answer[:200]
        scored[name] = sum(traced[g.name][f"oracle_{name}_covered"] for g in goldens)

    answerable = [t for t in traced.values() if not t["abstention"]]
    literal = [t for t in answerable if t["literal"]]
    summary = {
        "questions": len(traced),
        "non_abstention": len(answerable),
        "literal_answers": len(literal),
        "funnel_over_literal": {k: sum(t[k] for t in literal) for k in ("literal", "entity", "edge", "user_edge")},
        "oracle_covered": scored,
        "oracle_by_type": {
            name: dict(
                sorted(Counter(t["question_type"] for t in traced.values() if t[f"oracle_{name}_covered"]).items())
            )
            for name in scored
        },
        "questions_by_type": dict(sorted(Counter(t["question_type"] for t in traced.values()).items())),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "questions": traced}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
