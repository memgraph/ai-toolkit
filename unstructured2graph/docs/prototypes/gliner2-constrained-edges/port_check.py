"""Run the implemented GLiNER2Backend (feat/typed-relation-model, PR #373) over #350's 10 evidence sessions.

Checks the port against the prototype: the hand vocabulary with one turn per
window gave 2509 mentions here (windowing.py), and the three answer edges
(25:50, MoMA, Business Administration) must land on the user's node. Needs a
disposable Memgraph at MEMGRAPH_URL (it is wiped), the typed-relation-model
worktree's sources on PYTHONPATH, and ONTOLOGY pointing at its
context-graph/eval/src/context_graph_eval/ontologies/longmemeval.yaml.
"""

import asyncio
import hashlib
import json
import os
import time

import derive

from memgraph_toolbox.api.memgraph import Memgraph
from unstructured2graph import Document, Segment, from_documents, load_ontology, ontology_report
from unstructured2graph.gliner2_backend import GLiNER2Backend

ONTOLOGY = os.environ["ONTOLOGY"]
ANSWERS = {
    "25:50": "b.text CONTAINS '25:50'",
    "MoMA": "toLower(b.text) CONTAINS 'museum of modern art' OR toLower(b.text) = 'moma'",
    "Business Administration": "toLower(b.text) CONTAINS 'business administration'",
}


def document(sid, turns):
    """The session as reconciliation builds it: deduped role-prefixed turns, one segment each."""
    unique = {}
    for turn in turns:
        text = f"{turn['role']}: {turn['content']}".strip()
        if text:
            unique.setdefault(hashlib.sha256(text.encode()).hexdigest(), (text, turn["role"]))
    segments, cursor = [], 0
    for text, role in unique.values():
        segments.append(Segment(cursor, cursor + len(text), role, "2023-05-30T00:00:00+00:00"))
        cursor += len(text) + 2
    return Document("\n\n".join(t for t, _ in unique.values()), tuple(segments), f"anon-{sid}")


async def main():
    sample = json.loads((derive.HERE / "sample_sessions.json").read_text())
    docs = derive.load_texts({s["session_id"] for q in sample for s in q["sessions"]})
    memgraph = Memgraph()
    memgraph.query("MATCH (n) DETACH DELETE n")
    for sid in docs:
        memgraph.query("MERGE (:User {user_id: $u})", params={"u": f"anon-{sid}"})
    ontology = load_ontology(ONTOLOGY)
    backend = GLiNER2Backend(ontology=ontology)
    started = time.perf_counter()
    await from_documents(
        [document(sid, turns) for sid, (_, turns) in docs.items()],
        memgraph,
        backend,
        enforce_ontology=True,
        ontology_path=ONTOLOGY,
    )
    print(f"seconds {time.perf_counter() - started:.0f}; stats {backend.stats}")
    report = ontology_report(memgraph, "gliner2", ontology)
    print(
        f"relationships {report.relationships}, declared {report.declared_relationships}, "
        f"nonconformant entities {report.nonconformant_entities}, relations {report.nonconformant_relations}, "
        f"zero-instance {report.zero_instance_relation_types}"
    )
    for label, where in ANSWERS.items():
        rows = memgraph.query(
            f"MATCH (a)-[r]->(b:gliner2) WHERE {where} "
            "RETURN type(r) AS type, coalesce(a.user_id, a.text) AS head, b.text AS tail"
        )
        print(label, rows)
    print("edges onto User:", memgraph.query("MATCH (:User)-[r]->() RETURN count(r) AS n")[0]["n"])


if __name__ == "__main__":
    asyncio.run(main())
