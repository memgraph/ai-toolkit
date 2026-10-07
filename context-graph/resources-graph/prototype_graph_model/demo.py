"""PROTOTYPE — throwaway. Scripted run of the "500 Memgraph issues, twice" story against the scratch Memgraph.

  uv run --no-project --with neo4j python context-graph/resources-graph/prototype_graph_model/demo.py [limit]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from address import addresses_in_prompt, parse  # noqa: E402
from store import Store  # noqa: E402
from sweep import sweep  # noqa: E402

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 500
store = Store()
store.wipe()

print(f"== session A: agent lists open issues (limit {limit}), user pastes a PR URL, agent opens an issue")
store.touch("A", parse(f"gh issue list -R memgraph/memgraph --state open --limit {limit}"), "FETCHED", "toolu_a1")
for a in addresses_in_prompt("look at https://github.com/memgraph/memgraph/pull/3600 please"):
    store.touch("A", a, "PROMPTED")
store.touch("A", parse("memgraph/memgraph#2000"), "FETCHED", "toolu_a2")
store.touch("A", parse("gh issue view 999999 -R memgraph/memgraph"), "FETCHED", "toolu_a3")
started = time.monotonic()
sweep(store, lambda line: None)
print(f"sweep took {time.monotonic() - started:.0f}s")

print("== session B: same questions again, served from memory")
for text in (
    f"gh issue list -R memgraph/memgraph --state open --limit {limit}",
    "gh issue list -R memgraph/memgraph --state open --label bug",
    "gh issue list -R memgraph/memgraph --state open --label bug --label 'priority/p1'",
    "gh issue list -R memgraph/memgraph --state closed",
    "gh issue list -R memgraph/memgraph --search replication",
    "https://github.com/memgraph/memgraph/issues/2000",
    "https://github.com/memgraph/memgraph/pull/3600",
    "memgraph/memgraph#1",
):
    res = store.cache_read("B", parse(text), user="user-b")
    print(f"  {res['outcome']:<9} {len(res.get('index') or []):>4} rows  {text}")

shape = store.shape()
print("== shape")
for r in shape["labels"]:
    print(f"  {r['l']:<12} {r['c']}")
for r in shape["rels"]:
    print(f"  {r['t']:<16} {r['c']}")
for t in shape["touches"]:
    print(f"  touch {t}")
print("== sizes")
print(
    store.run(
        """MATCH (r:Resource) RETURN sum(size(coalesce(r.body, ''))) AS item_body_chars,
           sum(size(coalesce(r.readme, ''))) AS readme_chars"""
    )
)
print(store.run("MATCH (c:Comment) RETURN count(c) AS comments, sum(size(coalesce(c.body, ''))) AS comment_chars"))
print(store.run("MATCH (r:Resource) WHERE r.comments_truncated RETURN count(r) AS items_with_over_100_comments"))
print(store.run("MATCH ()-[x:REFERENCES|CLOSES]->() RETURN type(x) AS t, count(*) AS edges"))
print(
    store.run(
        """MATCH (r:Resource) RETURN sum(size(coalesce(r.referenced_by_urls, []))) AS recorded_refs,
           sum(size(coalesce(r.closes_urls, []))) AS recorded_closes"""
    )
)
