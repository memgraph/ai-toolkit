"""PROTOTYPE — throwaway TUI over the GitHub Resource graph model (#460).

Run (from the repo root):
  docker run -d --name ai-toolkit-460-proto-wipe-me-7771 -p 7771:7687 memgraph/memgraph:latest
  uv run --no-project --with neo4j python context-graph/resources-graph/prototype_graph_model/tui.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import github  # noqa: E402
from address import addresses_in_prompt, parse  # noqa: E402
from store import Store  # noqa: E402
from sweep import sweep  # noqa: E402

B, D, R = "\x1b[1m", "\x1b[2m", "\x1b[0m"

KEYS = [
    ("f", "fetch: FETCHED touch (URL, owner/repo#n, gh ...)"),
    ("p", "prompt: PROMPTED touches from pasted text"),
    ("s", "run Sweep"),
    ("r", "resource(address) cache read"),
    ("x", "show a Resource"),
    ("u", "updatedAt probe"),
    ("n", "new session"),
    ("w", "wipe graph"),
    ("q", "quit"),
]


def frame(store: Store, session: str, out: list[str]) -> None:
    print("\x1b[2J\x1b[H", end="")
    shape = store.shape()
    print(f"{B}session{R} {D}{session}{R}")
    print(f"{B}nodes{R}   " + "  ".join(f"{r['l']}={r['c']}" for r in shape["labels"]))
    print(f"{B}rels{R}    " + "  ".join(f"{r['t']}={r['c']}" for r in shape["rels"]))
    print(f"{B}touches{R}")
    for t in shape["touches"]:
        src = "memory" if t["mem"] else "fetch "
        print(f"  {t['prov']:<8} {src} outcome={t['outcome']:<8} {t['status']:<10} {D}{t['reason']}{R}  x{t['c']}")
    print(f"\n{B}last{R}")
    for line in out[-22:]:
        print("  " + line)
    print("\n" + "  ".join(f"{B}[{k}]{R} {D}{d}{R}" for k, d in KEYS))


def main() -> None:
    store = Store()
    session = f"proto-{uuid.uuid4().hex[:6]}"
    out: list[str] = ["fresh session"]
    while True:
        frame(store, session, out)
        key = input("> ").strip().lower()
        out = []
        if key == "q":
            return
        if key == "f":
            addr = parse(input("address> "))
            if addr is None:
                out.append("not a GitHub address")
                continue
            store.touch(session, addr, "FETCHED", tool_use_id=f"toolu_{uuid.uuid4().hex[:8]}")
            out.append(f"pending touch {addr.kind} {addr.key} limit={addr.limit}")
        elif key == "p":
            for addr in addresses_in_prompt(input("prompt> ")):
                store.touch(session, addr, "PROMPTED")
                out.append(f"pending PROMPTED touch {addr.key}")
        elif key == "s":
            sweep(store, out.append)
        elif key == "r":
            addr = parse(input("address> "))
            if addr is None:
                out.append("not a GitHub address")
                continue
            res = store.cache_read(session, addr)
            out.append(f"outcome={res['outcome']}  served_from={res.get('served_from')}")
            out.append(f"facts={json.dumps(res.get('facts'))}")
            for m in (res.get("index") or [])[:15]:
                out.append(f"  #{m['number']:<6} {m['state']:<7} {D}{','.join(m['labels'])[:30]:<30}{R} {m['title'][:60]}")
            if res.get("index") and len(res["index"]) > 15:
                out.append(f"  ... {len(res['index']) - 15} more")
        elif key == "x":
            addr = parse(input("address> "))
            rows = store.run(
                """MATCH (r:Resource {address: $a})
                   OPTIONAL MATCH (r)-[:HAS_COMMENT]->(c) WITH r, count(c) AS comments
                   OPTIONAL MATCH (r)-[ref:REFERENCES|CLOSES]-(o:Resource) WITH r, comments, collect(type(ref) + ' ' + o.address) AS links
                   OPTIONAL MATCH (l:Listing)-[:HAS_MEMBER]->(r) WITH r, comments, links, collect(l.key) AS listings
                   OPTIONAL MATCH (t:Touch)-[:TOUCHED]->(r)
                   RETURN r, comments, links, listings, count(t) AS touches""",
                a=addr.key if addr else "",
            )
            if not rows:
                out.append("not in memory")
                continue
            row = rows[0]
            for k, v in sorted(row["r"].items()):
                if k in ("body", "readme"):
                    v = (v or "")[:80].replace("\n", " ") + f"... ({len(v or '')} chars)"
                out.append(f"{k}: {v}")
            out += [f"comments: {row['comments']}", f"links: {row['links']}", f"listings: {row['listings']}"]
            out.append(f"direct touches: {row['touches']}")
        elif key == "u":
            rows = github.updated_at_probe("memgraph", "memgraph")
            for field in ("reaction", "label", "comment_reaction", "review_comment"):
                seen = [r for r in rows if r.get(field)]
                newer = [r for r in seen if r[field] > r["updatedAt"]]
                out.append(f"{field:<17} seen={len(seen):<3} newer-than-updatedAt={len(newer)}")
        elif key == "n":
            session = f"proto-{uuid.uuid4().hex[:6]}"
            out.append("new session")
        elif key == "w":
            store.wipe()
            out.append("wiped")


if __name__ == "__main__":
    main()
