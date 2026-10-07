"""PROTOTYPE — throwaway. The graph model under test (#460), written to a scratch Memgraph.

Shape:
  (:User)-[:HAD_SESSION]->(:Session)<-[:IN_SESSION]-(:Touch)-[:TOUCHED]->(:Resource | :Listing)
  (:Resource:Repository)-[:HAS_ISSUE|HAS_PULL_REQUEST]->(:Resource:Issue|PullRequest)-[:HAS_COMMENT]->(:Comment)
  (:Listing)-[:LISTS_FROM]->(:Repository), (:Listing)-[:HAS_MEMBER]->(:Issue|PullRequest)
  (:Resource)-[:REFERENCES]->(:Resource), (:PullRequest)-[:CLOSES]->(:Issue)   -- only between stored Resources
Resources and Listings belong to no user; Touches are private via their Session.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from address import Address, member_matches, parse, subsumes
from neo4j import GraphDatabase

URI = "bolt://localhost:7771"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, uri: str = URI):
        self.driver = GraphDatabase.driver(uri, auth=("", ""))
        for stmt in (
            "CREATE INDEX ON :Resource(node_id)",
            "CREATE INDEX ON :Resource(address)",
            "CREATE INDEX ON :Listing(key)",
            "CREATE INDEX ON :Touch(status)",
            "CREATE INDEX ON :Session(session_id)",
        ):
            try:
                self.run(stmt)
            except Exception:  # noqa: BLE001 — index already exists
                pass

    def run(self, cypher: str, **params) -> list[dict]:
        with self.driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

    def wipe(self) -> None:
        self.run("MATCH (n) DETACH DELETE n")

    # -- hook side: address-only Touches -------------------------------------------------

    def touch(
        self, session_id: str, addr: Address, provenance: str, tool_use_id: str | None = None, user: str = "user-a"
    ) -> str:
        """What a hook writes: an address-only, pending Touch. No GitHub call."""
        touch_id = str(uuid.uuid4())
        self.run(
            """MERGE (u:User {user_id: $user})
               MERGE (s:Session {session_id: $sid}) MERGE (u)-[:HAD_SESSION]->(s)
               CREATE (t:Touch {touch_id: $tid, address: $key, address_kind: $kind, limit: $limit,
                                provenance: $prov, served_from_memory: false, status: 'pending',
                                tool_use_id: $tuid, at: $at})-[:IN_SESSION]->(s)""",
            sid=session_id,
            tid=touch_id,
            key=addr.key,
            kind=addr.kind,
            limit=addr.limit,
            prov=provenance,
            tuid=tool_use_id,
            user=user,
            at=now(),
        )
        return touch_id

    # -- read side: resource(address) -------------------------------------------------

    def cache_read(self, session_id: str, addr: Address, user: str = "user-a") -> dict:
        """The ``resource`` tool. Never calls GitHub; records a served-from-memory Touch."""
        result = self._serve(addr)
        self.run(
            """MERGE (u:User {user_id: $user})
               MERGE (s:Session {session_id: $sid}) MERGE (u)-[:HAD_SESSION]->(s)
               CREATE (t:Touch {touch_id: $tid, address: $key, address_kind: $kind, limit: $limit,
                                provenance: 'FETCHED', served_from_memory: true, outcome: $outcome,
                                status: 'resolved', at: $at})-[:IN_SESSION]->(s)
               WITH t OPTIONAL MATCH (target) WHERE (target:Resource AND target.address = $target)
                                                  OR (target:Listing AND target.key = $target)
               FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END | CREATE (t)-[:TOUCHED]->(target))""",
            sid=session_id,
            tid=str(uuid.uuid4()),
            key=addr.key,
            kind=addr.kind,
            limit=addr.limit,
            outcome=result["outcome"],
            target=result.get("served_from"),
            user=user,
            at=now(),
        )
        return result

    def _serve(self, addr: Address) -> dict:
        if addr.kind in ("item", "repo"):
            rows = self.run(
                """MATCH (r:Resource {address: $a})
                   OPTIONAL MATCH (r)-[:HAS_COMMENT]->(c:Comment)
                   RETURN r, count(c) AS comments""",
                a=addr.key,
            )
            if not rows:
                return {"outcome": "miss"}
            r = rows[0]["r"]
            return {
                "outcome": "hit",
                "served_from": addr.key,
                "facts": {k: r.get(k) for k in ("title", "state", "fetched_at", "updated_at")},
                "comments": rows[0]["comments"],
            }
        exact = self.run("MATCH (l:Listing {key: $k}) RETURN l", k=addr.key)
        if exact:
            listing = exact[0]["l"]
            if listing["fully_expanded"] or (addr.limit and addr.limit <= listing["member_count"]):
                return self._index(listing, addr, "hit")
        candidates = self.run(
            "MATCH (l:Listing {repo: $repo, item_kind: $ik, fully_expanded: true}) RETURN l ORDER BY l.member_count",
            repo=addr.repo_key,
            ik=addr.item_kind,
        )
        for row in candidates:
            if subsumes(parse_listing(row["l"]), addr):
                return self._index(row["l"], addr, "subsumed")
        return {"outcome": "miss"}

    def _index(self, listing: dict, addr: Address, outcome: str) -> dict:
        members = self.run(
            """MATCH (:Listing {key: $k})-[:HAS_MEMBER]->(m)
               RETURN m.number AS number, m.title AS title, m.state AS state, m.labels AS labels,
                      m.author AS author, m.assignees AS assignees, m.milestone AS milestone,
                      m.updated_at AS updated_at, m.comment_count AS comment_count
               ORDER BY m.updated_at DESC""",
            k=listing["key"],
        )
        if outcome == "subsumed":
            members = [m for m in members if member_matches(m, addr)]
        if addr.limit:
            members = members[: addr.limit]
        return {
            "outcome": outcome,
            "served_from": listing["key"],
            "facts": {"fetched_at": listing["fetched_at"], "member_count": len(members)},
            "index": members,
        }

    # -- Sweep side: resolve pending Touches into Resources ----------------------------

    def pending(self) -> list[dict]:
        return self.run("MATCH (t:Touch {status: 'pending'}) RETURN t ORDER BY t.at")

    def resolve_touch(self, touch_id: str, target_key: str) -> None:
        self.run(
            """MATCH (t:Touch {touch_id: $tid})
               OPTIONAL MATCH (target) WHERE (target:Resource AND target.address = $k) OR (target:Listing AND target.key = $k)
               SET t.status = CASE WHEN target IS NULL THEN 'unresolved' ELSE 'resolved' END
               FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END | MERGE (t)-[:TOUCHED]->(target))""",
            tid=touch_id,
            k=target_key,
        )

    def unresolve_touch(self, touch_id: str, reason: str) -> None:
        self.run("MATCH (t:Touch {touch_id: $tid}) SET t.status = 'unresolved', t.reason = $r", tid=touch_id, r=reason)

    def upsert_repo(self, node: dict) -> str:
        key = f"github:{node['nameWithOwner']}".lower()
        self.run(
            """MERGE (r:Resource:Repository {node_id: $id})
               SET r.address = $key, r.name_with_owner = $nwo, r.url = $url, r.description = $d,
                   r.readme = $readme, r.updated_at = $u, r.pushed_at = $p, r.stars = $stars, r.fetched_at = $at""",
            id=node["id"],
            key=key,
            nwo=node["nameWithOwner"],
            url=node["url"],
            d=node["description"],
            readme=(node.get("readme") or {}).get("text"),
            u=node["updatedAt"],
            p=node["pushedAt"],
            stars=node["stargazerCount"],
            at=now(),
        )
        return key

    def upsert_item(self, node: dict) -> tuple[str, str | None]:
        """Upsert one Issue/PR at full capture depth. Returns (address, previous updated_at)."""
        repo = node["repository"]
        key = f"github:{repo['nameWithOwner']}#{node['number']}".lower()
        label = "PullRequest" if node["__typename"] == "PullRequest" else "Issue"
        rel = "HAS_PULL_REQUEST" if label == "PullRequest" else "HAS_ISSUE"
        prev = self.run("MATCH (r:Resource {node_id: $id}) RETURN r.updated_at AS u", id=node["id"])
        props = {
            "address": key,
            "number": node["number"],
            "url": node["url"],
            "title": node["title"],
            "body": node["body"],
            "state": node["state"],
            "author": (node.get("author") or {}).get("login"),
            "labels": [x["name"] for x in node["labels"]["nodes"]],
            "assignees": [x["login"] for x in node["assignees"]["nodes"]],
            "milestone": (node.get("milestone") or {}).get("title"),
            "created_at": node["createdAt"],
            "updated_at": node["updatedAt"],
            "closed_at": node["closedAt"],
            "comment_count": node["comments"]["totalCount"],
            "comments_truncated": node["comments"]["totalCount"] > len(node["comments"]["nodes"]),
            "fetched_at": now(),
        }
        if label == "PullRequest":
            props |= {
                "merged": node["merged"],
                "base_ref": node["baseRefName"],
                "changed_files": [f["path"] for f in node["files"]["nodes"]],
            }
        self.run(
            f"""MERGE (repo:Resource:Repository {{node_id: $rid}})
                ON CREATE SET repo.address = $rkey, repo.name_with_owner = $nwo
                MERGE (r:Resource:{label} {{node_id: $id}}) SET r += $props
                MERGE (repo)-[:{rel}]->(r)""",
            rid=repo["id"],
            rkey=f"github:{repo['nameWithOwner']}".lower(),
            nwo=repo["nameWithOwner"],
            id=node["id"],
            props=props,
        )
        comments = [(c, None) for c in node["comments"]["nodes"]]
        for thread in (node.get("reviewThreads") or {}).get("nodes", []):
            comments += [(c, thread["path"]) for c in thread["comments"]["nodes"]]
        self.run(
            """MATCH (r:Resource {node_id: $id})
               UNWIND $comments AS c
               MERGE (n:Comment {node_id: c.id})
               SET n.body = c.body, n.author = c.author, n.created_at = c.created_at,
                   n.updated_at = c.updated_at, n.url = c.url, n.path = c.path, n.review = c.path IS NOT NULL
               MERGE (r)-[:HAS_COMMENT]->(n)""",
            id=node["id"],
            comments=[
                {
                    "id": c["id"],
                    "body": c["body"],
                    "author": (c.get("author") or {}).get("login"),
                    "created_at": c["createdAt"],
                    "updated_at": c["updatedAt"],
                    "url": c["url"],
                    "path": path,
                }
                for c, path in comments
            ],
        )
        refs = [n["source"] for n in node["timelineItems"]["nodes"] if n and n.get("source")]
        closes = (node.get("closingIssuesReferences") or {}).get("nodes", [])
        self.run(
            """MATCH (r:Resource {node_id: $id})
               SET r.referenced_by_urls = $ref_urls, r.closes_urls = $close_urls
               WITH r UNWIND $refs AS ref MATCH (src:Resource {node_id: ref}) MERGE (src)-[:REFERENCES]->(r)""",
            id=node["id"],
            refs=[x["id"] for x in refs],
            ref_urls=[x["url"] for x in refs],
            close_urls=[x["url"] for x in closes],
        )
        self.run(
            """MATCH (pr:Resource {node_id: $id}) UNWIND $closes AS cid
               MATCH (i:Resource {node_id: cid}) MERGE (pr)-[:CLOSES]->(i)""",
            id=node["id"],
            closes=[x["id"] for x in closes],
        )
        # Backfill: stored Resources whose recorded references point at this one, now that it exists.
        self.run(
            """MATCH (r:Resource {node_id: $id}), (other:Resource)
               WHERE $url IN coalesce(other.referenced_by_urls, []) MERGE (r)-[:REFERENCES]->(other)""",
            id=node["id"],
            url=node["url"],
        )
        self.run(
            """MATCH (r:Resource {node_id: $id}), (pr:PullRequest)
               WHERE $url IN coalesce(pr.closes_urls, []) MERGE (pr)-[:CLOSES]->(r)""",
            id=node["id"],
            url=node["url"],
        )
        return key, prev[0]["u"] if prev else None

    def upsert_listing(self, addr: Address, total: int, members: list[str]) -> str:
        fully = addr.limit is None or len(members) >= total
        self.run(
            """MATCH (repo:Repository {address: $repo})
               MERGE (l:Listing {key: $k})
               SET l.repo = $repo, l.item_kind = $ik, l.filters = $f, l.query = $q, l.limit = $limit,
                   l.total_count = $total, l.member_count = size($members), l.fully_expanded = $fully, l.fetched_at = $at
               MERGE (l)-[:LISTS_FROM]->(repo)
               WITH l OPTIONAL MATCH (l)-[old:HAS_MEMBER]->() DELETE old
               WITH DISTINCT l UNWIND $members AS a MATCH (m:Resource {address: a}) MERGE (l)-[:HAS_MEMBER]->(m)""",
            repo=addr.repo_key,
            k=addr.key,
            ik=addr.item_kind,
            f=json.dumps(addr.filter_dict()),
            q=addr.query,
            limit=addr.limit,
            total=total,
            members=members,
            fully=fully,
            at=now(),
        )
        return addr.key

    # -- inspection -------------------------------------------------------------------

    def shape(self) -> dict:
        labels = self.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY l")
        rels = self.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t")
        touches = self.run(
            """MATCH (t:Touch) RETURN t.provenance AS prov, t.served_from_memory AS mem,
               coalesce(t.outcome, '-') AS outcome, t.status AS status, coalesce(t.reason, '-') AS reason, count(*) AS c"""
        )
        return {"labels": labels, "rels": rels, "touches": touches}


def parse_listing(row: dict) -> Address:
    owner, repo = row["repo"].removeprefix("github:").split("/")
    return Address(
        "listing",
        owner,
        repo,
        item_kind=row["item_kind"],
        filters=tuple(sorted(json.loads(row["filters"]).items())),
        query=row.get("query"),
        limit=row.get("limit"),
    )


def address_of(touch: dict) -> Address:
    """Rebuild an Address from a pending Touch (the Sweep's input)."""
    key, limit = touch["address"], touch.get("limit")
    body = key.removeprefix("github:")
    if touch["address_kind"] == "item":
        repo, num = body.split("#")
        a = parse(f"{repo}#{num}")
        return a
    if touch["address_kind"] == "repo":
        owner, repo = body.split("/")
        return Address("repo", owner, repo)
    path, _, qs = body.partition("?")
    owner, repo, item_kind = path.split("/")
    pairs = dict(p.split("=", 1) for p in qs.split("&") if p)
    query = pairs.pop("q", None)
    return Address("listing", owner, repo, item_kind=item_kind, filters=tuple(sorted(pairs.items())), query=query, limit=limit)
