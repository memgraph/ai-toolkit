"""ResourcesGraph: Touches, shared Resources and Cache Reads in Memgraph.

Graph model::

    (:Session)<-[:IN_SESSION]-(:Touch)-[:TOUCHED]->(:Resource | :Listing)
    (:Resource:Repository)-[:HAS_ISSUE|HAS_PULL_REQUEST]->(:Resource:Issue|PullRequest)
    (:Resource:Issue|PullRequest)-[:HAS_COMMENT]->(:Comment)
    (:Listing)-[:LISTS_FROM]->(:Repository), (:Listing)-[:HAS_MEMBER {position}]->(:Issue|PullRequest)
    (:Resource)-[:REFERENCES]->(:Resource), (:PullRequest)-[:CLOSES]->(:Issue)   -- between stored Resources only
    (:Touch)-[:CAUSED_BY]->(:ToolCall), (:Touch)-[:BY_AGENT]->(:Agent)           -- actions-graph's nodes, joined softly
    (:Resource)-[:MOVED_TO]->(:Resource)   -- a transferred issue: GitHub gives it a new node_id

Resources, Listings and Comments belong to no user. A Touch is private to the user
whose Session it sits in; sessions-graph owns ``(:User)-[:HAD_SESSION]->``,
this component only MERGEs the shared ``(:Session {session_id})``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from memgraph_toolbox.api.memgraph import Memgraph

from .address import Address
from .listing import exact_serves, member_matches, subsumes
from .models import FETCHED, HIT, MISS, SUBSUMED, Served

PENDING = "pending"
RESOLVED = "resolved"
UNRESOLVED = "unresolved"
RATE_LIMITED = "rate_limited"

_SCHEMA = (
    "CREATE CONSTRAINT ON (r:Resource) ASSERT r.node_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (c:Comment) ASSERT c.node_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (t:Touch) ASSERT t.touch_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (l:Listing) ASSERT l.key IS UNIQUE;",
    "CREATE INDEX ON :Listing(key);",
    "CREATE INDEX ON :Listing(scope);",
    "CREATE INDEX ON :Resource(node_id);",
    "CREATE INDEX ON :Resource(address);",
    "CREATE INDEX ON :Comment(node_id);",
    "CREATE INDEX ON :Touch(touch_id);",
    "CREATE INDEX ON :Touch(status);",
    "CREATE INDEX ON :Session(session_id);",
)


def now() -> str:
    """The current UTC time, ISO-8601, as every timestamp here is stored."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def touch_id(session_id: str, discriminator: str, provenance: str, address: Address) -> str:
    """A stable Touch id, so a re-delivered hook event writes the same Touch once."""
    raw = "|".join((session_id, discriminator, provenance, address.key))
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class ResourcesGraph:
    """Persist and serve Resources the agent touched."""

    def __init__(self, memgraph: Memgraph | None = None, **kwargs: Any) -> None:
        """Connect to Memgraph.

        Args:
            memgraph: An existing client; when omitted one is created from ``kwargs``
                (``url``, ``username``, ``password``, ``database``) or the environment.
        """
        self._db = memgraph or Memgraph(**kwargs)

    def setup(self) -> None:
        """Create the constraints and indexes this component relies on. Idempotent."""
        for statement in _SCHEMA:
            self._db.query(statement)

    # -- hook side: address-only Touches, no network -------------------------

    def record_touch(
        self,
        session_id: str,
        address: Address,
        provenance: str,
        *,
        discriminator: str,
        tool_use_id: str | None = None,
        agent_name: str | None = None,
        at: str | None = None,
    ) -> str:
        """Record that the agent touched ``address``; the Sweep resolves it later.

        ``discriminator`` (the tool use id, or the prompt's timestamp) keeps two
        touches of one Address in a session apart while making re-delivery idempotent.
        Returns the Touch id.
        """
        tid = touch_id(session_id, discriminator, provenance, address)
        self._db.query(
            """
            MERGE (s:Session {session_id: $session_id})
            MERGE (t:Touch {touch_id: $touch_id})
            ON CREATE SET t.address = $address, t.address_kind = $kind, t.limit = $limit, t.provenance = $provenance,
                          t.served_from_memory = false, t.status = $pending, t.tool_use_id = $tool_use_id,
                          t.agent_name = $agent_name, t.at = $at
            MERGE (t)-[:IN_SESSION]->(s)
            """,
            params={
                "session_id": session_id,
                "touch_id": tid,
                "address": address.key,
                "kind": address.kind,
                "limit": address.limit,
                "provenance": provenance,
                "pending": PENDING,
                "tool_use_id": tool_use_id,
                "agent_name": agent_name,
                "at": at or now(),
            },
        )
        return tid

    def record_cache_read(
        self,
        session_id: str,
        address: Address,
        outcome: str,
        *,
        discriminator: str,
        served_from: str | None = None,
        tool_use_id: str | None = None,
        agent_name: str | None = None,
        at: str | None = None,
    ) -> str:
        """Record a Cache Read: a FETCHED Touch served from memory, with its outcome.

        ``served_from`` is the key of what answered it (a Listing's own, or the
        broader Listing that subsumed it); the Touch points there. It never
        becomes pending, so the Sweep never refreshes anything because of it.
        Returns the Touch id.
        """
        tid = touch_id(session_id, discriminator, "MEMORY", address)
        self._db.query(
            """
            MERGE (s:Session {session_id: $session_id})
            MERGE (t:Touch {touch_id: $touch_id})
            ON CREATE SET t.address = $address, t.address_kind = $kind, t.limit = $limit, t.provenance = $provenance,
                          t.served_from_memory = true, t.outcome = $outcome, t.served_from = $served_from,
                          t.tool_use_id = $tool_use_id, t.agent_name = $agent_name, t.at = $at
            MERGE (t)-[:IN_SESSION]->(s)
            WITH t
            OPTIONAL MATCH (r:Resource {address: $served_from})
            OPTIONAL MATCH (l:Listing {key: $served_from})
            FOREACH (target IN [x IN [r, l] WHERE x IS NOT NULL AND $outcome <> $miss] |
                MERGE (t)-[:TOUCHED]->(target))
            """,
            params={
                "session_id": session_id,
                "touch_id": tid,
                "address": address.key,
                "kind": address.kind,
                "provenance": FETCHED,
                "limit": address.limit,
                "outcome": outcome,
                "served_from": served_from or address.key,
                "miss": MISS,
                "tool_use_id": tool_use_id,
                "agent_name": agent_name,
                "at": at or now(),
            },
        )
        return tid

    # -- read side: the resource tool ----------------------------------------

    def read(self, address: Address) -> Served:
        """Serve ``address`` from memory. Never calls GitHub and writes nothing."""
        if address.kind == "listing":
            return self._read_listing(address)
        rows = self._db.query(
            """
            MATCH (r:Resource {address: $address})
            WHERE r.fetched_at IS NOT NULL
            OPTIONAL MATCH (r)-[:HAS_COMMENT]->(c:Comment)
            WITH r, c ORDER BY c.created_at
            RETURN properties(r) AS resource, labels(r) AS labels, collect(properties(c)) AS comments
            """,
            params={"address": address.key},
        )
        if not rows:
            return Served(address=address.key, outcome=MISS)
        row = rows[0]
        kind = next((label for label in row["labels"] if label != "Resource"), None)
        return Served(address=address.key, outcome=HIT, kind=kind, resource=row["resource"], comments=row["comments"])

    def _read_listing(self, address: Address) -> Served:
        """An exact or subsumed Listing, as an index of its members (see :mod:`resources_graph.listing`)."""
        rows = self._db.query(
            "MATCH (l:Listing {key: $key}) RETURN properties(l) AS listing", params={"key": address.key}
        )
        if rows and exact_serves(
            fully_expanded=rows[0]["listing"]["fully_expanded"],
            member_count=rows[0]["listing"]["member_count"],
            wanted=address,
        ):
            return self._served_listing(address, HIT, rows[0]["listing"], self._members(address.key))
        candidates = self._db.query(
            """
            MATCH (l:Listing {scope: $scope})
            WHERE l.fully_expanded AND l.key <> $key
            RETURN properties(l) AS listing
            ORDER BY l.member_count
            """,
            params={"scope": _scope(address), "key": address.key},
        )
        for row in candidates:
            if subsumes(Address.from_key(row["listing"]["key"]), address):
                members = [m for m in self._members(row["listing"]["key"]) if member_matches(m, address)]
                return self._served_listing(address, SUBSUMED, row["listing"], members)
        return Served(address=address.key, outcome=MISS, kind="Listing")

    def _members(self, key: str) -> list[dict[str, Any]]:
        return self._db.query(
            """
            MATCH (:Listing {key: $key})-[m:HAS_MEMBER]->(r:Resource)
            RETURN r.number AS number, r.title AS title, r.state AS state, r.labels AS labels,
                   r.author AS author, r.assignees AS assignees, r.milestone AS milestone,
                   r.updated_at AS updated_at, r.comment_count AS comment_count, r.address AS address
            ORDER BY m.position
            """,
            params={"key": key},
        )

    @staticmethod
    def _served_listing(
        address: Address, outcome: str, listing: dict[str, Any], members: list[dict[str, Any]]
    ) -> Served:
        if address.limit is not None:
            members = members[: address.limit]
        return Served(
            address=address.key,
            outcome=outcome,
            kind="Listing",
            resource=listing,
            index=members,
            served_from=listing["key"],
        )

    def has_listing(self, address: Address) -> bool:
        """Whether ``address`` is stored as a Listing that answers it exactly."""
        rows = self._db.query(
            "MATCH (l:Listing {key: $key}) RETURN l.fully_expanded AS fully_expanded, l.member_count AS member_count",
            params={"key": address.key},
        )
        return bool(rows) and exact_serves(
            fully_expanded=rows[0]["fully_expanded"], member_count=rows[0]["member_count"], wanted=address
        )

    def in_memory(self, address: Address) -> dict[str, Any] | None:
        """What a Nudge says about ``address``: whether memory answers it, and the facts — never content.

        Cheap enough for a hook: indexed lookups only, no member rows. Returns
        ``{"outcome", "fetched_at", "updated_at"}`` (``served_from`` for a
        subsumed Listing), or None when memory would miss.
        """
        if address.kind != "listing":
            rows = self._db.query(
                "MATCH (r:Resource {address: $address}) WHERE r.fetched_at IS NOT NULL "
                "RETURN r.fetched_at AS fetched_at, r.updated_at AS updated_at",
                params={"address": address.key},
            )
            return {"outcome": HIT, **rows[0]} if rows else None
        rows = self._db.query(
            "MATCH (l:Listing {key: $key}) RETURN properties(l) AS listing", params={"key": address.key}
        )
        if rows and exact_serves(
            fully_expanded=rows[0]["listing"]["fully_expanded"],
            member_count=rows[0]["listing"]["member_count"],
            wanted=address,
        ):
            return {"outcome": HIT, "fetched_at": rows[0]["listing"]["fetched_at"], "updated_at": None}
        candidates = self._db.query(
            "MATCH (l:Listing {scope: $scope}) WHERE l.fully_expanded RETURN l.key AS key, l.fetched_at AS fetched_at",
            params={"scope": _scope(address)},
        )
        for row in candidates:
            if row["key"] != address.key and subsumes(Address.from_key(row["key"]), address):
                return {
                    "outcome": SUBSUMED,
                    "fetched_at": row["fetched_at"],
                    "updated_at": None,
                    "served_from": row["key"],
                }
        return None

    def has_listing_key(self, address: Address) -> bool:
        """Whether any Listing is stored under ``address``'s key, however many members it holds."""
        rows = self._db.query("MATCH (l:Listing {key: $key}) RETURN count(l) AS n", params={"key": address.key})
        return bool(rows and rows[0]["n"])

    def has_resource(self, address: Address) -> bool:
        """Whether ``address`` is stored with content."""
        rows = self._db.query(
            "MATCH (r:Resource {address: $address}) WHERE r.fetched_at IS NOT NULL RETURN count(r) AS n",
            params={"address": address.key},
        )
        return bool(rows and rows[0]["n"])

    def touches(self, user_id: str | None) -> list[dict[str, Any]]:
        """``user_id``'s own Touches, newest first; nobody else's, and none at all without a user."""
        if not user_id:
            return []
        return self._db.query(
            """
            MATCH (:User {user_id: $user_id})-[:HAD_SESSION]->(s:Session)<-[:IN_SESSION]-(t:Touch)
            RETURN properties(t) AS touch, s.session_id AS session_id
            ORDER BY t.at DESC
            """,
            params={"user_id": user_id},
        )

    # -- Sweep side -----------------------------------------------------------

    def pending_touches(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Touches waiting for the Sweep, oldest first — rate-limited ones included, to retry."""
        return self._db.query(
            f"""
            MATCH (t:Touch)
            WHERE t.status = $pending OR (t.status = $unresolved AND t.reason = $rate_limited)
            RETURN t.touch_id AS touch_id, t.address AS address, t.provenance AS provenance, t.limit AS limit
            ORDER BY t.at
            {"LIMIT $limit" if limit else ""}
            """,
            params={"pending": PENDING, "unresolved": UNRESOLVED, "rate_limited": RATE_LIMITED, "limit": limit},
        )

    def stored_freshness(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        """``updated_at`` and ``address`` of the stored Resources among ``node_ids``, by node id."""
        rows = self._db.query(
            """
            UNWIND $node_ids AS node_id
            MATCH (r:Resource {node_id: node_id})
            WHERE r.fetched_at IS NOT NULL
            RETURN node_id, r.updated_at AS updated_at, r.address AS address
            """,
            params={"node_ids": node_ids},
        )
        return {row["node_id"]: row for row in rows}

    def stored_node_id(self, address: Address) -> str | None:
        """The node id of the Resource stored under ``address``, if any."""
        rows = self._db.query(
            "MATCH (r:Resource {address: $address}) WHERE r.fetched_at IS NOT NULL RETURN r.node_id AS node_id",
            params={"address": address.key},
        )
        return rows[0]["node_id"] if rows else None

    def confirm_fresh(self, node_ids: list[str]) -> None:
        """GitHub reports these Resources unchanged since their fetch: their content is current as of now."""
        self._db.query(
            "UNWIND $node_ids AS node_id MATCH (r:Resource {node_id: node_id}) SET r.fetched_at = $now",
            params={"node_ids": node_ids, "now": now()},
        )

    def mark_moved(self, old_address: str, new_address: str) -> None:
        """The Resource touched as ``old_address`` now lives at ``new_address`` under a new node id.

        Releases the old Address and links old to new, so nothing it held is lost.
        Nothing happens when both name the same Resource (a renamed repository keeps its id).
        """
        self._db.query(
            """
            MATCH (old:Resource {address: $old}), (new:Resource {address: $new})
            WHERE old.node_id <> new.node_id
            SET old.address = null
            MERGE (old)-[:MOVED_TO]->(new)
            """,
            params={"old": old_address, "new": new_address},
        )

    def resolve_touch(self, touch_id: str, resource_address: str) -> None:
        """Point a pending Touch at the Resource or Listing stored under ``resource_address``.

        That is the Touch's own Address unless the Resource has moved since
        (a renamed repository, a transferred issue).
        """
        self._db.query(
            """
            MATCH (t:Touch {touch_id: $touch_id})
            OPTIONAL MATCH (r:Resource {address: $resource_address})
            OPTIONAL MATCH (l:Listing {key: $resource_address})
            WITH t, [x IN [r, l] WHERE x IS NOT NULL] AS targets
            WHERE size(targets) > 0
            SET t.status = $resolved, t.reason = null
            FOREACH (target IN targets | MERGE (t)-[:TOUCHED]->(target))
            """,
            params={"touch_id": touch_id, "resource_address": resource_address, "resolved": RESOLVED},
        )

    def unresolve_touch(self, touch_id: str, reason: str) -> None:
        """Mark a Touch the Sweep could not resolve, keeping why."""
        self._db.query(
            "MATCH (t:Touch {touch_id: $touch_id}) SET t.status = $unresolved, t.reason = $reason",
            params={"touch_id": touch_id, "unresolved": UNRESOLVED, "reason": reason},
        )

    def store_repository(self, repository: dict[str, Any], *, fetched_at: str | None = None) -> str:
        """Upsert a Repository Resource from a GitHub ``repository`` node. Returns its Address key."""
        owner, _, name = repository["nameWithOwner"].partition("/")
        address = Address("repo", owner, name).key
        self._release_address(address, repository["id"])
        self._db.query(
            """
            MERGE (r:Resource:Repository {node_id: $node_id})
            SET r.address = $address, r.name_with_owner = $name_with_owner, r.url = $url,
                r.description = $description, r.readme = $readme, r.stars = $stars,
                r.updated_at = $updated_at, r.pushed_at = $pushed_at, r.fetched_at = $fetched_at
            """,
            params={
                "node_id": repository["id"],
                "address": address,
                "name_with_owner": repository["nameWithOwner"],
                "url": repository["url"],
                "description": repository.get("description"),
                "readme": (repository.get("readme") or {}).get("text"),
                "stars": repository.get("stargazerCount"),
                "updated_at": repository.get("updatedAt"),
                "pushed_at": repository.get("pushedAt"),
                "fetched_at": fetched_at or now(),
            },
        )
        return address

    def store_item(self, item: dict[str, Any], *, fetched_at: str | None = None) -> str:
        """Upsert an Issue or PullRequest with all its Comments. Returns its Address key.

        Its Repository must already be stored (the Sweep stores it first), so no
        content-less Repository is ever created. Comments no longer on GitHub are removed.

        Raises:
            ValueError: the item's Repository isn't stored.
        """
        repository = item["repository"]
        owner, _, name = repository["nameWithOwner"].partition("/")
        address = Address("item", owner, name, item["number"]).key
        is_pull = item["__typename"] == "PullRequest"
        properties: dict[str, Any] = {
            "address": address,
            "number": item["number"],
            "url": item["url"],
            "title": item["title"],
            "body": item["body"],
            "state": item["state"],
            "author": (item.get("author") or {}).get("login"),
            "labels": [label["name"] for label in item["labels"]["nodes"]],
            "assignees": [assignee["login"] for assignee in item["assignees"]["nodes"]],
            "milestone": (item.get("milestone") or {}).get("title"),
            "created_at": item["createdAt"],
            "updated_at": item["updatedAt"],
            "closed_at": item.get("closedAt"),
            "comment_count": item["comments"]["totalCount"],
            "fetched_at": fetched_at or now(),
            # Kept even when the other end isn't stored, so the edge appears once it is.
            "referenced_by_ids": [source["id"] for source in _sources(item)],
            "referenced_by_urls": [source["url"] for source in _sources(item)],
        }
        comments = [_comment(comment, path=None) for comment in item["comments"]["nodes"]]
        if is_pull:
            properties |= {
                "merged": item["merged"],
                "base_ref": item["baseRefName"],
                "changed_files": [file["path"] for file in item["files"]["nodes"]],
                "closes_ids": [issue["id"] for issue in item["closingIssuesReferences"]["nodes"]],
                "closes_urls": [issue["url"] for issue in item["closingIssuesReferences"]["nodes"]],
            }
            comments += [
                _comment(comment, path=thread["path"])
                for thread in item["reviewThreads"]["nodes"]
                for comment in thread["comments"]["nodes"]
            ]
        self._release_address(address, item["id"])
        # Labels and the HAS_* type are fixed per Resource Kind, never caller input, so
        # interpolating them is safe; every value goes in as a parameter.
        label, relationship = ("PullRequest", "HAS_PULL_REQUEST") if is_pull else ("Issue", "HAS_ISSUE")
        stored = self._db.query(
            f"""
            MATCH (repo:Resource:Repository {{node_id: $repository_id}})
            MERGE (r:Resource:{label} {{node_id: $node_id}})
            SET r += $properties
            MERGE (repo)-[:{relationship}]->(r)
            WITH r
            OPTIONAL MATCH (r)-[:HAS_COMMENT]->(gone:Comment)
            WHERE NOT gone.node_id IN $comment_ids
            DETACH DELETE gone
            RETURN count(DISTINCT r) AS n
            """,
            params={
                "repository_id": repository["id"],
                "node_id": item["id"],
                "properties": properties,
                "comment_ids": [comment["node_id"] for comment in comments],
            },
        )
        if not stored or not stored[0]["n"]:
            raise ValueError(f"repository {repository['nameWithOwner']} must be stored before {address}")
        self._db.query(
            """
            MATCH (r:Resource {node_id: $node_id})
            UNWIND $comments AS comment
            MERGE (c:Comment {node_id: comment.node_id})
            SET c += comment
            MERGE (r)-[:HAS_COMMENT]->(c)
            """,
            params={"node_id": item["id"], "comments": comments},
        )
        self._link_references(item["id"])
        return address

    def _link_references(self, node_id: str) -> None:
        """Draw REFERENCES/CLOSES between this Resource and every stored one it names or is named by.

        Both directions, so the edge appears whichever end was stored first;
        nothing is drawn to a Resource that isn't stored.
        """
        for statement in (
            """
            MATCH (r:Resource {node_id: $node_id})
            UNWIND coalesce(r.referenced_by_ids, []) AS source_id
            MATCH (source:Resource {node_id: source_id})
            MERGE (source)-[:REFERENCES]->(r)
            """,
            """
            MATCH (r:Resource {node_id: $node_id}), (other:Resource)
            WHERE $node_id IN coalesce(other.referenced_by_ids, [])
            MERGE (r)-[:REFERENCES]->(other)
            """,
            """
            MATCH (pull:PullRequest {node_id: $node_id})
            UNWIND coalesce(pull.closes_ids, []) AS issue_id
            MATCH (issue:Resource {node_id: issue_id})
            MERGE (pull)-[:CLOSES]->(issue)
            """,
            """
            MATCH (r:Resource {node_id: $node_id}), (pull:PullRequest)
            WHERE $node_id IN coalesce(pull.closes_ids, [])
            MERGE (pull)-[:CLOSES]->(r)
            """,
        ):
            self._db.query(statement, params={"node_id": node_id})

    def link_touches(self) -> int:
        """Link Touches to the actions-graph ToolCall and Agent they came from, where those exist.

        A soft join on what both components record — the tool use id and the
        agent id within one Session — with no import of actions-graph. Touches
        whose Action isn't recorded (actions-graph not enabled) stay unlinked and
        are tried again next time. Returns how many links were drawn.
        """
        rows = self._db.query(
            """
            MATCH (t:Touch)-[:IN_SESSION]->(s:Session)
            WHERE t.tool_use_id IS NOT NULL AND NOT (t)-[:CAUSED_BY]->()
            OPTIONAL MATCH (s)-[:HAS_ACTION]->(direct:ToolCall {tool_use_id: t.tool_use_id})
            OPTIONAL MATCH (s)-[:HAS_AGENT]->(:Agent)-[:HAS_ACTION]->(nested:ToolCall {tool_use_id: t.tool_use_id})
            WITH t, coalesce(direct, nested) AS action
            WHERE action IS NOT NULL
            MERGE (t)-[:CAUSED_BY]->(action)
            RETURN count(t) AS n
            """
        )
        agents = self._db.query(
            """
            MATCH (t:Touch)-[:IN_SESSION]->(s:Session)-[:HAS_AGENT]->(agent:Agent)
            WHERE t.agent_name IS NOT NULL AND agent.agent_id = t.agent_name AND NOT (t)-[:BY_AGENT]->()
            MERGE (t)-[:BY_AGENT]->(agent)
            RETURN count(t) AS n
            """
        )
        return (rows[0]["n"] if rows else 0) + (agents[0]["n"] if agents else 0)

    def store_listing(self, address: Address, total_count: int, member_addresses: list[str]) -> str:
        """Upsert a Listing and replace its members with ``member_addresses``, in order. Returns its key.

        The members and the Repository must already be stored. A Listing is
        fully expanded when it holds every member GitHub counted.
        """
        self._db.query(
            """
            MATCH (repo:Resource:Repository {address: $repository})
            MERGE (l:Listing {key: $key})
            SET l.scope = $scope, l.item_kind = $item_kind, l.query = $query, l.limit = $limit,
                l.total_count = $total_count, l.member_count = size($members),
                l.fully_expanded = size($members) >= $total_count, l.fetched_at = $fetched_at
            MERGE (l)-[:LISTS_FROM]->(repo)
            WITH l
            OPTIONAL MATCH (l)-[old:HAS_MEMBER]->()
            DELETE old
            WITH DISTINCT l
            UNWIND range(0, size($members) - 1) AS position
            MATCH (r:Resource {address: $members[position]})
            MERGE (l)-[:HAS_MEMBER {position: position}]->(r)
            """,
            params={
                "repository": address.repository.key,
                "key": address.key,
                "scope": _scope(address),
                "item_kind": address.item_kind,
                "query": address.query,
                "limit": address.limit,
                "total_count": total_count,
                "members": member_addresses,
                "fetched_at": now(),
            },
        )
        return address.key

    def _release_address(self, address: str, node_id: str) -> None:
        """An Address names one Resource: after a rename or transfer, the old holder loses it."""
        self._db.query(
            "MATCH (r:Resource {address: $address}) WHERE r.node_id <> $node_id SET r.address = null",
            params={"address": address, "node_id": node_id},
        )


def _sources(item: dict[str, Any]) -> list[dict[str, Any]]:
    """The Issues and PRs that cross-referenced ``item`` (other event kinds and hidden sources dropped)."""
    nodes = (item.get("timelineItems") or {}).get("nodes") or []
    return [node["source"] for node in nodes if node and (node.get("source") or {}).get("id")]


def _scope(address: Address) -> str:
    """What a Listing could subsume within: one repository's issues, or its pulls."""
    return f"{address.repository.key}/{address.item_kind}"


def _comment(comment: dict[str, Any], *, path: str | None) -> dict[str, Any]:
    return {
        "node_id": comment["id"],
        "url": comment["url"],
        "body": comment["body"],
        "author": (comment.get("author") or {}).get("login"),
        "created_at": comment["createdAt"],
        "updated_at": comment["updatedAt"],
        "path": path,
        "review": path is not None,
    }
