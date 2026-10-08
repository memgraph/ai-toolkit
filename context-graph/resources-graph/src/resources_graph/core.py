"""ResourcesGraph: Touches, shared Resources and Cache Reads in Memgraph.

Graph model::

    (:Session)<-[:IN_SESSION]-(:Touch)-[:TOUCHED]->(:Resource)
    (:Resource:Repository)-[:HAS_ISSUE|HAS_PULL_REQUEST]->(:Resource:Issue|PullRequest)
    (:Resource:Issue|PullRequest)-[:HAS_COMMENT]->(:Comment)

Resources and Comments belong to no user. A Touch is private to the user
whose Session it sits in; sessions-graph owns ``(:User)-[:HAD_SESSION]->``,
this component only MERGEs the shared ``(:Session {session_id})``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from memgraph_toolbox.api.memgraph import Memgraph

from .address import Address
from .models import FETCHED, HIT, MISS, Served

PENDING = "pending"
RESOLVED = "resolved"
UNRESOLVED = "unresolved"

_SCHEMA = (
    "CREATE CONSTRAINT ON (r:Resource) ASSERT r.node_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (c:Comment) ASSERT c.node_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (t:Touch) ASSERT t.touch_id IS UNIQUE;",
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
            ON CREATE SET t.address = $address, t.address_kind = $kind, t.provenance = $provenance,
                          t.served_from_memory = false, t.status = $pending, t.tool_use_id = $tool_use_id,
                          t.agent_name = $agent_name, t.at = $at
            MERGE (t)-[:IN_SESSION]->(s)
            """,
            params={
                "session_id": session_id,
                "touch_id": tid,
                "address": address.key,
                "kind": address.kind,
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
        tool_use_id: str | None = None,
        agent_name: str | None = None,
        at: str | None = None,
    ) -> str:
        """Record a Cache Read: a FETCHED Touch served from memory, with its outcome.

        It never becomes pending, so the Sweep never refreshes anything because of it.
        Returns the Touch id.
        """
        tid = touch_id(session_id, discriminator, "MEMORY", address)
        self._db.query(
            """
            MERGE (s:Session {session_id: $session_id})
            MERGE (t:Touch {touch_id: $touch_id})
            ON CREATE SET t.address = $address, t.address_kind = $kind, t.provenance = $provenance,
                          t.served_from_memory = true, t.outcome = $outcome, t.tool_use_id = $tool_use_id,
                          t.agent_name = $agent_name, t.at = $at
            MERGE (t)-[:IN_SESSION]->(s)
            WITH t
            OPTIONAL MATCH (r:Resource {address: $address})
            FOREACH (_ IN CASE WHEN r IS NULL OR $outcome <> $hit THEN [] ELSE [1] END |
                MERGE (t)-[:TOUCHED]->(r))
            """,
            params={
                "session_id": session_id,
                "touch_id": tid,
                "address": address.key,
                "kind": address.kind,
                "provenance": FETCHED,
                "outcome": outcome,
                "hit": HIT,
                "tool_use_id": tool_use_id,
                "agent_name": agent_name,
                "at": at or now(),
            },
        )
        return tid

    # -- read side: the resource tool ----------------------------------------

    def read(self, address: Address) -> Served:
        """Serve ``address`` from memory. Never calls GitHub and writes nothing."""
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
        """Touches waiting for the Sweep, oldest first."""
        return self._db.query(
            f"""
            MATCH (t:Touch {{status: $pending}})
            RETURN t.touch_id AS touch_id, t.address AS address, t.provenance AS provenance
            ORDER BY t.at
            {"LIMIT $limit" if limit else ""}
            """,
            params={"pending": PENDING, "limit": limit},
        )

    def resolve_touch(self, touch_id: str, resource_address: str) -> None:
        """Point a pending Touch at the Resource stored under ``resource_address``.

        That is the Touch's own Address unless the Resource has moved since
        (a renamed repository, a transferred issue).
        """
        self._db.query(
            """
            MATCH (t:Touch {touch_id: $touch_id})
            MATCH (r:Resource {address: $resource_address})
            SET t.status = $resolved, t.reason = null
            MERGE (t)-[:TOUCHED]->(r)
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
        }
        comments = [_comment(comment, path=None) for comment in item["comments"]["nodes"]]
        if is_pull:
            properties |= {
                "merged": item["merged"],
                "base_ref": item["baseRefName"],
                "changed_files": [file["path"] for file in item["files"]["nodes"]],
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
        return address

    def _release_address(self, address: str, node_id: str) -> None:
        """An Address names one Resource: after a rename or transfer, the old holder loses it."""
        self._db.query(
            "MATCH (r:Resource {address: $address}) WHERE r.node_id <> $node_id SET r.address = null",
            params={"address": address, "node_id": node_id},
        )


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
