"""Every Cypher query the viewer runs, apart from how the results are drawn.

The views are read-only: nothing here may create, merge, set or delete (a test enforces it). Queries never
return whole nodes or ``properties()``: Actions, Memory files and entities carry embedding vectors.

All queries take a :class:`Scope`, which narrows sessions to an owner and a project the way the dashboard does,
so every view agrees on what "in scope" means.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .source import Reader

USER = "user"
USER_AND_UNATTRIBUTED = "user+unattributed"
ALL = "all"
#: The scope cycle behind the ``a`` key, narrowest first.
SCOPES = (USER, USER_AND_UNATTRIBUTED, ALL)

#: How many of the most recently started sessions the dashboard expands into action counts per poll.
RECENT_SESSIONS = 200
#: Upper bound on actions loaded for one session's timeline.
MAX_ACTIONS = 2000
#: Upper bound on rows in the merged entity list.
MAX_ENTITIES = 1000

#: hygm.VALUE_LABELS: span-scoped values get one node per mention, so they swamp entity lists (#383).
VALUE_LABELS = ("Duration", "Quantity", "Money", "Date", "TimeWindow")

# Entities are whatever is MENTIONED_IN a Chunk (unstructured2graph's contract, for every backend), plus the
# (:User), which facts about the user point at but which is never mentioned. Keys fall back across backends:
# GLiNER2's entity_id is a hash, LightRAG's is the name, a User's is its user_id.
_KEY = "coalesce({v}.entity_id, {v}.name, {v}.id, {v}.user_id)"
_LABEL = (
    "CASE WHEN {v}:User THEN 'you (' + {v}.user_id + ')' ELSE coalesce({v}.text, {v}.name, {v}.entity_id, {v}.id) END"
)
_IS_ENTITY = "({v}:User OR exists(({v})-[:MENTIONED_IN]->(:Chunk)))"


def _key(v: str) -> str:
    return _KEY.format(v=v)


def _label(v: str) -> str:
    return _LABEL.format(v=v)


def _is_entity(v: str) -> str:
    return _IS_ENTITY.format(v=v)


# Reconcile synthesizes an `anon-<session_id>` user for a session captured without one, and a later
# SessionStart can link the real user too, so a session may have two owners: prefer the real one.
_SCOPED_SESSIONS = """
MATCH (s:Session)
OPTIONAL MATCH (u:User)-[:HAD_SESSION]->(s)
WITH s, collect(u.user_id) AS owners
WITH s, head([o IN owners WHERE NOT o STARTS WITH 'anon-'] + owners) AS owner
WITH s, owner, owner IS NULL OR owner STARTS WITH 'anon-' AS unattributed
WHERE ($scope = 'all' OR owner = $uid OR ($scope = 'user+unattributed' AND unattributed))
  AND ($project IS NULL OR s.working_directory = $project)
"""


@dataclass(frozen=True)
class Scope:
    """Which sessions a view shows.

    Attributes:
        mode: One of :data:`SCOPES`.
        user_id: The configured user; None when none is configured, which leaves only unattributed sessions.
        project: A ``Session.working_directory`` to narrow to, or None for every project.
    """

    mode: str = USER
    user_id: str | None = None
    project: str | None = None

    def params(self) -> dict[str, Any]:
        """The ``$scope``, ``$uid`` and ``$project`` parameters of the scoped-sessions fragment."""
        return {"scope": self.mode, "uid": self.user_id, "project": self.project}

    def memory_owner(self) -> str | None:
        """The user whose memory files are in scope, or None for every user's (scope ``all``)."""
        return None if self.mode == ALL else self.user_id

    def includes_memory_files(self) -> bool:
        """Memory files belong to a user, not a project, so a project filter leaves them out."""
        return self.project is None

    def widened(self) -> Scope:
        """The next scope in the ``a`` cycle."""
        return replace(self, mode=SCOPES[(SCOPES.index(self.mode) + 1) % len(SCOPES)])


# ---------------------------------------------------------------- dashboard


@dataclass
class Dashboard:
    """Everything the landing screen draws, from one poll."""

    #: The most recently started sessions in scope, newest activity first, with action statistics.
    sessions: list[dict[str, Any]]
    #: In-scope session count by (reconciliation_status, embedding_status, extraction_backend).
    pipeline: list[dict[str, Any]]
    #: In-scope sessions whose reconcile, re-extraction or embedding failed, newest first.
    failures: list[dict[str, Any]]
    #: Sessions in the graph (within the project filter) that no real user owns.
    unattributed: int
    memories: list[dict[str, Any]]
    #: Resource Touch count by status across in-scope sessions.
    touches: dict[str, int]
    ontology: list[dict[str, Any]]
    #: New entities per day, by the day their first supporting session was reconciled.
    growth: dict[str, int]
    #: Every working directory seen under the current owner scope, for the project picker.
    projects: list[str]
    total_sessions: int = 0
    tokens: int = 0


def dashboard(reader: Reader, scope: Scope) -> Dashboard:
    """Load the landing screen's data for *scope*."""
    params = scope.params()
    sessions = reader.rows(
        _SCOPED_SESSIONS
        + """
        WITH s, owner, unattributed ORDER BY s.started_at DESC LIMIT $recent
        OPTIONAL MATCH (s)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:Action)
        WITH s, owner, unattributed,
             count(CASE WHEN a.action_type <> 'tool_result' THEN 1 END) AS actions,
             max(a.timestamp) AS last_event,
             count(CASE WHEN a.is_error = true THEN 1 END) AS errors
        OPTIONAL MATCH (s)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk)
        WITH s, owner, unattributed, actions, last_event, errors, count(DISTINCT c) AS chunks
        OPTIONAL MATCH (t:Touch)-[:IN_SESSION]->(s)
        RETURN s.session_id AS id, owner, unattributed, s.working_directory AS wd, s.status AS status,
               s.started_at AS started, s.ended_at AS ended, s.reconciliation_status AS rstatus,
               s.reconciled_at AS reconciled_at, s.extraction_backend AS backend,
               s.embedding_status AS estatus,
               coalesce(s.total_input_tokens, 0) + coalesce(s.total_output_tokens, 0) AS tokens,
               actions, last_event, errors, chunks, count(t) AS touches,
               count(CASE WHEN t.status IN ['pending', 'unresolved'] THEN 1 END) AS touches_open
        ORDER BY coalesce(last_event, started) DESC
        """,
        **params,
        recent=RECENT_SESSIONS,
    )
    pipeline = reader.rows(
        _SCOPED_SESSIONS
        + """
        RETURN s.reconciliation_status AS rstatus, s.embedding_status AS estatus,
               s.extraction_backend AS backend, count(s) AS n,
               sum(coalesce(s.total_input_tokens, 0) + coalesce(s.total_output_tokens, 0)) AS tokens
        """,
        **params,
    )
    failures = reader.rows(
        _SCOPED_SESSIONS
        + """
        WITH s WHERE s.reconciliation_status = 'failed' OR s.reextraction_error IS NOT NULL
                  OR s.embedding_status = 'failed'
        RETURN s.session_id AS id, s.reconciliation_status AS rstatus, s.reconciliation_error AS rerror,
               s.reextraction_error AS reerror, s.embedding_status AS estatus, s.embedding_error AS eerror
        ORDER BY s.started_at DESC LIMIT 20
        """,
        **params,
    )
    unattributed = reader.rows(
        """
        MATCH (s:Session) WHERE $project IS NULL OR s.working_directory = $project
        OPTIONAL MATCH (u:User)-[:HAD_SESSION]->(s)
        WITH s, collect(u.user_id) AS owners
        WHERE all(o IN owners WHERE o STARTS WITH 'anon-')
        RETURN count(s) AS n
        """,
        project=scope.project,
    )[0]["n"]
    memories = (
        reader.rows(
            """
            MATCH (u:User)-[:HAS_MEMORY]->(m:Memory)
            WHERE $uid IS NULL OR u.user_id = $uid
            RETURN m.path AS path, m.extraction_status AS status, m.extraction_error AS error
            ORDER BY path
            """,
            uid=scope.memory_owner(),
        )
        if scope.includes_memory_files() and (scope.mode == ALL or scope.user_id)
        else []
    )
    touches = reader.rows(
        _SCOPED_SESSIONS
        + """
        MATCH (t:Touch)-[:IN_SESSION]->(s)
        RETURN coalesce(t.status, 'unknown') AS status, count(t) AS n
        """,
        **params,
    )
    ontology = reader.rows(
        """
        MATCH (u:User) WHERE NOT u.user_id STARTS WITH 'anon-' AND ($uid IS NULL OR u.user_id = $uid)
        OPTIONAL MATCH (u)-[:ADOPTED]->(v:OntologyVersion)
        RETURN u.user_id AS user, max(v.version) AS version, u.derive_milestone AS milestone
        ORDER BY user
        """,
        uid=scope.memory_owner(),
    )
    growth = reader.rows(
        _SCOPED_SESSIONS
        + """
        WITH s WHERE s.reconciled_at IS NOT NULL
        MATCH (s)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(e)
        WITH e, min(toString(s.reconciled_at)) AS first
        RETURN substring(first, 0, 10) AS day, count(e) AS n
        """,
        **params,
    )
    projects = reader.rows(
        _SCOPED_SESSIONS + "WITH DISTINCT s.working_directory AS wd WHERE wd IS NOT NULL RETURN wd ORDER BY wd",
        **{**params, "project": None},
    )
    return Dashboard(
        sessions=sessions,
        pipeline=pipeline,
        failures=failures,
        unattributed=unattributed,
        memories=memories,
        touches={row["status"]: row["n"] for row in touches},
        ontology=ontology,
        growth={row["day"]: row["n"] for row in growth},
        projects=[row["wd"] for row in projects],
        total_sessions=sum(row["n"] for row in pipeline),
        tokens=sum(row["tokens"] or 0 for row in pipeline),
    )


def scoped_session_ids(reader: Reader, scope: Scope) -> list[str]:
    """Every in-scope session id."""
    return [row["id"] for row in reader.rows(_SCOPED_SESSIONS + "RETURN s.session_id AS id", **scope.params())]


# ---------------------------------------------------------------- session anatomy


@dataclass
class SessionAnatomy:
    """One session: its timeline, its subagents, what it touched and what it produced."""

    session: dict[str, Any]
    #: Tool calls and messages in time order; tool results are folded into their call (``result_at``).
    actions: list[dict[str, Any]]
    #: Subagents, with the tool call that spawned each one when actions-graph could infer it.
    agents: list[dict[str, Any]]
    #: Resource Touches keyed by the action that caused them.
    touches: dict[str, list[dict[str, Any]]]
    #: Touches no action could be matched to.
    loose_touches: list[dict[str, Any]]
    produced: dict[str, Any] = field(default_factory=dict)


def session_anatomy(reader: Reader, session_id: str) -> SessionAnatomy | None:
    """Load one session, or None when the graph has no such session."""
    head = reader.rows(
        """
        MATCH (s:Session {session_id: $id})
        OPTIONAL MATCH (u:User)-[:HAD_SESSION]->(s)
        WITH s, collect(u.user_id) AS owners
        RETURN s.session_id AS id, head([o IN owners WHERE NOT o STARTS WITH 'anon-'] + owners) AS owner,
               s.working_directory AS wd, s.status AS status, s.started_at AS started, s.ended_at AS ended,
               s.model AS model
        """,
        id=session_id,
    )
    if not head:
        return None
    actions = reader.rows(
        """
        MATCH (s:Session {session_id: $id})
        CALL {
          WITH s MATCH (s)-[:HAS_ACTION]->(a:Action) RETURN a, null AS agent_id
          UNION
          WITH s MATCH (s)-[:HAS_AGENT]->(g:Agent)-[:HAS_ACTION]->(a:Action) RETURN a, g.agent_id AS agent_id
        }
        WITH a, agent_id WHERE a.action_type <> 'tool_result'
        OPTIONAL MATCH (a)-[:PARENT_OF]->(r:Action)
        WITH a, agent_id, collect(r)[0] AS r
        RETURN a.action_id AS id, a.action_type AS type, a.tool_name AS tool, a.tool_use_id AS tool_use_id,
               a.timestamp AS at, a.status AS status, a.is_error AS is_error, agent_id,
               CASE WHEN a.action_type = 'tool_call' THEN a.properties END AS props,
               substring(coalesce(a.text, ''), 0, 400) AS text,
               r.timestamp AS result_at, coalesce(r.is_error = true, false) AS result_error
        ORDER BY at
        LIMIT $limit
        """,
        id=session_id,
        limit=MAX_ACTIONS,
    )
    agents = reader.rows(
        """
        MATCH (:Session {session_id: $id})-[:HAS_AGENT]->(g:Agent)
        OPTIONAL MATCH (p:Action)-[:SPAWNED]->(g)
        RETURN g.agent_id AS id, g.agent_type AS type, g.description AS description, g.status AS status,
               g.started_at AS started, g.ended_at AS ended, collect(p.action_id)[0] AS spawned_by
        ORDER BY started
        """,
        id=session_id,
    )
    touches = reader.rows(
        """
        MATCH (t:Touch)-[:IN_SESSION]->(:Session {session_id: $id})
        OPTIONAL MATCH (t)-[:CAUSED_BY]->(c:Action)
        RETURN t.touch_id AS id, collect(c.action_id)[0] AS action_id, t.tool_use_id AS tool_use_id,
               t.status AS status, t.served_from_memory AS from_memory, t.address AS address,
               t.address_kind AS kind, t.provenance AS provenance, t.at AS at
        ORDER BY at
        """,
        id=session_id,
    )
    produced = reader.rows(
        f"""
        MATCH (s:Session {{session_id: $id}})
        OPTIONAL MATCH (s)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(e)
        WITH s, count(DISTINCT c) AS chunks, count(DISTINCT e) AS entity_count,
             collect(DISTINCT {_label("e")})[..12] AS entities
        OPTIONAL MATCH (s)-[:PRODUCED_MEMORY]->(m:Memory)
        RETURN s.reconciliation_status AS rstatus, s.reconciliation_error AS rerror,
               s.reconciled_at AS reconciled_at, s.extraction_backend AS backend,
               s.ontology_version AS ontology_version, s.embedding_status AS estatus,
               s.embedding_error AS eerror, s.reextraction_error AS reerror,
               chunks, entity_count, entities, collect(DISTINCT m.path) AS memories
        """,
        id=session_id,
    )
    # A Touch the resolver hasn't reached yet has no CAUSED_BY edge; the harness's tool-use id still joins it.
    by_tool_use = {a["tool_use_id"]: a["id"] for a in actions if a["tool_use_id"]}
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loose: list[dict[str, Any]] = []
    for touch in touches:
        action_id = touch["action_id"] or by_tool_use.get(touch["tool_use_id"])
        if action_id:
            by_action[action_id].append(touch)
        else:
            loose.append(touch)
    return SessionAnatomy(
        session=head[0],
        actions=actions,
        agents=agents,
        touches=dict(by_action),
        loose_touches=loose,
        produced=produced[0] if produced else {},
    )


# ---------------------------------------------------------------- knowledge graph


@dataclass(frozen=True)
class Focus:
    """The entity a knowledge screen is about: one display label, possibly several graph nodes.

    GLiNER2 gives one name several nodes (its ids are hashes), so the viewer shows them as one entity.
    """

    keys: tuple[str, ...]
    label: str


def entities(
    reader: Reader, scope: Scope, *, session_id: str | None = None, hide_values: bool = True
) -> list[dict[str, Any]]:
    """In-scope entities merged by label, most supported first.

    Args:
        session_id: Only entities this session produced (the ``g`` entry point); memory files are left out.
        hide_values: Leave out span-scoped value types (dates, durations, amounts).

    Returns:
        One row per label: ``keys`` (graph keys merged into it), ``label``, ``types``, ``support`` (distinct
        in-scope sessions and memory files grounding it), ``degree`` (relations to other entities, summed across
        merged nodes), ``mentions`` and ``last_seen`` (ISO string of the latest supporting reconcile).
    """
    ids = [session_id] if session_id else scoped_session_ids(reader, scope)
    with_memories = session_id is None and scope.includes_memory_files() and (scope.mode == ALL or bool(scope.user_id))
    rows = reader.rows(
        f"""
        CALL {{
          MATCH (s:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(e)
          WHERE s.session_id IN $ids
          RETURN e, s.session_id AS source, toString(s.reconciled_at) AS seen
          UNION
          MATCH (u:User)-[:HAS_MEMORY]->(m:Memory)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(e)
          WHERE $memories AND ($uid IS NULL OR u.user_id = $uid)
          RETURN e, 'memory:' + m.path AS source, toString(m.updated_at) AS seen
        }}
        WITH e, collect(DISTINCT source) AS sources, max(seen) AS last_seen
        WHERE NOT ($hide AND any(l IN labels(e) WHERE l IN $values))
        OPTIONAL MATCH (e)-[r]-(f) WHERE type(r) <> 'MENTIONED_IN' AND {_is_entity("f")}
        WITH e, sources, last_seen, count(DISTINCT f) AS degree
        OPTIONAL MATCH (e)-[mi:MENTIONED_IN]->()
        RETURN {_key("e")} AS key, {_label("e")} AS label, e.entity_type AS type, sources, last_seen, degree,
               count(mi) AS mentions
        """,
        ids=ids,
        memories=with_memories,
        uid=scope.memory_owner(),
        hide=hide_values,
        values=list(VALUE_LABELS),
    )
    return merge_by_label(rows)[:MAX_ENTITIES]


def merge_by_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold entity rows that share a label (case-insensitively) into one row, most supported first."""
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row["label"] if row["label"] is not None else row["key"])
        entry = merged.setdefault(
            label.casefold(),
            {"keys": [], "label": label, "types": [], "sources": set(), "degree": 0, "mentions": 0, "last_seen": None},
        )
        entry["keys"].append(row["key"])
        if row["type"] and row["type"] not in entry["types"]:
            entry["types"].append(row["type"])
        entry["sources"].update(row["sources"] or [])
        entry["degree"] += row["degree"] or 0
        entry["mentions"] += row["mentions"] or 0
        if row["last_seen"] and (entry["last_seen"] is None or row["last_seen"] > entry["last_seen"]):
            entry["last_seen"] = row["last_seen"]
    result = []
    for entry in merged.values():
        sources = entry.pop("sources")
        result.append({**entry, "support": len(sources)})
    result.sort(key=lambda e: (-e["support"], -e["degree"], str(e["label"]).casefold()))
    return result


def entity_keys(reader: Reader, scope: Scope) -> list[str] | None:
    """Keys of every entity grounded in scope, or None when nothing narrows it (scope ``all``, every project)."""
    if scope.mode == ALL and scope.project is None:
        return None
    return [row["key"] for row in _entity_key_rows(reader, scope)]


def _entity_key_rows(reader: Reader, scope: Scope) -> list[dict[str, Any]]:
    """Distinct keys of the entities in-scope sessions and memory files mention, unmerged."""
    return reader.rows(
        f"""
        CALL {{
          MATCH (s:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(e)
          WHERE s.session_id IN $ids
          RETURN e
          UNION
          MATCH (u:User)-[:HAS_MEMORY]->(:Memory)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(e)
          WHERE $memories AND ($uid IS NULL OR u.user_id = $uid)
          RETURN e
        }}
        RETURN DISTINCT {_key("e")} AS key
        """,
        ids=scoped_session_ids(reader, scope),
        memories=scope.includes_memory_files() and (scope.mode == ALL or scope.user_id is not None),
        uid=scope.memory_owner(),
    )


def neighbours(reader: Reader, focus: Focus, scope: Scope, allowed: list[str] | None) -> list[dict[str, Any]]:
    """The focused entity's relations, one row per (direction, relation type, neighbour label).

    GLiNER2 keys relations per chunk and source turn, so one fact is often several parallel edges; they fold into
    one row that keeps the most confident edge's evidence.

    Args:
        allowed: Keys of in-scope entities (from :func:`entity_keys`); neighbours outside it are left out.
            None allows every entity.

    Returns:
        Rows with ``outgoing``, ``rel_type``, ``keys`` and ``label`` of the neighbour, ``type``, ``confidence``
        (GLiNER2), ``weight`` and ``keywords`` (LightRAG), ``evidence`` (relation text or description),
        ``created``, ``role``, ``edges`` (parallel edges folded in) and ``nonconformant`` (outside the ontology).
    """
    return reader.rows(
        f"""
        MATCH (e)-[r]-(f)
        WHERE {_key("e")} IN $keys AND type(r) <> 'MENTIONED_IN' AND NOT {_key("f")} IN $keys
          AND {_is_entity("f")}
          AND ((f:User AND ($uid IS NULL OR f.user_id = $uid))
               OR (NOT f:User AND ($allowed IS NULL OR {_key("f")} IN $allowed)))
        WITH startNode(r) = e AS outgoing, type(r) AS rel_type, f, r
        ORDER BY coalesce(r.confidence, r.weight, 0) DESC
        WITH outgoing, rel_type, toLower({_label("f")}) AS fold, collect(r) AS rs, collect(f) AS fs
        WITH outgoing, rel_type, rs, fs, rs[0] AS best, fs[0] AS f
        RETURN outgoing, rel_type,
               [x IN fs | {_key("x")}] AS keys, {_label("f")} AS label,
               coalesce(f.entity_type, CASE WHEN f:User THEN 'User' END) AS type,
               best.confidence AS confidence, best.weight AS weight, best.keywords AS keywords,
               coalesce(best.text, best.description) AS evidence,
               toString(coalesce(best.valid_at, best.created_at)) AS created, best.role AS role,
               size(rs) AS edges,
               any(x IN rs WHERE toString(x.ontology_conformant) = 'false') AS nonconformant
        """,
        keys=list(focus.keys),
        uid=scope.memory_owner(),
        allowed=allowed,
    )


def provenance(reader: Reader, focus: Focus, scope: Scope) -> dict[str, Any]:
    """Where the focused entity comes from: its description, in-scope sessions and memory files.

    Provenance is session-level (#381): a session with several Chunks can't say which turn mentioned it.
    """
    head = reader.rows(
        f"""
        MATCH (e) WHERE {_key("e")} IN $keys
        RETURN coalesce(e.description, e.text) AS description, e.entity_type AS type
        LIMIT 1
        """,
        keys=list(focus.keys),
    )
    sessions = reader.rows(
        f"""
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)<-[:HAS_CHUNK]-(:Action)<-[:HAS_ACTION]-(s:Session)
        WHERE {_key("e")} IN $keys AND s.session_id IN $ids
        RETURN s.session_id AS session, s.reconciled_at AS reconciled_at, s.working_directory AS wd,
               count(DISTINCT c) AS chunks
        ORDER BY reconciled_at DESC
        """,
        keys=list(focus.keys),
        ids=scoped_session_ids(reader, scope),
    )
    memories = (
        reader.rows(
            f"""
            MATCH (e)-[:MENTIONED_IN]->(:Chunk)<-[:HAS_CHUNK]-(m:Memory)<-[:HAS_MEMORY]-(u:User)
            WHERE {_key("e")} IN $keys AND ($uid IS NULL OR u.user_id = $uid)
            RETURN DISTINCT m.path AS path ORDER BY path
            """,
            keys=list(focus.keys),
            uid=scope.memory_owner(),
        )
        if scope.includes_memory_files()
        else []
    )
    return {
        **(head[0] if head else {"description": None, "type": None}),
        "sessions": sessions,
        "memories": [row["path"] for row in memories],
    }
