"""A user's ontology versions: which graph model their sessions are extracted under.

Each version is one ``(:OntologyVersion {user_id, version})`` node holding
its model as JSON (#434). ``(:User)-[:ADOPTED]->`` points at the current
one and ``NEXT`` chains them, oldest first. A user with no version is on
version 0, ``hygm.default_model()``, which is never stored.

A version comes from one of two places:

- **supplied**: a schema the user gave (#436), from ``[ontology] path`` in
  the config file or ``sessions-graph ontology load``. Its types are
  *pinned*: derivation may add types beside them but never merge, rename or
  retire one. With ``derive = "extend"`` the fixed core is added to it;
  with ``"off"`` it is used exactly as given and nothing is derived.
- **derived**: a learned run (``sessions-graph derive``), adopted by the gate.
  A derived version also holds the run's observation counts, the pool of
  retired types, its changelog and the gate's report (#435). Candidates the
  gate rejects are kept as ``status: 'rejected'`` versions the user points at
  with ``REJECTED``, never ``ADOPTED``.

Requires hygm, which the ``sessions-graph[reconciliation]`` extra installs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from hygm import (
    HygmModel,
    ManualStrategy,
    default_model,
    model_from_mapping,
    model_to_mapping,
    validate_model,
    with_core,
)

if TYPE_CHECKING:
    from memgraph_toolbox.api.memgraph import Memgraph

Derive = Literal["extend", "off"]
DERIVE_MODES: tuple[Derive, ...] = ("extend", "off")
DEFAULT_DERIVE: Derive = "extend"
Source = Literal["default", "supplied", "derived"]


@dataclass(frozen=True)
class OntologyVersion:
    """One version of a user's model.

    Attributes:
        user_id: Whose model this is.
        version: 0 for the default, then 1, 2, … in the order they were created.
        model: The graph model sessions are extracted under.
        source: "default", "supplied" or "derived".
        derive: Whether derivation may extend this model ("extend") or not ("off").
        pinned: Labels derivation never merges, renames or retires: a supplied schema's types.
        source_hash: SHA-256 of the supplied schema file this version descends from, if any.
        created_at: ISO timestamp; None for the default.
        counts: Observation counts over every derivation run, ``{"nodes": {...}, "relations": {...}}``.
        pool: Types and relations derivation retired, kept so they can return.
        changelog: What the derivation run that made this version changed.
        report: The gate's numbers and the run's reporting signals.
    """

    user_id: str
    version: int
    model: HygmModel
    source: Source
    derive: Derive = DEFAULT_DERIVE
    pinned: tuple[str, ...] = ()
    source_hash: str | None = None
    created_at: str | None = None
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    pool: HygmModel = field(default_factory=lambda: HygmModel(node_types=()))
    changelog: tuple[dict[str, Any], ...] = ()
    report: dict[str, Any] = field(default_factory=dict)


def default_version(user_id: str) -> OntologyVersion:
    """Version 0: what a user with no stored version is extracted under."""
    return OntologyVersion(user_id=user_id, version=0, model=default_model(), source="default")


def adopted(db: Memgraph, user_id: str) -> OntologyVersion:
    """The version `user_id`'s sessions are extracted under now."""
    rows = db.query(
        "MATCH (:User {user_id: $user_id})-[:ADOPTED]->(v:OntologyVersion) RETURN properties(v) AS v",
        params={"user_id": user_id},
    )
    return _from_properties(rows[0]["v"]) if rows else default_version(user_id)


def supply(
    db: Memgraph, user_id: str, model: HygmModel, *, derive: Derive = DEFAULT_DERIVE, source_hash: str | None = None
) -> OntologyVersion:
    """Adopt a user-supplied schema as `user_id`'s next version.

    With ``derive="extend"`` the fixed core is added and types derivation
    learned on the current version carry forward, unless one clashes with a
    supplied label. With ``"off"`` the schema is used exactly as given. Either
    way it passes ``validate_model()`` only: the user chose it, so the outcome
    gate a derived version faces does not apply.

    Raises:
        ValueError: if `derive` is not a mode, or the resulting model fails validation.
    """
    if derive not in DERIVE_MODES:
        raise ValueError(f"derive must be one of {DERIVE_MODES}, got {derive!r}")
    current = adopted(db, user_id)
    pinned = model.node_labels() + model.relation_labels()
    if derive == "extend":
        model = _carry_learned(with_core(model), current)
    result = validate_model(model)
    if not result.success:
        raise ValueError("; ".join(issue.message for issue in result.critical_issues))
    return _adopt(
        db,
        OntologyVersion(
            user_id=user_id,
            version=_next_version(db, user_id),
            model=model,
            source="supplied",
            derive=derive,
            pinned=pinned,
            source_hash=source_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


def supply_file(db: Memgraph, user_id: str, path: str | Path, *, derive: Derive = DEFAULT_DERIVE) -> OntologyVersion:
    """Load the schema at `path` (``ManualStrategy`` YAML) and supply it; see :func:`supply`.

    Raises:
        ValueError: if the file can't be read or parsed, or the model fails validation.
    """
    model = ManualStrategy().create_model(path)
    return supply(db, user_id, model, derive=derive, source_hash=file_hash(path))


def sync_file(
    db: Memgraph, user_id: str, path: str | Path, *, derive: Derive = DEFAULT_DERIVE
) -> OntologyVersion | None:
    """Supply the configured schema file if it changed since the adopted version came from it.

    A version descends from the file when it records the same content hash
    and derive mode, so an unchanged file never makes a new version.

    Returns:
        The new version, or None when the file is unchanged.

    Raises:
        ValueError: as for :func:`supply_file`.
    """
    current = adopted(db, user_id)
    if current.source_hash == file_hash(path) and current.derive == derive:
        return None
    return supply_file(db, user_id, path, derive=derive)


def file_hash(path: str | Path) -> str:
    """SHA-256 of the file's bytes: what decides whether a configured schema changed.

    Raises:
        ValueError: if the file can't be read.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as e:
        raise ValueError(f"Could not read ontology file {path}: {e}") from e


def _carry_learned(model: HygmModel, current: OntologyVersion) -> HygmModel:
    """`model` plus the types derivation learned on `current`, minus any that clash with `model`'s labels.

    Learned means neither pinned on `current` nor part of the default model.
    A relation carries only when every endpoint it names exists afterwards.
    """
    default_labels = set(default_model().node_labels()) | set(default_model().relation_labels())
    fixed = set(current.pinned) | default_labels
    nodes = model.node_types + tuple(
        t for t in current.model.node_types if t.label not in fixed and t.label not in model.node_labels()
    )
    declared = {t.label for t in nodes}
    relations = model.relation_types + tuple(
        r
        for r in current.model.relation_types
        if r.label not in fixed
        and r.label not in model.relation_labels()
        and set(r.start_labels) | set(r.end_labels) <= declared
    )
    return HygmModel(node_types=nodes, relation_types=relations)


def _next_version(db: Memgraph, user_id: str) -> int:
    rows = db.query(
        "MATCH (v:OntologyVersion {user_id: $user_id}) RETURN max(v.version) AS latest",
        params={"user_id": user_id},
    )
    latest = rows[0]["latest"] if rows else None
    return (latest or 0) + 1


def adopt(db: Memgraph, version: OntologyVersion) -> OntologyVersion:
    """Store `version` and make it the user's adopted one."""
    return _adopt(db, version)


def reject(db: Memgraph, version: OntologyVersion) -> OntologyVersion:
    """Store `version` as a candidate the gate rejected: kept with its numbers, never adopted."""
    db.query(
        """
        MERGE (u:User {user_id: $user_id})
        CREATE (v:OntologyVersion {
            user_id: $user_id, version: $version, created_at: $created_at, status: 'rejected',
            source: $source, derive: $derive, model: $model, pinned: $pinned, source_hash: $source_hash,
            counts: $counts, pool: $pool, changelog: $changelog, report: $report
        })
        CREATE (u)-[:REJECTED]->(v)
        """,
        params=_params(version),
    )
    return version


def next_version(db: Memgraph, user_id: str) -> int:
    """The number the user's next stored version, adopted or rejected, takes."""
    return _next_version(db, user_id)


def _params(version: OntologyVersion) -> dict[str, Any]:
    return {
        "user_id": version.user_id,
        "version": version.version,
        "created_at": version.created_at,
        "source": version.source,
        "derive": version.derive,
        "model": json.dumps(model_to_mapping(version.model)),
        "pinned": list(version.pinned),
        "source_hash": version.source_hash,
        "counts": json.dumps(version.counts),
        "pool": json.dumps(model_to_mapping(version.pool)),
        "changelog": json.dumps(list(version.changelog)),
        "report": json.dumps(version.report),
    }


def _adopt(db: Memgraph, version: OntologyVersion) -> OntologyVersion:
    # One query is one transaction: readers see the old ADOPTED edge or the
    # new one, never a user with none or two.
    db.query(
        """
        MERGE (u:User {user_id: $user_id})
        WITH u
        OPTIONAL MATCH (u)-[adopted:ADOPTED]->(previous:OntologyVersion)
        CREATE (v:OntologyVersion {
            user_id: $user_id, version: $version, created_at: $created_at, status: 'adopted',
            source: $source, derive: $derive, model: $model, pinned: $pinned, source_hash: $source_hash,
            counts: $counts, pool: $pool, changelog: $changelog, report: $report
        })
        CREATE (u)-[:ADOPTED]->(v)
        FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END |
            SET previous.status = 'superseded'
            CREATE (previous)-[:NEXT]->(v)
            DELETE adopted
        )
        """,
        params=_params(version),
    )
    return version


def _from_properties(properties: dict[str, Any]) -> OntologyVersion:
    version = properties["version"]
    return OntologyVersion(
        user_id=properties["user_id"],
        version=version,
        model=model_from_mapping(json.loads(properties["model"]), f"version {version}"),
        source=properties["source"],
        derive=properties["derive"],
        pinned=tuple(properties.get("pinned") or ()),
        source_hash=properties.get("source_hash"),
        created_at=properties.get("created_at"),
        counts=json.loads(properties.get("counts") or "{}"),
        pool=model_from_mapping(json.loads(properties["pool"]), f"version {version} pool")
        if properties.get("pool")
        else HygmModel(node_types=()),
        changelog=tuple(json.loads(properties.get("changelog") or "[]")),
        report=json.loads(properties.get("report") or "{}"),
    )
