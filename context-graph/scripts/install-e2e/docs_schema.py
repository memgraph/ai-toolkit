"""Record and verify the docs' user-supplied-schema workflow against real Memgraph."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml
from neo4j import GraphDatabase

from memgraph_context_graph.events import MessageEvent
from memgraph_context_graph.sdk import ContextGraphMemory

SESSION = "docs-1847-custom-schema"
SCHEMA = Path.home() / "schemas/coding.yaml"
if "--verify" not in sys.argv:
    schema = yaml.safe_load(SCHEMA.read_text())
    schema["entity_types"].append(
        {"label": "SoftwareProject", "identity": "global", "description": "a named software project"}
    )
    SCHEMA.write_text(yaml.safe_dump(schema))
    memory = ContextGraphMemory(user_id="docs-schema-user", connectors=["sessions-graph", "actions-graph"])
    memory.start_session(SESSION, source_sdk="openai")
    memory.emit(
        MessageEvent(
            session_id=SESSION,
            source_sdk="openai",
            role="user",
            content="Alice works for Memgraph in Zagreb. Alice maintains the Orion software project. Bob reviews the release on Friday.",
        )
    )
    memory.end_session()
else:
    with GraphDatabase.driver("bolt://localhost:7687", auth=("", "")) as driver, driver.session() as session:
        rows = session.run(
            "MATCH (s:Session {session_id:$id}) RETURN s.reconciliation_status AS status", id=SESSION
        ).data()
        assert rows == [{"status": "completed"}], rows
        versions = session.run("MATCH (v:OntologyVersion) RETURN properties(v) AS props").data()
        assert versions, "User-supplied schema was not applied"
        adopted = next(row["props"] for row in versions if row["props"]["status"] == "adopted")
        assert adopted["source"] == "supplied", adopted
        assert adopted["source_hash"] == hashlib.sha256(SCHEMA.read_bytes()).hexdigest()
        assert "SoftwareProject" in [item["label"] for item in json.loads(adopted["model"])["entity_types"]]
        print("Custom-schema extraction completed; ontology versions:", len(versions))
