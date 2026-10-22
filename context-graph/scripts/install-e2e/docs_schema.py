"""Record and verify the docs' user-supplied-schema workflow against real Memgraph."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import psutil
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
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        workers = [
            process
            for process in psutil.process_iter(["cmdline"])
            if any(
                Path(arg).name == "sessions-graph" or arg == "sessions_graph.cli"
                for arg in (process.info["cmdline"] or [])
            )
            and "embed" in (process.info["cmdline"] or [])
            and SESSION in (process.info["cmdline"] or [])
        ]
        if not workers:
            break
        print("Waiting for the fixture's background embedding job", flush=True)
        time.sleep(2)
    else:
        raise TimeoutError("The fixture's background embedding worker did not finish")
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
