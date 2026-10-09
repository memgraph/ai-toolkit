"""End to end on a real Memgraph: hook events -> Touches -> Sweep -> ``resource`` -> Cache Read.

GitHub is replayed from recorded responses (``fixtures/github.json``); everything else is real,
including sessions-graph's connector, which owns ``(:User)-[:HAD_SESSION]->(:Session)``.
"""

from __future__ import annotations

import os

import pytest

from agent_context_graph.adapters._identity import HookConfig
from agent_context_graph.events import MessageEvent, SessionStartEvent, ToolEndEvent, ToolStartEvent
from agent_context_graph.tools import ToolError
from resources_graph.address import parse_address
from resources_graph.sweep import sweep
from resources_graph.tool import ResourceTool

MCP_RESOURCE = "mcp__plugin_context-graph_context-graph__resource"


def config(user_id):
    """The config file the MCP server would read, pointed at the test Memgraph."""
    return HookConfig(
        user_id=user_id,
        memgraph_url=os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687"),
        memgraph_user=os.environ.get("MEMGRAPH_USER", ""),
        memgraph_password=os.environ.get("MEMGRAPH_PASSWORD", ""),
    )


def start(emit, session_id, user_id):
    emit(SessionStartEvent(session_id=session_id, user_id=user_id))


def fetch(emit, session_id, command, tool_use_id):
    emit(
        ToolStartEvent(
            session_id=session_id, tool_name="Bash", tool_input={"command": command}, tool_use_id=tool_use_id
        )
    )


def ask_memory(emit, session_id, address, tool_use_id, user_id="ante"):
    """The model calls ``resource``; the harness reports the call to the hooks like any other tool."""
    result = ResourceTool().call({"address": address}, config(user_id))
    emit(
        ToolStartEvent(
            session_id=session_id, tool_name=MCP_RESOURCE, tool_input={"address": address}, tool_use_id=tool_use_id
        )
    )
    emit(
        ToolEndEvent(
            session_id=session_id,
            tool_name=MCP_RESOURCE,
            tool_use_id=tool_use_id,
            result=[{"type": "text", "text": result.text}],
        )
    )
    return result


def touches(graph, user_id):
    return sorted(
        (
            row["touch"]["address"],
            row["touch"]["provenance"],
            row["touch"]["served_from_memory"],
            row["touch"].get("outcome"),
            row["touch"].get("status"),
        )
        for row in graph.touches(user_id)
    )


def test_read_twice_second_time_from_memory(graph, harness, source):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph --comments", "toolu_1")

    first = ask_memory(harness, "s1", "memgraph/memgraph#2000", "toolu_2")
    assert first.text.startswith("[resource miss] github:memgraph/memgraph#2000")

    report = sweep(graph, source)
    assert (report.resolved, report.unresolved) == (1, {})

    second = ask_memory(harness, "s1", "https://github.com/memgraph/memgraph/issues/2000", "toolu_3")
    assert second.text.startswith("[resource hit] github:memgraph/memgraph#2000")
    assert second.structured["kind"] == "Issue"
    assert len(second.structured["comments"]) == 4
    assert second.structured["resource"]["fetched_at"] and second.structured["resource"]["updated_at"]
    assert "fetched_at:" in second.text and "updated_at (GitHub):" in second.text

    assert touches(graph, "ante") == [
        ("github:memgraph/memgraph#2000", "FETCHED", False, None, "resolved"),
        ("github:memgraph/memgraph#2000", "FETCHED", True, "hit", None),
        ("github:memgraph/memgraph#2000", "FETCHED", True, "miss", None),
    ]
    hit_edges = graph._db.query("MATCH (t:Touch {outcome: 'hit'})-[:TOUCHED]->(r:Issue) RETURN r.address AS address")
    assert [row["address"] for row in hit_edges] == ["github:memgraph/memgraph#2000"]


def test_cache_reads_never_make_the_sweep_fetch(graph, harness, source, replay):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    sweep(graph, source)
    calls = replay.calls

    ask_memory(harness, "s1", "memgraph/memgraph#2000", "toolu_2")
    report = sweep(graph, source)

    assert (report.resolved, report.fetched, replay.calls) == (0, 0, calls)


def test_issues_url_naming_a_pull_request_stores_a_pull_request(graph, harness, source):
    start(harness, "s1", "ante")
    harness(
        ToolStartEvent(
            session_id="s1",
            tool_name="WebFetch",
            tool_input={"url": "https://github.com/memgraph/memgraph/issues/4962"},
            tool_use_id="t1",
        )
    )
    sweep(graph, source)

    served = graph.read(address("memgraph/memgraph#4962"))
    assert served.kind == "PullRequest"
    assert served.resource["changed_files"]
    review = [comment for comment in served.comments if comment["review"]]
    assert review and all(comment["path"] for comment in review)
    assert "diff" not in served.resource
    rows = graph._db.query(
        "MATCH (:Repository {address: 'github:memgraph/memgraph'})-[:HAS_PULL_REQUEST]->(p:PullRequest) "
        "RETURN count(p) AS n"
    )
    assert rows[0]["n"] == 1


def test_resources_are_shared_but_touches_stay_private(graph, harness, source):
    start(harness, "s1", "ante")
    start(harness, "s2", "marko")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    sweep(graph, source)

    marko = ask_memory(harness, "s2", "memgraph/memgraph#2000", "toolu_9", user_id="marko")

    assert marko.structured["outcome"] == "hit"
    assert [touch[1:] for touch in touches(graph, "marko")] == [("FETCHED", True, "hit", None)]
    assert [touch[1:] for touch in touches(graph, "ante")] == [("FETCHED", False, None, "resolved")]
    assert graph.touches(None) == []
    with pytest.raises(ToolError):
        ResourceTool().call({"address": "memgraph/memgraph#2000"}, config(None))


def test_prompted_link_of_a_stored_resource_resolves_without_fetching(graph, harness, source, replay):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    sweep(graph, source)
    calls = replay.calls

    harness(
        MessageEvent(
            session_id="s1", role="user", content="what did https://github.com/memgraph/memgraph/issues/2000 conclude?"
        )
    )
    report = sweep(graph, source)

    assert (report.resolved, report.fetched, replay.calls) == (1, 0, calls)
    assert ("github:memgraph/memgraph#2000", "PROMPTED", False, None, "resolved") in touches(graph, "ante")


def test_a_real_refetch_refreshes_and_drops_deleted_comments(graph, harness, source, replay):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    sweep(graph, source)
    assert len(graph.read(address("memgraph/memgraph#2000")).comments) == 4

    # GitHub now has one comment fewer (it was deleted) and a newer updatedAt.
    def delete_last_comment(body):
        node = body["data"]["node"]
        node["comments"]["nodes"] = node["comments"]["nodes"][:-1]

    replay.edit("MoreComments:", delete_last_comment)
    for operation in ("Item:", "Freshness:"):  # the cheap check sees the change first
        replay.edit(
            operation,
            lambda body: (body["data"]["repository"]["issueOrPullRequest"] or {}).update(
                updatedAt="2030-01-01T00:00:00Z"
            ),
        )
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_2")
    report = sweep(graph, source)

    served = graph.read(address("memgraph/memgraph#2000"))
    assert (report.checked, report.fetched) == (1, 1)  # one cheap check, then one refetch; the Repository is stored
    assert served.resource["updated_at"] == "2030-01-01T00:00:00Z"
    assert len(served.comments) == 3
    assert graph._db.query("MATCH (c:Comment) RETURN count(c) AS n")[0]["n"] == 3


def test_unresolvable_touches_keep_their_reason_and_create_no_resource(graph, harness, source, replay):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh issue view 999999 -R memgraph/memgraph", "toolu_1")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_2")
    replay.edit(
        "Item:",
        lambda body: (
            (body["data"]["repository"]["issueOrPullRequest"] or {}).get("repository", {}).update(visibility="PRIVATE")
        ),
    )

    report = sweep(graph, source)

    assert report.unresolved == {"not_found": 1, "not_public": 1}
    statuses = {(touch[0], touch[4]) for touch in touches(graph, "ante")}
    assert statuses == {
        ("github:memgraph/memgraph#999999", "unresolved"),
        ("github:memgraph/memgraph#2000", "unresolved"),
    }
    reasons = graph._db.query("MATCH (t:Touch) RETURN t.address AS address, t.reason AS reason ORDER BY address")
    assert [row["reason"] for row in reasons] == ["not_public", "not_found"]
    assert graph._db.query("MATCH (r:Resource) RETURN count(r) AS n")[0]["n"] == 0


def test_redelivered_hook_events_write_one_touch(graph, harness):
    start(harness, "s1", "ante")
    for _ in range(3):
        fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    assert len(graph.touches("ante")) == 1


def test_one_fetch_per_address_per_sweep(graph, harness, source, replay):
    start(harness, "s1", "ante")
    start(harness, "s2", "marko")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    fetch(harness, "s2", "gh issue view 2000 -R memgraph/memgraph", "toolu_2")

    report = sweep(graph, source)

    assert (report.resolved, report.fetched) == (2, 2)  # the issue and its repository, once each


def test_repository_touch_stores_readme(graph, harness, source):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh repo view memgraph/memgraph", "toolu_1")
    sweep(graph, source)

    served = ResourceTool().call({"address": "https://github.com/memgraph/memgraph"}, config("ante"))
    assert served.structured["kind"] == "Repository"
    assert served.structured["resource"]["readme"]


def address(text):
    parsed = parse_address(text)
    assert parsed is not None
    return parsed


def test_a_refetch_of_unchanged_content_is_one_cheap_check(graph, harness, source, replay):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    sweep(graph, source)
    fetched_at = graph.read(address("memgraph/memgraph#2000")).resource["fetched_at"]

    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_2")
    report = sweep(graph, source)

    assert (report.resolved, report.checked, report.unchanged, report.fetched) == (1, 1, 1, 0)
    assert graph.read(address("memgraph/memgraph#2000")).resource["fetched_at"] >= fetched_at


def test_rate_limit_stops_the_sweep_and_the_next_one_retries(graph, harness, source, replay):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    fetch(harness, "s1", "gh pr view 3600 -R memgraph/memgraph", "toolu_2")
    recorded = dict(replay.responses)
    for key in list(replay.responses):
        replay.responses[key] = {"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]}

    limited = sweep(graph, source)

    assert (limited.resolved, limited.unresolved) == (0, {"rate_limited": 1})  # stopped at the first
    statuses = sorted(touch[4] for touch in touches(graph, "ante"))
    assert statuses == ["pending", "unresolved"]

    replay.responses = recorded
    retried = sweep(graph, source)

    assert (retried.resolved, retried.unresolved) == (2, {})


def test_a_transferred_issue_links_old_to_new(graph, harness, source, replay):
    start(harness, "s1", "ante")
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_1")
    fetch(harness, "s1", "gh repo view memgraph/gqlalchemy", "toolu_2")
    sweep(graph, source)

    def transfer(body):  # GitHub now answers memgraph/memgraph#2000 with the issue moved to gqlalchemy
        item = body["data"]["repository"]["issueOrPullRequest"]
        if item and item["number"] == 2000:
            item.update(id="I_transferred", number=901, updatedAt="2030-01-01T00:00:00Z")
            for connection in ("comments", "timelineItems"):  # its pages were recorded under the old id
                if connection in item:
                    item[connection]["pageInfo"]["hasNextPage"] = False
            item["repository"] = {"id": gqlalchemy_id, "nameWithOwner": "memgraph/gqlalchemy", "visibility": "PUBLIC"}

    gqlalchemy_id = graph._db.query(
        "MATCH (r:Repository {address: 'github:memgraph/gqlalchemy'}) RETURN r.node_id AS id"
    )[0]["id"]
    replay.edit("Item:", transfer)
    replay.edit("Freshness:", transfer)
    fetch(harness, "s1", "gh issue view 2000 -R memgraph/memgraph", "toolu_3")
    sweep(graph, source)

    moved = graph._db.query(
        "MATCH (old:Issue)-[:MOVED_TO]->(new:Issue) RETURN old.address AS old, new.address AS new, old.number AS number"
    )
    assert moved == [{"old": None, "new": "github:memgraph/gqlalchemy#901", "number": 2000}]
    assert graph.read(address("memgraph/memgraph#2000")).outcome == "miss"
    assert graph.read(address("memgraph/gqlalchemy#901")).outcome == "hit"


@pytest.mark.parametrize(
    "state,label,expected",
    [("CLOSED", "wayfinder:map", True), ("OPEN", "wayfinder:map", False), ("CLOSED", "bug", False)],
)
def test_closed_design_maps_distinguish_source_freshness_from_implementation(
    graph, harness, source, replay, state, label, expected
):
    """Reading a fresh cached plan must preserve its historical interpretation."""
    replay.edit(
        "Item:",
        lambda body: (body["data"]["repository"]["issueOrPullRequest"] or {}).update(
            state=state, labels={"nodes": [{"name": label}]}
        ),
    )
    start(harness, "map-session", "ante")
    fetch(harness, "map-session", "gh issue view 2000 -R memgraph/memgraph", "map-fetch")
    assert sweep(graph, source).resolved == 1
    calls = replay.calls
    result = ask_memory(harness, "map-session", "memgraph/memgraph#2000", "map-read")
    assert ("Closed design map:" in result.text) is expected
    assert result.text.startswith("[resource hit]")
    assert "fetched_at:" in result.text and "updated_at (GitHub):" in result.text
    assert result.structured["resource"]["state"] == state
    assert replay.calls == calls
