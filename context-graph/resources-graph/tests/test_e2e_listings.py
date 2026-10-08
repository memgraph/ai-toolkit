"""Listings end to end on a real Memgraph: one Touch, members at full depth, exact and subsumed Cache Reads.

GitHub is replayed from memgraph/gqlalchemy's open issues, recorded whole (``fixtures/record.py``).
"""

from __future__ import annotations

import pytest

from agent_context_graph.events import SessionStartEvent, ToolStartEvent
from resources_graph import tool as tool_module
from resources_graph.address import parse_address
from resources_graph.sweep import sweep

from .test_e2e import ask_memory, config

ALL_OPEN = "gh issue list -R memgraph/gqlalchemy --limit 100"
FIRST_TEN = "gh issue list -R memgraph/gqlalchemy --limit 10"
SEARCH = 'gh issue list -R memgraph/gqlalchemy --search "graph" --limit 5'


def run(emit, command, tool_use_id, session_id="s1"):
    emit(
        ToolStartEvent(
            session_id=session_id, tool_name="Bash", tool_input={"command": command}, tool_use_id=tool_use_id
        )
    )


@pytest.fixture()
def started(harness):
    harness(SessionStartEvent(session_id="s1", user_id="ante"))
    harness(SessionStartEvent(session_id="s2", user_id="marko"))
    return harness


def stored_open_issues(graph):
    rows = graph._db.query(
        "MATCH (:Listing {key: 'github:memgraph/gqlalchemy/issues?state=open'})-[:HAS_MEMBER]->(i:Issue) "
        "RETURN i.labels AS labels"
    )
    return [row["labels"] for row in rows]


def test_a_listing_is_one_touch_and_its_members_are_stored_at_full_depth(graph, started, source):
    run(started, ALL_OPEN, "toolu_1")
    report = sweep(graph, source)

    listing = graph._db.query("MATCH (l:Listing)-[:LISTS_FROM]->(:Repository) RETURN properties(l) AS l")[0]["l"]
    assert listing["fully_expanded"] is True
    assert listing["member_count"] == listing["total_count"] > 25  # paged past the first 25
    assert report.resolved == 1
    assert graph._db.query("MATCH (t:Touch) RETURN count(t) AS n")[0]["n"] == 1  # no Touch per member
    touched = graph._db.query("MATCH (:Touch)-[:TOUCHED]->(l:Listing) RETURN l.key AS key")
    assert [row["key"] for row in touched] == ["github:memgraph/gqlalchemy/issues?state=open"]
    with_comments = graph._db.query(
        "MATCH (i:Issue)-[:HAS_COMMENT]->(c) WITH i, count(c) AS n WHERE n = i.comment_count RETURN count(i) AS n"
    )
    commented = graph._db.query("MATCH (i:Issue) WHERE i.comment_count > 0 RETURN count(i) AS n")
    assert with_comments[0]["n"] == commented[0]["n"]  # every comment of every member


def test_the_500_issues_twice_case_reask_hits_and_a_narrower_ask_is_subsumed(graph, started, source):
    run(started, ALL_OPEN, "toolu_1")
    sweep(graph, source)

    again = ask_memory(started, "s2", ALL_OPEN, "toolu_2", user_id="marko")
    bugs = ask_memory(started, "s2", "gh issue list -R memgraph/gqlalchemy --label bug", "toolu_3", user_id="marko")

    assert again.structured["outcome"] == "hit"
    assert again.structured["index_size"] == len(stored_open_issues(graph))
    assert bugs.structured["outcome"] == "subsumed"
    assert bugs.text.startswith(
        "[resource subsumed] github:memgraph/gqlalchemy/issues?labels=bug&state=open "
        "from github:memgraph/gqlalchemy/issues?state=open"
    )
    expected_bugs = [labels for labels in stored_open_issues(graph) if "bug" in [label.lower() for label in labels]]
    assert bugs.structured["index_size"] == len(expected_bugs) > 0
    assert all("bug" in [label.lower() for label in row["labels"]] for row in bugs.structured["index"])
    reads = graph._db.query(
        "MATCH (t:Touch {served_from_memory: true, outcome: 'subsumed'})-[:TOUCHED]->(l:Listing) RETURN l.key AS key"
    )
    assert [row["key"] for row in reads] == ["github:memgraph/gqlalchemy/issues?state=open"]


def test_a_truncated_listing_answers_only_shorter_asks_of_itself(graph, started, source):
    run(started, FIRST_TEN, "toolu_1")
    sweep(graph, source)

    assert graph._db.query("MATCH (l:Listing) RETURN l.fully_expanded AS f")[0]["f"] is False
    first_five = ask_memory(started, "s1", "gh issue list -R memgraph/gqlalchemy --limit 5", "toolu_2")
    first_ten = ask_memory(started, "s1", FIRST_TEN, "toolu_3")
    assert (first_five.structured["outcome"], first_five.structured["index_size"]) == ("hit", 5)
    assert first_ten.structured["index"][:5] == first_five.structured["index"]  # newest created first, both
    for broader in ("gh issue list -R memgraph/gqlalchemy --limit 50", "gh issue list -R memgraph/gqlalchemy -l bug"):
        assert ask_memory(started, "s1", broader, f"toolu_{broader}").structured["outcome"] == "miss"


def test_free_text_searches_only_hit_exactly(graph, started, source):
    run(started, SEARCH, "toolu_1")
    sweep(graph, source)

    exact = ask_memory(started, "s1", SEARCH, "toolu_2")
    narrower = ask_memory(started, "s1", SEARCH + " --label bug", "toolu_3")

    assert exact.structured["outcome"] == "hit"
    assert narrower.structured["outcome"] == "miss"


def test_members_are_read_one_at_a_time_from_the_index(graph, started, source):
    run(started, ALL_OPEN, "toolu_1")
    sweep(graph, source)
    index = ask_memory(started, "s1", ALL_OPEN, "toolu_2").structured["index"]

    member = ask_memory(started, "s1", f"memgraph/gqlalchemy#{index[0]['number']}", "toolu_3")

    assert member.structured["outcome"] == "hit"
    assert member.structured["kind"] == "Issue"
    assert len(member.structured["comments"]) == index[0]["comment_count"]


def test_a_refetch_replaces_the_members(graph, started, source, replay):
    run(started, ALL_OPEN, "toolu_1")
    sweep(graph, source)
    before = len(stored_open_issues(graph))

    def close_the_oldest(body):  # GitHub now lists one open issue fewer: the oldest was closed
        members = body["data"]["repository"]["members"]
        members["totalCount"] -= 1
        if not members["pageInfo"]["hasNextPage"]:
            members["nodes"] = members["nodes"][:-1]

    replay.edit("IndexIssues:", close_the_oldest)
    run(started, ALL_OPEN, "toolu_2")
    report = sweep(graph, source)

    assert len(stored_open_issues(graph)) == before - 1
    assert (report.checked, report.fetched, report.unchanged) == (1, 0, before - 1)  # nothing changed was refetched
    assert graph._db.query("MATCH (l:Listing) RETURN l.fully_expanded AS f")[0]["f"] is True


def test_the_index_is_paged(graph, started, source, monkeypatch):
    run(started, ALL_OPEN, "toolu_1")
    sweep(graph, source)
    monkeypatch.setattr(tool_module, "PAGE_SIZE", 10)

    first = ask_memory(started, "s1", ALL_OPEN, "toolu_2")
    result = tool_module.ResourceTool().call({"address": ALL_OPEN, "page": 2}, config("ante"))

    assert len(first.structured["index"]) == 10
    assert result.structured["page"] == 2
    assert [row["number"] for row in result.structured["index"]] != [row["number"] for row in first.structured["index"]]
    assert "page 2 of" in result.text


def test_prompted_listing_url_of_a_stored_listing_resolves_without_fetching(graph, started, source, replay):
    run(started, ALL_OPEN, "toolu_1")
    sweep(graph, source)
    calls = replay.calls
    address = parse_address("https://github.com/memgraph/gqlalchemy/issues")
    assert address is not None and address.limit == 25

    graph.record_touch("s1", address, "PROMPTED", discriminator="prompt-1")
    report = sweep(graph, source)

    assert (report.resolved, report.fetched, replay.calls) == (1, 0, calls)


def test_revalidating_a_listing_refetches_only_the_members_that_changed(graph, started, source, replay):
    run(started, ALL_OPEN, "toolu_1")
    sweep(graph, source)
    index = graph._db.query(
        "MATCH (:Listing)-[m:HAS_MEMBER {position: 0}]->(i:Issue) RETURN i.number AS number, i.updated_at AS updated_at"
    )[0]

    def touch_the_newest(body):  # GitHub reports the newest member updated since
        nodes = body["data"]["repository"]["members"]["nodes"]
        if nodes and nodes[0]["number"] == index["number"]:
            nodes[0]["updatedAt"] = "2030-01-01T00:00:00Z"

    replay.edit("IndexIssues:", touch_the_newest)
    # Its refetch by number answers with the content recorded for it on the listing's first page.
    recorded = next(
        node
        for key, body in replay.responses.items()
        if key.startswith("Issues:")
        for node in body["data"]["repository"]["members"]["nodes"]
        if node["number"] == index["number"]
    )
    replay.responses[
        f'Item:{{"number": {index["number"]}, "owner": "memgraph", "pageSize": 2, "repo": "gqlalchemy"}}'
    ] = {"data": {"repository": {"visibility": "PUBLIC", "issueOrPullRequest": recorded}}}
    run(started, ALL_OPEN, "toolu_2")
    report = sweep(graph, source)

    assert report.checked == 1
    assert report.fetched == 1
    assert report.unchanged == len(stored_open_issues(graph)) - 1
