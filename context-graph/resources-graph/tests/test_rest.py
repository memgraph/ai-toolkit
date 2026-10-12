"""The no-credentials REST source: the same content as GraphQL for single items and repositories; no listings."""

from __future__ import annotations

import json

import pytest

from resources_graph.address import Address, parse_address
from resources_graph.github import GitHubSource, UnresolvedError
from resources_graph.models import FETCHED
from resources_graph.rest import RestSource
from resources_graph.sweep import sweep

from .fixtures.replay import RestReplay


def item(number: int) -> Address:
    return Address("item", "memgraph", "memgraph", number)


@pytest.fixture()
def rest_replay():
    return RestReplay()


def same_fields(node):
    return {
        "typename": node["__typename"],
        "id": node["id"],
        "number": node["number"],
        "state": node["state"],
        "labels": sorted(label["name"] for label in node["labels"]["nodes"]),
        "comments": sorted(comment["id"] for comment in node["comments"]["nodes"]),
        "comment_count": node["comments"]["totalCount"],
        "repository": node["repository"]["id"],
        "references": sorted(n["source"]["id"] for n in node["timelineItems"]["nodes"] if n.get("source")),
    }


@pytest.mark.parametrize("number", [2000, 4962])
def test_rest_and_graphql_return_the_same_item(replay, page_size, rest_replay, number):
    graphql = GitHubSource(replay, page_size=page_size).fetch_item(item(number))
    rest = RestSource(rest_replay).fetch_item(item(number))

    assert same_fields(rest) == same_fields(graphql)
    if graphql["__typename"] == "PullRequest":
        assert [f["path"] for f in rest["files"]["nodes"]] == [f["path"] for f in graphql["files"]["nodes"]]
        review = sorted(c["id"] for t in rest["reviewThreads"]["nodes"] for c in t["comments"]["nodes"])
        assert review == sorted(c["id"] for t in graphql["reviewThreads"]["nodes"] for c in t["comments"]["nodes"])
        assert rest["merged"] == graphql["merged"]


def test_a_sweep_without_credentials_stores_the_same_resource(graph, rest_replay):
    graph.record_touch("s1", item(2000), FETCHED, discriminator="toolu_1")

    report = sweep(graph, RestSource(rest_replay))

    assert (report.resolved, report.unresolved) == (1, {})
    served = graph.read(item(2000))
    assert served.outcome == "hit"
    assert served.resource["node_id"] == "I_kwDOEbh4L86Hv9cx"  # GraphQL's id: one Resource whichever path fetched it
    assert len(served.comments) == 4
    assert graph.read(Address("repo", "memgraph", "memgraph")).resource["readme"]


def test_listings_wait_for_credentials(graph, rest_replay):
    listing = parse_address("gh issue list -R memgraph/memgraph --limit 10")
    assert listing is not None
    graph.record_touch("s1", listing, FETCHED, discriminator="toolu_1")

    report = sweep(graph, RestSource(rest_replay))

    assert report.unresolved == {"rate_limited": 1}
    assert graph.pending_touches()  # retried by the next Sweep, credentials or not


def test_missing_and_non_public(rest_replay):
    with pytest.raises(UnresolvedError) as missing:
        RestSource(rest_replay).fetch_item(item(999999))
    assert missing.value.reason == "not_found"

    key = "/repos/memgraph/memgraph|application/vnd.github+json"
    status, body = rest_replay.responses[key]
    rest_replay.responses[key] = [status, json.dumps({**json.loads(body), "private": True, "visibility": "private"})]
    with pytest.raises(UnresolvedError) as private:
        RestSource(rest_replay).fetch_item(item(2000))
    assert private.value.reason == "not_public"


def test_spent_budget_is_rate_limited(rest_replay):
    for key in rest_replay.responses:
        rest_replay.responses[key] = [403, '{"message": "API rate limit exceeded"}']

    with pytest.raises(UnresolvedError) as limited:
        RestSource(rest_replay).fetch_item(item(2000))
    assert limited.value.reason == "rate_limited"
