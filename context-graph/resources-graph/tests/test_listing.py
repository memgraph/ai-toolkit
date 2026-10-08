"""Listing matching: exact answers, subsumption truth table, local member filtering."""

from __future__ import annotations

import pytest

from resources_graph.address import listing
from resources_graph.listing import exact_serves, member_matches, subsumes


def issues(limit=None, query=None, **filters):
    return listing("memgraph", "memgraph", "issues", filters, query, limit)


def pulls(**filters):
    return listing("memgraph", "memgraph", "pulls", filters)


@pytest.mark.parametrize(
    ("fully_expanded", "member_count", "asked_limit", "serves"),
    [
        (True, 686, None, True),
        (True, 686, 30, True),
        (False, 500, 30, True),  # a prefix of what is stored, in the same order
        (False, 500, 500, True),
        (False, 500, 1000, False),
        (False, 500, None, False),
    ],
)
def test_exact_match(fully_expanded, member_count, asked_limit, serves):
    wanted = issues(limit=asked_limit)
    assert exact_serves(fully_expanded=fully_expanded, member_count=member_count, wanted=wanted) is serves


@pytest.mark.parametrize(
    ("stored", "wanted", "expected"),
    [
        (issues(), issues(labels="bug"), True),
        (issues(), issues(labels="bug,docs"), True),
        (issues(labels="bug"), issues(labels="bug,docs"), True),
        (issues(labels="bug"), issues(), False),
        (issues(labels="bug,docs"), issues(labels="bug"), False),
        (issues(state="all"), issues(state="closed"), True),
        (issues(state="all"), issues(state="open", author="alice"), True),
        (issues(state="open"), issues(state="closed"), False),
        (issues(state="open"), issues(state="all"), False),
        (issues(author="alice"), issues(author="alice", labels="bug"), True),
        (issues(author="alice"), issues(author="bob"), False),
        (issues(author="alice"), issues(), False),
        (issues(milestone="v3"), issues(milestone="v3", assignee="bob"), True),
        (issues(), issues(query="replication"), False),  # free text is exact-only
        (issues(query="replication"), issues(query="replication", labels="bug"), False),
        (pulls(state="closed"), pulls(state="merged"), True),  # gh's closed PRs include merged ones
        (pulls(state="merged"), pulls(state="closed"), False),
        (issues(), pulls(), False),
        (issues(), listing("memgraph", "mage", "issues"), False),
    ],
)
def test_subsumption(stored, wanted, expected):
    assert subsumes(stored, wanted) is expected


MEMBER = {
    "state": "OPEN",
    "labels": ["bug", "Docs"],
    "author": "Alice",
    "assignees": ["bob"],
    "milestone": "v3",
}


@pytest.mark.parametrize(
    ("wanted", "matches"),
    [
        (issues(), True),
        (issues(labels="bug"), True),
        (issues(labels="docs,bug"), True),  # labels compare case-insensitively
        (issues(labels="bug,feature"), False),
        (issues(state="closed"), False),
        (issues(state="all"), True),
        (issues(author="alice"), True),
        (issues(author="carol"), False),
        (issues(assignee="bob"), True),
        (issues(assignee="alice"), False),
        (issues(milestone="v3"), True),
        (issues(milestone="v4"), False),
    ],
)
def test_member_matches(wanted, matches):
    assert member_matches(MEMBER, wanted) is matches


def test_merged_pull_requests_match_closed_but_not_open():
    merged = {**MEMBER, "state": "MERGED"}
    assert member_matches(merged, pulls(state="closed"))
    assert member_matches(merged, pulls(state="merged"))
    assert not member_matches(merged, pulls(state="open"))
