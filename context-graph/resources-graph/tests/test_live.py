"""One Sweep against real GitHub, so a GraphQL schema change can't hide behind the recordings.

Skipped unless GITHUB_TOKEN is set (tests only; the Sweep itself reads the config file).
"""

from __future__ import annotations

import os

import pytest

from resources_graph.address import Address
from resources_graph.github import GitHubSource, http_transport
from resources_graph.models import FETCHED
from resources_graph.sweep import sweep

requires_github_token = pytest.mark.skipif(not os.environ.get("GITHUB_TOKEN"), reason="GITHUB_TOKEN not set")


@requires_github_token
def test_live_sweep_stores_a_public_pull_request(graph):
    graph.record_touch("live", Address("item", "memgraph", "memgraph", 3600), FETCHED, discriminator="toolu_live")

    report = sweep(graph, GitHubSource(http_transport(os.environ["GITHUB_TOKEN"])))

    assert (report.resolved, report.unresolved) == (1, {})
    served = graph.read(Address("item", "memgraph", "memgraph", 3600))
    assert served.kind == "PullRequest"
    assert served.resource["changed_files"]
