"""Tests for the sessions-graph enrich CLI (argument parsing and dispatch).

Requires the sessions-graph[enrichment] extra; skips cleanly if unavailable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("actions_graph", reason="actions-graph not installed")
pytest.importorskip("unstructured2graph", reason="unstructured2graph not installed")

from sessions_graph.cli import main
from sessions_graph.enrichment import EnrichmentSummary


def test_main_no_args_prints_help_and_returns_2(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out


def test_main_help_flag_returns_0(capsys):
    assert main(["--help"]) == 0
    assert "usage" in capsys.readouterr().out


def test_main_unknown_command_returns_2(capsys):
    assert main(["bogus"]) == 2


def test_enrich_requires_session_or_pending():
    with pytest.raises(SystemExit):
        main(["enrich"])


def test_enrich_single_session_success(capsys):
    fake_graph = MagicMock()
    fake_graph.enrich_session = AsyncMock(
        return_value=EnrichmentSummary(session_id="s-1", status="completed", texts_considered=2, texts_deduped=1)
    )
    fake_wrapper = MagicMock()
    fake_wrapper.initialize = AsyncMock()
    fake_wrapper.afinalize = AsyncMock()

    with (
        patch("sessions_graph.SessionsGraph", return_value=fake_graph),
        patch("lightrag_memgraph.MemgraphLightRAGWrapper", return_value=fake_wrapper),
    ):
        exit_code = main(["enrich", "--session", "s-1"])

    assert exit_code == 0
    fake_graph.enrich_session.assert_awaited_once()
    assert fake_graph.enrich_session.call_args.args[0] == "s-1"
    assert "OK s-1" in capsys.readouterr().out


def test_enrich_pending_sweeps_all_sessions_and_reports_failure(capsys):
    fake_graph = MagicMock()
    fake_graph.get_pending_enrichment_sessions.return_value = ["s-1", "s-2"]
    fake_graph.enrich_session = AsyncMock(
        side_effect=[
            EnrichmentSummary(session_id="s-1", status="completed", texts_considered=1, texts_deduped=1),
            EnrichmentSummary(session_id="s-2", status="failed", texts_considered=1, texts_deduped=1, error="boom"),
        ]
    )
    fake_wrapper = MagicMock()
    fake_wrapper.initialize = AsyncMock()
    fake_wrapper.afinalize = AsyncMock()

    with (
        patch("sessions_graph.SessionsGraph", return_value=fake_graph),
        patch("lightrag_memgraph.MemgraphLightRAGWrapper", return_value=fake_wrapper),
    ):
        exit_code = main(["enrich", "--pending", "--limit", "5"])

    assert exit_code == 1  # one session failed
    fake_graph.get_pending_enrichment_sessions.assert_called_once_with(limit=5)
    assert fake_graph.enrich_session.await_count == 2
    out = capsys.readouterr()
    assert "OK s-1" in out.out
    assert "boom" in out.err


def test_enrich_pending_with_no_sessions_returns_0_without_constructing_lightrag(capsys):
    fake_graph = MagicMock()
    fake_graph.get_pending_enrichment_sessions.return_value = []

    with (
        patch("sessions_graph.SessionsGraph", return_value=fake_graph),
        patch("lightrag_memgraph.MemgraphLightRAGWrapper") as mock_wrapper_cls,
    ):
        exit_code = main(["enrich", "--pending"])

    assert exit_code == 0
    assert "No sessions to enrich" in capsys.readouterr().out
    mock_wrapper_cls.assert_not_called()
