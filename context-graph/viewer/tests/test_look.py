"""How the viewer looks and what it says when there is nothing to show: Memgraph's brand, and honest empty states."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_app import SIZE, _app, _rows, _settle, _text
from textual.widgets import DataTable

from context_graph_viewer import theme
from context_graph_viewer.dashboard import DashboardScreen
from context_graph_viewer.knowledge import EntityListScreen
from context_graph_viewer.session import SessionScreen

pytest.importorskip("actions_graph")
pytest.importorskip("sessions_graph")

SOURCE = Path(__file__).parent.parent / "src" / "context_graph_viewer"


def test_the_viewer_is_the_brands_dark_surface_with_orange_as_its_accent():
    assert theme.MEMGRAPH.background == "#231F20"
    assert theme.MEMGRAPH.primary == theme.MEMGRAPH.accent == "#FB6E00"
    assert (theme.MEMGRAPH.success, theme.MEMGRAPH.error, theme.MEMGRAPH.warning) == ("#19AF66", "#DC2223", "#FFC500")


def test_no_view_names_a_colour_the_theme_does_not_own():
    colours = r"style=\"(?:bold )?(?:red|green|yellow|blue|cyan|magenta|white|black|dim)\""
    for path in SOURCE.glob("*.py"):
        assert not re.search(colours, path.read_text()), f"{path.name} names a colour; use context_graph_viewer.theme"


async def test_the_app_opens_in_the_memgraph_theme(capture):
    capture.session("billing-1")
    async with _app().run_test(size=SIZE) as pilot:
        await _settle(pilot)
        assert pilot.app.theme == theme.NAME


async def test_no_entities_says_where_they_come_from(capture):
    capture.session("billing-1")
    async with _app().run_test(size=SIZE) as pilot:
        await _settle(pilot)
        await pilot.press("k")
        await _settle(pilot)
        screen = pilot.app.screen
        assert isinstance(screen, EntityListScreen)
        text = _text(screen, "#prov")
        assert "No entities yet" in text
        assert "sessions-graph[reconciliation]" in text
        assert "no LLM key" in text


async def test_the_details_pane_says_how_to_fill_it(capture):
    capture.session("billing-1")
    async with _app().run_test(size=SIZE) as pilot:
        pilot.app.push_screen(SessionScreen("billing-1"))
        await _settle(pilot)
        assert "Move along the timeline" in _text(pilot.app.screen, "#details")


async def test_two_sessions_sharing_an_id_prefix_are_told_apart(capture):
    """Codex session ids start with a timestamp, so eight characters can name two sessions."""
    capture.session("billing-aaaa")
    capture.session("billing-bbbb")
    async with _app().run_test(size=SIZE) as pilot:
        await _settle(pilot)
        screen = pilot.app.screen
        assert isinstance(screen, DashboardScreen)
        ids = [row[0] for row in _rows(screen.query_one("#sessions", DataTable))]
        assert sorted(ids) == ["billing-a", "billing-b"]
