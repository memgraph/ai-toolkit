"""The viewer driven headless against a real Memgraph: what each view shows and how it moves between them."""

from __future__ import annotations

import sys

import pytest
from conftest import OTHER_PROJECT, USER, extract, mark_reconciled
from textual.app import App
from textual.widgets import DataTable, OptionList, Static, Tree

from context_graph_viewer.app import ViewerApp
from context_graph_viewer.base import ViewStatus, cursor_key, sync_table
from context_graph_viewer.dashboard import DashboardScreen, ProjectPicker
from context_graph_viewer.knowledge import EntityListScreen, NeighbourhoodScreen
from context_graph_viewer.session import SessionScreen
from context_graph_viewer.source import Reader, resolve_identity

pytest.importorskip("actions_graph")
pytest.importorskip("sessions_graph")

SIZE = (160, 48)


def _app() -> ViewerApp:
    return ViewerApp(resolve_identity(), poll_seconds=0)


async def _settle(pilot) -> None:
    """Let every running poll finish and its result reach the screen."""
    for _ in range(3):
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def _text(screen, selector: str) -> str:
    return str(screen.query_one(selector, Static).render())


def _status(screen) -> str:
    return str(screen.query_one(ViewStatus).render())


def _rows(table: DataTable) -> list[list[str]]:
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


async def test_dashboard_shows_sessions_pipeline_and_health(capture, memgraph):
    capture.session("billing-1", commands=("ls", "make test"), failing=("make test",))
    capture.session("crawler-1", cwd=OTHER_PROJECT)
    mark_reconciled(memgraph, "billing-1")
    async with _app().run_test(size=SIZE) as pilot:
        await _settle(pilot)
        screen = pilot.app.screen
        assert isinstance(screen, DashboardScreen)
        rows = _rows(screen.query_one("#sessions", DataTable))
        assert sorted(row[0] for row in rows) == ["billing-", "crawler-"]
        billing = next(row for row in rows if row[0] == "billing-")
        assert (billing[1], billing[4], billing[5]) == ("billing", "1", "done")
        assert "sessions: completed 1  pending 1" in _text(screen, "#pipeline")
        health = _text(screen, "#health")
        assert "no capture failures in the last 24h" in health
        assert "1 failed tool result(s)" in health
        assert "live" in _status(screen)
        assert "2 of 2 sessions in scope" in _status(screen)
        assert f"user {USER}" in screen.sub_title


async def test_an_empty_default_scope_says_where_the_sessions_are(capture):
    """Sessions captured before a user was configured: the landing screen says so, `a` shows them (#383)."""
    capture.session("early", user=None)
    async with _app().run_test(size=SIZE) as pilot:
        await _settle(pilot)
        screen = pilot.app.screen
        banner = screen.query_one("#banner", Static)
        assert banner.display
        assert "1 session(s) in the graph have no user: press a" in str(banner.render())
        await pilot.press("a")
        await _settle(pilot)
        assert not banner.display
        assert [row[0] for row in _rows(screen.query_one("#sessions", DataTable))] == ["early"]
        assert _rows(screen.query_one("#sessions", DataTable))[0][-1] == "unattributed"


async def test_capture_failures_reach_the_health_pane_and_their_session(capture):
    """A failure the hooks logged locally shows on the dashboard and on the session it belongs to."""
    from memgraph_context_graph import capture_log

    capture.session("flaky")
    assert capture_log.record_failure(
        "emit", ValueError("constraint violated"), connector="actions-graph", session_id="flaky", runtime="claude-code"
    )
    async with _app().run_test(size=SIZE) as pilot:
        await _settle(pilot)
        health = _text(pilot.app.screen, "#health")
        assert "1 capture failure(s) in the last 24h" in health
        assert "emit/actions-graph ValueError ×1" in health
        await pilot.press("enter")
        await _settle(pilot)
        assert isinstance(pilot.app.screen, SessionScreen)
        assert "1 capture failure(s)" in _text(pilot.app.screen, "#produced")


async def test_a_failed_poll_stays_on_its_own_view(capture, config_file):
    """One view's error must not be hidden by another view's successful poll (#383)."""
    from memgraph_context_graph.spaces import MemgraphTarget, Space

    capture.session("s1")
    app = _app()
    async with app.run_test(size=SIZE) as pilot:
        await _settle(pilot)
        dashboard = app.screen
        assert isinstance(dashboard, DashboardScreen)
        good = app.reader
        app.reader = Reader(Space("personal", MemgraphTarget("bolt://127.0.0.1:1", "", "", None)))
        dashboard.action_poll()
        await _settle(pilot)
        assert dashboard.error is not None
        assert "⚠" in _status(dashboard)
        assert dashboard.query_one("#sessions", DataTable).row_count == 1  # the last good data stays

        app.reader = good
        await pilot.press("enter")
        await _settle(pilot)
        session = app.screen
        assert isinstance(session, SessionScreen)
        assert session.error is None
        assert dashboard.error is not None


async def test_session_view_folds_the_subagent_and_shows_graph_text_literally(capture):
    """The Task call holds its subagent; a prompt with markup characters renders as written."""
    capture.subagent_session("parent")
    capture.session("plain")
    async with _app().run_test(size=SIZE) as pilot:
        await _settle(pilot)
        table = pilot.app.screen.query_one("#sessions", DataTable)
        table.move_cursor(row=[r[0] for r in _rows(table)].index("parent"))
        await pilot.press("enter")
        await _settle(pilot)
        tree = pilot.app.screen.query_one("#timeline", Tree)
        task = next(node for node in tree.root.children if "Task" in str(node.label))
        assert [str(child.label).split("  ")[0] for child in task.children] == ["▾ subagent Explore"]
        assert "Grep" in str(task.children[0].children[0].label)

        await pilot.press("escape")
        table.move_cursor(row=[r[0] for r in _rows(table)].index("plain"))
        await pilot.press("enter")
        await _settle(pilot)
        tree = pilot.app.screen.query_one("#timeline", Tree)
        tree.move_cursor(tree.root.children[0])
        await pilot.pause()
        assert "tidy the [billing] module" in _text(pilot.app.screen, "#details")


async def test_from_a_session_to_its_entities_and_through_the_neighbourhood(capture, memgraph):
    """`g`, `k`, enter to focus, enter to refocus, esc back along the path, `v` for the table."""
    capture.session("s1")
    mark_reconciled(memgraph, "s1")
    extract(memgraph, "s1", [("billing", "Project"), ("tokio", "Artifact"), ("serde", "Artifact")],
            [("billing", "depends_on", "tokio", 0.9), ("billing", "depends_on", "serde", 0.3),
             ("tokio", "related_to", "serde", 0.8)])  # fmt: skip
    app = _app()
    async with app.run_test(size=SIZE) as pilot:
        await _settle(pilot)
        await pilot.press("enter", "g")
        await _settle(pilot)
        assert isinstance(app.screen, EntityListScreen)
        assert sorted(row[0] for row in _rows(app.screen.query_one("#entities", DataTable))) == [
            "billing",
            "serde",
            "tokio",
        ]
        await pilot.press("escape", "escape", "k")
        await _settle(pilot)
        entities = app.screen.query_one("#entities", DataTable)
        entities.move_cursor(row=[r[0] for r in _rows(entities)].index("billing"))
        await pilot.press("enter")
        await _settle(pilot)
        nb = app.screen
        assert isinstance(nb, NeighbourhoodScreen)
        tree = nb.query_one("#nb", Tree)
        assert [str(group.label) for group in tree.root.children] == ["→ depends_on (1)"]
        assert "below it hidden" in _status(nb)  # serde's 0.3 is under the 0.5 floor

        await pilot.press("minus", "minus", "minus")
        await _settle(pilot)
        assert [str(group.label) for group in nb.query_one("#nb", Tree).root.children] == ["→ depends_on (2)"]

        tree = nb.query_one("#nb", Tree)
        leaf = next(c for c in tree.root.children[0].children if str(c.label).startswith("tokio"))
        tree.select_node(leaf)
        await _settle(pilot)
        assert isinstance(app.screen, NeighbourhoodScreen)
        assert app.screen.title.endswith("billing ▸ tokio")

        await pilot.press("v")
        await _settle(pilot)
        table = app.screen.query_one("#nb", DataTable)
        assert {row[2].rstrip(" ↺") for row in _rows(table)} == {"billing", "serde"}

        await pilot.press("escape")
        assert app.screen is nb
    NeighbourhoodScreen.view_mode, NeighbourhoodScreen.floor = "tree", 0.5


async def test_keys_jump_between_panes_and_fullscreen_one(capture):
    capture.session("s1")
    async with _app().run_test(size=SIZE) as pilot:
        await _settle(pilot)
        await pilot.press("2")
        assert pilot.app.focused is not None and pilot.app.focused.id == "pipeline-pane"
        await pilot.press("f")
        assert pilot.app.screen.maximized is not None
        await pilot.press("f")
        assert pilot.app.screen.maximized is None
        await pilot.press("1")
        assert pilot.app.focused.id == "sessions"


async def test_filter_and_project_picker_narrow_the_sessions(capture):
    capture.session("billing-1")
    capture.session("crawler-1", cwd=OTHER_PROJECT)
    app = _app()
    async with app.run_test(size=SIZE) as pilot:
        await _settle(pilot)
        await pilot.press("slash", *"crawl")
        table = app.screen.query_one("#sessions", DataTable)
        assert [row[0] for row in _rows(table)] == ["crawler-"]
        await pilot.press("escape")
        assert table.row_count == 2

        await pilot.press("p")
        await pilot.pause()
        assert isinstance(app.screen, ProjectPicker)
        options = app.screen.query_one(OptionList)
        options.highlighted = 2  # "all projects", then the projects in order
        await pilot.press("enter")
        await _settle(pilot)
        assert app.scope.project == OTHER_PROJECT
        assert [row[0] for row in _rows(app.screen.query_one("#sessions", DataTable))] == ["crawler-"]
        assert "project crawler" in app.screen.sub_title


async def test_sync_table_updates_in_place_and_follows_the_selected_row():
    """Same rows in the same order: cells change, the cursor stays. Reordered: the cursor follows its row."""

    class Harness(App[None]):
        def compose(self):
            yield DataTable(id="t", cursor_type="row")

    columns = [("name", "name"), ("n", "n")]
    async with Harness().run_test() as pilot:
        table = pilot.app.query_one(DataTable)
        sync_table(table, columns, [("a", ["a", 1]), ("b", ["b", 2]), ("c", ["c", 3])])
        table.move_cursor(row=1)
        sync_table(table, columns, [("a", ["a", 1]), ("b", ["b", 20]), ("c", ["c", 3])])
        await pilot.pause()
        assert (cursor_key(table), table.get_cell("b", "n")) == ("b", 20)
        sync_table(table, columns, [("c", ["c", 3]), ("a", ["a", 1]), ("b", ["b", 20])])
        await pilot.pause()
        assert (cursor_key(table), table.cursor_row) == ("b", 2)
        sync_table(table, columns, [("c", ["c", 3])])
        await pilot.pause()
        assert cursor_key(table) == "c"


def test_ui_without_the_viewer_installed_says_to_run_setup(monkeypatch, capsys):
    from memgraph_context_graph.cli import main

    monkeypatch.setitem(sys.modules, "context_graph_viewer.cli", None)
    assert main(["ui"]) == 1
    assert "memgraph-context-graph setup" in capsys.readouterr().err


def test_ui_reports_an_edition_it_cannot_resolve(config_file, capsys):
    from context_graph_viewer.cli import main
    from memgraph_context_graph import spaces
    from memgraph_context_graph.adapters import _identity

    config_file.write_text('edition = "galactic"\n', encoding="utf-8")
    _identity._reset_cache()
    spaces._reset_cache()
    assert main([]) == 2
    assert "galactic" in capsys.readouterr().err
