"""Turning graph values into terminal text: pure functions, no I/O."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from context_graph_viewer import fmt

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-13T12:00:00Z",
        "2026-10-13T12:00:00.000000000+00:00",
        "2026-10-13T12:00:00.000000000+00:00[Etc/UTC]",
        "2026-10-13T12:00:00",
        1791892800,
        1791892800000,
        NOW,
    ],
)
def test_to_datetime_reads_every_shape_the_graph_stores(value):
    """Memgraph's nanosecond ISO strings, epoch seconds or ms, and naive values all read as the same UTC moment."""
    assert fmt.to_datetime(value) == NOW


@pytest.mark.parametrize("value", [None, "", "not a time"])
def test_to_datetime_gives_none_for_nothing_readable(value):
    assert fmt.to_datetime(value) is None


def test_ago_and_duration():
    assert fmt.ago(NOW - timedelta(seconds=5), NOW) == "5s ago"
    assert fmt.ago(NOW - timedelta(minutes=3), NOW) == "3m ago"
    assert fmt.ago(NOW - timedelta(days=2, hours=1), NOW) == "2d ago"
    assert fmt.ago(None, NOW) == "—"
    assert fmt.duration(NOW, NOW + timedelta(seconds=4.25)) == "4.2s"
    assert fmt.duration(NOW, NOW + timedelta(seconds=187)) == "3m07s"
    assert fmt.duration(NOW, None) == ""


def test_bar_places_the_span_within_the_window():
    start, end = NOW, NOW + timedelta(seconds=100)
    assert fmt.bar(start + timedelta(seconds=50), end, start, end, width=10) == "     █████"
    assert fmt.bar(start, None, start, end, width=10) == "▌"
    assert fmt.bar(None, None, start, end, width=10) == ""


def test_action_label_prefers_the_tools_self_description():
    call = {
        "type": "tool_call",
        "tool": "Bash",
        "props": json.dumps({"tool_input": {"command": "ls", "description": "List files"}}),
    }
    assert fmt.action_label(call) == "Bash List files"
    read = {"type": "tool_call", "tool": "Read", "props": json.dumps({"tool_input": {"file_path": "/a/b.py"}})}
    assert fmt.action_label(read) == "Read /a/b.py"
    broken = {"type": "tool_call", "tool": "Bash", "props": "{not json"}
    assert fmt.action_label(broken) == "Bash"
    message = {"type": "user_message", "text": "fix\n the   bug"}
    assert fmt.action_label(message) == "user_message fix the bug"
    assert fmt.action_label({"type": "user_message", "text": "x" * 100}, width=10) == "user_mess…"


def test_project_name():
    assert fmt.project_name("/work/billing/") == "billing"
    assert fmt.project_name(None) == "—"


def test_session_ids_are_cut_only_as_far_as_it_takes_to_tell_them_apart():
    codex = ["01a13c3e-aaaa", "01a13c3e-bbbb", "01a13a69-cccc"]
    assert fmt.id_width(codex) == 10
    assert fmt.id_width(["6142ffe9-1", "01a13eef-2"]) == 8
    assert fmt.id_width([]) == 8


def test_a_path_is_cut_from_the_left_so_its_own_name_survives():
    assert fmt.clip_start("/a/very/long/path/to/billing", 14) == "…th/to/billing"
    assert fmt.clip_start("/short", 14) == "/short"
