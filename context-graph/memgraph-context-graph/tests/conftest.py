import pytest

from memgraph_context_graph import capture_log


@pytest.fixture(autouse=True)
def _capture_log_in_tmp(monkeypatch, tmp_path):
    """No test appends to the user's real capture-failure log; each gets its own."""
    log = tmp_path / "capture-failures.jsonl"
    monkeypatch.setattr(capture_log, "_path_override", log)
    return log
