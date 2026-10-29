"""The local capture-failure log: where it lives, what a line holds, and that it can't hurt a hook."""

import io
import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone

import pytest

from memgraph_context_graph import capture_log
from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.hooks.cli import main as hook_main


@pytest.fixture(autouse=True)
def _reset_config_cache():
    _identity._reset_cache()
    yield
    _identity._reset_cache()


def _lines(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


class TestPath:
    def test_default_config_logs_under_local_state(self, monkeypatch):
        monkeypatch.setattr(capture_log, "_path_override", None)
        monkeypatch.delenv(_identity.CONFIG_PATH_ENV, raising=False)
        assert capture_log.path().parts[-4:] == (".local", "state", "context-graph", "capture-failures.jsonl")

    def test_config_override_logs_beside_that_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(capture_log, "_path_override", None)
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "eval.toml"))
        assert capture_log.path() == tmp_path / "eval.capture-failures.jsonl"

    def test_naming_the_default_config_explicitly_keeps_the_default_log(self, monkeypatch):
        monkeypatch.setattr(capture_log, "_path_override", None)
        monkeypatch.delenv(_identity.CONFIG_PATH_ENV, raising=False)
        default = capture_log.path()
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(_identity.default_config_file()))
        assert capture_log.path() == default


class TestWrite:
    def test_round_trip(self, _capture_log_in_tmp):
        try:
            raise ValueError("Could not connect") from ConnectionRefusedError("refused")
        except ValueError as exc:
            assert capture_log.record_failure(
                "connect", exc, runtime="claude-code", session_id="s1", connector="actions-graph"
            )

        (line,) = _lines(_capture_log_in_tmp)
        assert line["v"] == capture_log.SCHEMA_VERSION
        assert line["stage"] == "connect"
        assert line["error_type"] == "ValueError"
        assert line["cause_type"] == "ConnectionRefusedError"
        assert line["where"].endswith("in test_round_trip")
        assert line["level"] == "error"
        assert "memgraph-context-graph" in line["versions"]

        (failure,) = capture_log.read_recent()
        assert (failure.stage, failure.connector, failure.session_id) == ("connect", "actions-graph", "s1")

    def test_file_and_directory_are_private(self, monkeypatch, tmp_path):
        log = tmp_path / "state" / "capture-failures.jsonl"
        monkeypatch.setattr(capture_log, "_path_override", log)
        capture_log.record_failure("args", error="x")
        assert log.stat().st_mode & 0o777 == 0o600
        assert log.parent.stat().st_mode & 0o777 == 0o700

    def test_configured_secrets_are_redacted(self, monkeypatch, tmp_path, _capture_log_in_tmp):
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
        _identity.write_config(
            user_id="u", memgraph_user="alice", memgraph_password="hunter2", openai_api_key="sk-abc123"
        )
        capture_log.record_failure("connect", error="auth failed for alice with hunter2 using sk-abc123")

        (line,) = _lines(_capture_log_in_tmp)
        assert line["error"] == "auth failed for *** with *** using ***"
        assert line["memgraph"] == "localhost:7687/memgraph"

    def test_a_line_never_exceeds_the_cap(self, _capture_log_in_tmp):
        capture_log.record_failure("emit", error="x" * 20_000, cwd="/" + "d" * 20_000, tool_name="t" * 1000)
        raw = _capture_log_in_tmp.read_bytes()
        assert len(raw) <= capture_log.MAX_LINE_BYTES
        assert json.loads(raw)["stage"] == "emit"

    def test_never_raises_when_the_log_cannot_be_written(self, monkeypatch, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.mkdir(mode=0o500)
        monkeypatch.setattr(capture_log, "_path_override", blocked / "sub" / "capture-failures.jsonl")
        assert capture_log.record_failure("connect", error="x") is False

    def test_rotates_past_the_size_cap(self, monkeypatch, _capture_log_in_tmp):
        monkeypatch.setattr(capture_log, "ROTATE_BYTES", 2000)
        for index in range(20):
            capture_log.record_failure("emit", error=f"failure {index}")

        rotated = _capture_log_in_tmp.with_name(_capture_log_in_tmp.name + ".1")
        assert rotated.exists()
        assert _capture_log_in_tmp.stat().st_size < 2000 + capture_log.MAX_LINE_BYTES
        # Both files are read; nothing in them is lost to the reader.
        assert {f.error for f in capture_log.read_recent()} >= {"failure 19", "failure 18"}

    def test_concurrent_writers_never_tear_a_line(self, tmp_path):
        log = tmp_path / "capture-failures.jsonl"
        writer = textwrap.dedent(
            f"""
            from pathlib import Path
            from memgraph_context_graph import capture_log
            capture_log._path_override = Path({str(log)!r})
            for index in range(50):
                capture_log.record_failure("emit", error="y" * 3000)
            """
        )
        processes = [subprocess.Popen([sys.executable, "-c", writer]) for _ in range(8)]
        assert all(process.wait(timeout=120) == 0 for process in processes)

        lines = log.read_text().splitlines()
        assert len(lines) == 400
        assert all(json.loads(line)["stage"] == "emit" for line in lines)


class TestRead:
    def test_skips_lines_that_do_not_parse(self, _capture_log_in_tmp):
        capture_log.record_failure("connect", error="first")
        with _capture_log_in_tmp.open("a") as handle:
            handle.write('{"v":1,"at":"torn\n')
            handle.write("not json at all\n")
        capture_log.record_failure("connect", error="second")

        assert [f.error for f in capture_log.read_recent()] == ["second", "first"]

    def test_since_and_summary_group_repeats(self, _capture_log_in_tmp):
        old = datetime.now(timezone.utc) - timedelta(days=2)
        with _capture_log_in_tmp.open("a") as handle:
            handle.write(
                json.dumps(
                    {"v": 1, "at": old.isoformat(), "stage": "connect", "error_type": "ValueError", "error": "old"}
                )
                + "\n"
            )
        for session in ("s1", "s2", "s1"):
            capture_log.record_failure(
                "emit", error_type="ClientError", error=session, connector="actions-graph", session_id=session
            )
            time.sleep(0.002)
        capture_log.record_failure("config", level="warning", error_type="MalformedConfig", error="line 3")

        (config, emit) = capture_log.recent_summary(hours=24)
        assert (config.stage, config.level, config.count) == ("config", "warning", 1)
        assert (emit.stage, emit.connector, emit.count, emit.last_error) == ("emit", "actions-graph", 3, "s1")
        assert emit.session_ids == ("s1", "s2")
        assert len(capture_log.summarize(capture_log.read_recent())) == 3


class TestHookFailuresAreLogged:
    """Hook runs that fail before reaching any Memgraph: no database needed."""

    def _run(self, monkeypatch, argv, payload=None):
        monkeypatch.setattr(
            "sys.stdin", io.StringIO(json.dumps(payload or {"hook_event_name": "Stop", "session_id": "s1"}))
        )
        return hook_main(argv)

    def test_a_bad_flag_never_exits_2(self, monkeypatch, capsys, _capture_log_in_tmp):
        assert self._run(monkeypatch, ["run", "claude-code", "--no-such-flag"]) == 0
        assert capsys.readouterr().out.strip() == '{"continue": true}'

        (line,) = _lines(_capture_log_in_tmp)
        assert (line["stage"], line["error_type"], line["runtime"]) == ("args", "HookArgumentError", "claude-code")

    def test_an_unknown_runtime_is_a_non_blocking_failure(self, monkeypatch, _capture_log_in_tmp):
        assert self._run(monkeypatch, ["run", "no-such-runtime"]) == 1
        assert self._run(monkeypatch, ["run"]) == 1

        assert [line["error_type"] for line in _lines(_capture_log_in_tmp)] == ["UnknownRuntime", "MissingRuntime"]

    def test_an_unreachable_memgraph_is_logged_with_its_cause(self, monkeypatch, tmp_path, _capture_log_in_tmp):
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
        _identity.write_config(user_id="u", memgraph_url="bolt://127.0.0.1:1")
        payload = {"hook_event_name": "PostToolUse", "session_id": "s9", "tool_name": "Bash", "cwd": "/repo"}

        assert self._run(monkeypatch, ["run", "claude-code", "--connector", "actions-graph"], payload) == 0

        (line,) = _lines(_capture_log_in_tmp)
        assert line["stage"] == "connect"
        assert (line["session_id"], line["tool_name"], line["cwd"], line["hook_event"]) == (
            "s9",
            "Bash",
            "/repo",
            "PostToolUse",
        )
        assert line["cause_type"].startswith("neo4j.exceptions.")
        assert line["memgraph"] == "127.0.0.1:1/memgraph"

    def test_strict_mode_logs_before_raising(self, monkeypatch, tmp_path, _capture_log_in_tmp):
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
        _identity.write_config(user_id="u", memgraph_url="bolt://127.0.0.1:1")

        with pytest.raises(ValueError):
            self._run(monkeypatch, ["run", "claude-code", "--connector", "actions-graph", "--strict"])
        assert _lines(_capture_log_in_tmp)[0]["stage"] == "connect"

    def test_an_unsupported_connector_is_a_setup_failure(self, monkeypatch, _capture_log_in_tmp):
        assert self._run(monkeypatch, ["run", "claude-code", "--connector", "nope-graph"]) == 0
        (line,) = _lines(_capture_log_in_tmp)
        assert (line["stage"], line["error"]) == (
            "setup",
            "Unsupported connector: nope-graph; choose from skills-graph, actions-graph, sessions-graph, resources-graph",
        )

    def test_a_malformed_config_is_logged_as_a_warning(self, monkeypatch, tmp_path, _capture_log_in_tmp):
        config = tmp_path / "config.toml"
        config.write_text('[memgraph]\nurl "bolt://elsewhere:7687"\n\n[capture]\nconnectors = ""\n')
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(config))

        assert self._run(monkeypatch, ["run", "claude-code"]) == 0

        (line,) = _lines(_capture_log_in_tmp)
        assert (line["stage"], line["level"], line["error_type"]) == ("config", "warning", "MalformedConfig")
        assert "line 2" in line["error"]

    def test_a_successful_hook_writes_nothing(self, monkeypatch, _capture_log_in_tmp):
        assert self._run(monkeypatch, ["run", "codex"]) == 0
        assert not _capture_log_in_tmp.exists()


def test_doctor_reports_failures_without_failing(monkeypatch, capsys, _capture_log_in_tmp):
    from memgraph_context_graph.cli import _check_capture_failures

    capture_log.record_failure("connect", error_type="ValueError", error="Could not connect", connector=None)
    check = _check_capture_failures()
    assert check["ok"] is True
    assert check.get("info") is True
    assert check["detail"].startswith("1 in the last 24h; last: connect ValueError")
    assert check["data"][0]["count"] == 1


def test_malformed_config_fails_doctor_and_shows_in_config_show(monkeypatch, tmp_path, capsys):
    from memgraph_context_graph.cli import _check_config
    from memgraph_context_graph.cli import main as top_level_main

    config = tmp_path / "config.toml"
    config.write_text('[identity]\nuser_id = "u"\n\n[memgraph]\nurl: bolt://elsewhere:7687\n')
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(config))

    check = _check_config()
    assert check["ok"] is False
    assert "malformed" in check["detail"]
    assert "line 5" in check["detail"]

    assert top_level_main(["config", "show"]) == 0
    assert "WARNING: malformed config" in capsys.readouterr().out


def test_an_unreadable_config_is_a_problem_not_a_crash(monkeypatch, tmp_path):
    config = tmp_path / "config.toml"
    config.write_bytes(b'[identity]\nuser_id = "\xff\xfe"\n')
    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(config))

    loaded = _identity.load_config()
    assert loaded.user_id is None
    assert loaded.problems and loaded.problems[0].startswith("unreadable: UnicodeDecodeError")


def test_a_blackholed_memgraph_fails_inside_the_hook_timeout(monkeypatch, tmp_path, _capture_log_in_tmp):
    """The driver's own 30 s connect timeout equalled the hook timeout, so the runtime killed the hook unlogged."""
    from memgraph_context_graph.hooks.runner import HOOK_CONNECT_TIMEOUT

    monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(tmp_path / "config.toml"))
    _identity.write_config(user_id="u", memgraph_url="bolt://10.255.255.1:7687")
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name":"Stop","session_id":"s1"}'))

    start = time.monotonic()
    assert hook_main(["run", "claude-code", "--connector", "actions-graph", "--connector", "skills-graph"]) == 0
    assert time.monotonic() - start < HOOK_CONNECT_TIMEOUT + 10
    assert [line["stage"] for line in _lines(_capture_log_in_tmp)] == ["connect"]


def test_a_hook_blocked_past_its_deadline_answers_and_logs_where(tmp_path):
    """A query waiting on Memgraph's storage never raises, so the runtime killed the hook at its timeout, unlogged.

    Run as its own process: the deadline ends the process, as it must to beat a hook thread that is still blocked.
    """
    log = tmp_path / "capture-failures.jsonl"
    hook = textwrap.dedent(
        f"""
        import dataclasses, sys, threading
        from pathlib import Path
        from memgraph_context_graph import AgentLink, capture_log
        from memgraph_context_graph.adapters.claude_code import PLUGIN
        from memgraph_context_graph.hooks import runner

        capture_log._path_override = Path({str(log)!r})
        runner.create_link = lambda *args, **kwargs: AgentLink()

        class BlockedAdapter:
            def __init__(self, link, session_id=None):
                pass

            def handle_payload(self, payload):
                threading.Event().wait()

        sys.exit(runner.run_hook(dataclasses.replace(PLUGIN, adapter_class=BlockedAdapter), ["--deadline", "1"]))
        """
    )

    start = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", hook],
        input='{"hook_event_name":"Stop","session_id":"s1"}',
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert time.monotonic() - start < 30
    assert (done.returncode, done.stdout.strip()) == (0, '{"continue": true}')
    (line,) = _lines(log)
    assert (line["stage"], line["error_type"], line["session_id"]) == ("translate", "HookDeadline", "s1")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_writer_cost_is_small(_capture_log_in_tmp):
    start = time.perf_counter()
    for _ in range(200):
        capture_log.record_failure("emit", error="z")
    assert (time.perf_counter() - start) / 200 < 0.01


def test_forgetting_a_session_removes_its_lines_from_the_log_and_its_rotation(_capture_log_in_tmp):
    capture_log.record_failure("emit", error="old", session_id="gone")
    os.replace(_capture_log_in_tmp, _capture_log_in_tmp.with_name(_capture_log_in_tmp.name + ".1"))
    capture_log.record_failure("emit", error="a", session_id="gone")
    capture_log.record_failure("emit", error="b", session_id="kept")
    _capture_log_in_tmp.write_bytes(_capture_log_in_tmp.read_bytes() + b"not json\n")

    assert capture_log.forget_session("gone") == 2

    assert [failure.session_id for failure in capture_log.read_recent()] == ["kept"]
    assert b"not json" in _capture_log_in_tmp.read_bytes(), "lines it can't read are left alone"
    assert capture_log.forget_session("gone") == 0
