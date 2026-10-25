"""Tests for managing a workload's named eval instance, with Docker stubbed at the command line."""

import subprocess

import pytest
from context_graph_eval.instance import IMAGE, ensure_running, run_args, state, volume
from context_graph_eval.workloads import workload

DEV = workload("lme-s-dev")


class Docker:
    """Records docker commands and answers ``inspect`` with a fixed container state."""

    def __init__(self, status: str | None, *, fail: bool = False):
        self.status, self.fail, self.calls = status, fail, []

    def __call__(self, args):
        self.calls.append(list(args))
        if args[1] == "inspect":
            ok = self.status is not None
            return subprocess.CompletedProcess(args, 0 if ok else 1, stdout=f"{self.status}\n" if ok else "")
        return subprocess.CompletedProcess(args, 1 if self.fail else 0, stdout="", stderr="boom")


def test_an_instance_is_named_for_its_workload_and_keeps_its_graph_on_a_volume():
    args = run_args(DEV)

    assert args[args.index("--name") + 1] == "context-graph-eval-lme-s-dev"
    assert f"{volume(DEV)}:/var/lib/memgraph" in args
    assert "127.0.0.1:7792:7687" in args
    assert IMAGE in args


def test_an_instance_survives_a_crash_and_a_restart():
    """A build is hours of work: the WAL and recovery on start keep it."""
    args = run_args(DEV)

    assert "--storage-wal-enabled=true" in args
    assert "--data-recovery-on-startup=true" in args
    assert "--storage-snapshot-on-exit=true" in args


def test_a_missing_instance_is_created():
    docker = Docker(None)

    instance = ensure_running(DEV, run=docker, wait=False)

    assert docker.calls[-1] == run_args(DEV)
    assert instance.url == "bolt://localhost:7792"


def test_a_stopped_instance_is_started_not_recreated():
    docker = Docker("exited")

    ensure_running(DEV, run=docker, wait=False)

    assert docker.calls[-1] == ["docker", "start", DEV.instance]


def test_a_running_instance_is_left_alone():
    docker = Docker("running")

    ensure_running(DEV, run=docker, wait=False)

    assert len(docker.calls) == 1
    assert state(DEV, run=docker) == "running"


def test_a_container_that_exits_on_start_fails_at_once_with_its_logs():
    """A rejected flag stops Memgraph immediately; waiting out the timeout would hide why."""
    from context_graph_eval.instance import _alive

    docker = Docker("exited")

    with pytest.raises(RuntimeError, match="is exited"):
        _alive(DEV, run=docker)
    assert docker.calls[-1][:2] == ["docker", "logs"]


def test_a_docker_failure_is_reported_not_ignored():
    with pytest.raises(RuntimeError, match="boom"):
        ensure_running(DEV, run=Docker(None, fail=True), wait=False)
