"""Each workload's own eval instance: a named Memgraph container whose graph outlives the run.

A workload always runs against ``context-graph-eval-<workload>`` on its own
port, with its data on a named volume. Building a graph takes hours, asking
it minutes, so the graph is kept: the next run of the same workload finds it
and asks it again unless what built it has changed (``pipeline.decide``).

Durability is on, because a build is too expensive to lose to a crash or a
restart: the write-ahead log records every change, a snapshot is taken on
exit, and the instance recovers both on start. Periodic snapshots run only
hourly -- Memgraph refuses the WAL without them, and a snapshot in progress
fails DDL with a transient error (``answering.setup_with_retry``).

Only for local, dedicated instances: a build wipes the graph whenever it
starts over. ``--memgraph-url`` bypasses all of this for an instance managed
elsewhere (CI).
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workloads import Workload

#: Pinned: recall's Cypher depends on the server (3.9.0 rejected
#: ``text_search.search_all``'s limit option), and the official numbers were
#: measured on 3.13.2.
IMAGE = "memgraph/memgraph-mage:3.13.2"

MEMGRAPH_FLAGS = (
    "--storage-wal-enabled=true",
    "--storage-snapshot-interval-sec=3600",
    "--storage-snapshot-on-exit=true",
    "--data-recovery-on-startup=true",
    "--telemetry-enabled=false",
)

#: Recovering a large graph's snapshot and WAL takes a while; a build never waits on a cold start this long.
READY_TIMEOUT_SECONDS = 900.0

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _run(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, check=False)


@dataclass(frozen=True)
class Instance:
    """Where a workload's graph lives."""

    name: str
    url: str
    #: False for an instance named by ``--memgraph-url``, which eval neither starts nor removes.
    managed: bool


def volume(workload: Workload) -> str:
    return f"{workload.instance}-data"


def run_args(workload: Workload) -> list[str]:
    """The ``docker run`` command that creates ``workload``'s instance."""
    return [
        "docker",
        "run",
        "-d",
        "--name",
        workload.instance,
        # Loopback only: the instance has no auth.
        "-p",
        f"127.0.0.1:{workload.port}:7687",
        "-v",
        f"{volume(workload)}:/var/lib/memgraph",
        "--label",
        f"context-graph-eval.workload={workload.name}",
        IMAGE,
        *MEMGRAPH_FLAGS,
    ]


def state(workload: Workload, *, run: Runner = _run) -> str:
    """The container's state: "running", "exited", ..., or "absent"."""
    result = run(["docker", "inspect", "-f", "{{.State.Status}}", workload.instance])
    return result.stdout.strip() if result.returncode == 0 else "absent"


def ensure_running(workload: Workload, *, run: Runner = _run, wait: bool = True) -> Instance:
    """Start ``workload``'s instance, creating it on first use, and wait until it answers.

    Raises:
        RuntimeError: Docker refused to create or start it, or it never answered.
    """
    current = state(workload, run=run)
    if current == "absent":
        print(f"creating eval instance {workload.instance} ({IMAGE}) on port {workload.port}", flush=True)
        result = run(run_args(workload))
    elif current != "running":
        print(f"starting eval instance {workload.instance}", flush=True)
        result = run(["docker", "start", workload.instance])
    else:
        result = None
    if result is not None and result.returncode != 0:
        raise RuntimeError(f"docker could not start {workload.instance}: {result.stderr.strip()}")
    instance = Instance(name=workload.instance, url=f"bolt://localhost:{workload.port}", managed=True)
    if wait:
        wait_until_ready(instance.url, alive=lambda: _alive(workload, run=run))
    return instance


def _alive(workload: Workload, *, run: Runner) -> None:
    """Raise with the container's last log lines once it has stopped: waiting longer can't help.

    Raises:
        RuntimeError: the container is not running.
    """
    current = state(workload, run=run)
    if current != "running":
        logs = run(["docker", "logs", "--tail", "20", workload.instance])
        raise RuntimeError(f"{workload.instance} is {current}: {(logs.stdout + logs.stderr).strip()}")


def remove(workload: Workload, *, run: Runner = _run) -> None:
    """Delete ``workload``'s instance and its volume: the graph is gone for good."""
    run(["docker", "rm", "-f", workload.instance])
    run(["docker", "volume", "rm", volume(workload)])


def wait_until_ready(
    url: str, *, timeout: float = READY_TIMEOUT_SECONDS, alive: Callable[[], None] | None = None
) -> None:
    """Block until Memgraph at ``url`` answers a query.

    Args:
        alive: Called between attempts; raises once the server has died, so a
            container that exited on start fails at once instead of at ``timeout``.

    Raises:
        RuntimeError: it didn't within ``timeout`` seconds, or ``alive`` raised.
    """
    from memgraph_toolbox.api.memgraph import Memgraph

    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            db = Memgraph(url=url, username="", password="")
            try:
                db.query("RETURN 1;")
                return
            finally:
                db.close()
        except Exception as exc:
            last = exc
            if alive is not None:
                alive()
            time.sleep(2)
    raise RuntimeError(f"Memgraph at {url} did not answer within {timeout:.0f}s: {last}")
