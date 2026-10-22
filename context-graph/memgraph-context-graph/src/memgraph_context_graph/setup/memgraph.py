"""Find or start the Memgraph that hooks write to.

The order of cases is decided in #535: a Memgraph asked for by URL that answers wins; a remote one that doesn't
is an error, never a reason to touch local containers; otherwise setup uses its own container, restarted where
it is, or starts it, on another free port when the default one is taken.

A local Memgraph is ours by container name and published port, never because it answers on the default port:
whatever else listens on localhost:7687 may be a dev, test or demo instance whose data isn't ours to write
into, or which is thrown away under the hooks (#536).
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable

#: Pinned so a re-run never pulls a different server; bump it with the version CI tests against.
IMAGE = "memgraph/memgraph-mage:3.13.1"
CONTAINER = "context-graph-memgraph"
DEFAULT_PORT = 7687
READY_SECONDS = 180.0
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_PORT_TAKEN = ("port is already allocated", "address already in use")


class ProvisionError(Exception):
    """Memgraph could not be found or started; ``hints`` say what the user can do about it."""

    def __init__(self, message: str, *hints: str) -> None:
        super().__init__(message)
        self.hints = hints


@dataclass(frozen=True)
class Memgraph:
    """The Memgraph setup settled on."""

    url: str
    version: str
    #: "configured", "found", "restarted" or "started".
    how: str
    container: str | None = None
    notes: tuple[str, ...] = field(default=())

    def describe(self) -> str:
        """One line for the plan and the report."""
        where = f" (container {self.container})" if self.container else ""
        verb = {"configured": "using", "found": "using", "restarted": "restarted", "started": "started"}[self.how]
        return f"{verb} Memgraph {self.version} at {self.url}{where}"


def version_at(url: str, user: str = "", password: str = "", timeout: float = 3.0) -> str | None:
    """The Memgraph version answering at *url*, or None when nothing (or not Memgraph) answers.

    Sent auto-commit: Memgraph rejects ``SHOW VERSION`` inside the driver's managed transactions, so a probe
    through ``execute_query`` would never succeed (#535).
    """
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(
            url, auth=(user, password), connection_timeout=timeout, max_connection_lifetime=timeout
        )
    except Exception:
        return None
    try:
        with driver.session() as session:
            rows = session.run("SHOW VERSION").data()
        return str(rows[0]["version"]) if rows else None
    except Exception:
        return None
    finally:
        driver.close()


def is_local(url: str) -> bool:
    """Whether *url* points at this machine, where setup may start a container."""
    return (urlsplit(url).hostname or "") in _LOCAL_HOSTS


def local_url(port: int) -> str:
    """The Bolt URL of a Memgraph published on *port* on this machine."""
    return f"bolt://localhost:{port}"


def find_or_start(
    *,
    configured_url: str | None,
    explicit_url: str | None = None,
    user: str = "",
    password: str = "",
    container: str = CONTAINER,
    image: str = IMAGE,
    default_port: int = DEFAULT_PORT,
    ready_seconds: float = READY_SECONDS,
    log: Callable[[str], None] = print,
) -> Memgraph:
    """Settle on a Memgraph, starting one in Docker when the one asked for isn't there.

    *explicit_url* is what the user asked for on the command line; it is never swapped for another port.
    *configured_url* is what the config file holds. A remote one, or a local one on a port other than
    *default_port*, was chosen by someone and is used when it answers; a local one on *default_port* may be the
    default nobody chose, so it counts only through *container*.

    Raises:
        ProvisionError: the asked-for Memgraph doesn't answer and can't be started here, Docker is
            missing or stopped, the asked-for port is taken, or a started container never answered.
    """
    chosen = explicit_url or (configured_url if configured_url and was_chosen(configured_url, default_port) else None)
    if chosen:
        version = version_at(chosen, user, password)
        if version:
            return Memgraph(chosen, version, "found" if explicit_url else "configured")
        if not is_local(chosen):
            raise ProvisionError(
                f"Memgraph at {chosen} doesn't answer",
                "Start it, or pass --memgraph-url with one that does. No local container was changed.",
            )

    docker = _docker_or_fail()
    if (existing := _inspect(docker, container)) is not None:
        return _restart(docker, existing, container, user, password, ready_seconds, log)
    port = _port_of(explicit_url) if explicit_url else default_port
    return _start(docker, container, image, port, bool(explicit_url), user, password, ready_seconds, log)


def was_chosen(configured_url: str, default_port: int = DEFAULT_PORT) -> bool:
    """Whether a configured URL names a Memgraph someone picked, rather than the default nobody had to pick."""
    return not is_local(configured_url) or _port_of(configured_url) != default_port


def _docker_or_fail() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise ProvisionError(
            "Docker is needed to start Memgraph, and it isn't installed",
            "Install it: https://docs.docker.com/get-docker/",
            "Or point setup at a Memgraph you run yourself: --memgraph-url bolt://<host>:<port>",
        )
    if _run([docker, "info"]).returncode != 0:
        raise ProvisionError(
            "Docker is installed but its engine isn't responding",
            "Start Docker (Docker Desktop, Colima, OrbStack, ...) and run setup again.",
        )
    return docker


def _inspect(docker: str, name: str) -> dict | None:
    result = _run([docker, "inspect", name])
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)[0]


def _restart(
    docker: str, info: dict, name: str, user: str, password: str, ready_seconds: float, log: Callable[[str], None]
) -> Memgraph:
    if not info["State"]["Running"]:
        log(f"Starting the existing container {name} (its data is kept)")
        started = _run([docker, "start", name])
        if started.returncode != 0:
            raise ProvisionError(
                f"Couldn't start the existing container {name}: {_last_line(started.stderr)}",
                f"See `docker logs {name}`. Setup never removes it, because it holds your graph.",
            )
    port = _published_port(docker, name)
    if port is None and (current := _inspect(docker, name)) is not None and _lost_its_network(current):
        log(f"The container {name} lost its network, so its port isn't published; reattaching it")
        _run([docker, "network", "connect", "bridge", name])
        _run([docker, "restart", name])
        port = _published_port(docker, name)
    if port is None:
        raise ProvisionError(
            f"The container {name} doesn't publish Memgraph's Bolt port 7687",
            "Publish it, or pass --memgraph-container with another name to start a fresh one.",
        )
    url = local_url(port)
    version = _wait_ready(url, user, password, ready_seconds, name, docker)
    how = "found" if info["State"]["Running"] else "restarted"
    return Memgraph(url, version, how, container=name)


def _lost_its_network(info: dict) -> bool:
    """Whether a container asked for Bolt's port but is attached to no network, so Docker can't publish it.

    Docker leaves a container in this state after its network is removed underneath it; the binding is still
    in its config, and reattaching it (then a restart) makes Docker apply it. Its data is never touched.
    """
    bindings = info["HostConfig"].get("PortBindings") or {}
    return "7687/tcp" in bindings and not info["NetworkSettings"].get("Networks")


def _start(
    docker: str,
    name: str,
    image: str,
    port: int,
    port_was_asked_for: bool,
    user: str,
    password: str,
    ready_seconds: float,
    log: Callable[[str], None],
) -> Memgraph:
    notes: list[str] = []
    log(f"Starting Memgraph ({image}) as {name}; the first run downloads the image")
    result = _docker_run(docker, name, image, f"127.0.0.1:{port}:7687")
    if result.returncode != 0 and _port_taken(result.stderr):
        # Docker keeps the container it failed to start; remove only that attempt, never the port's owner.
        _run([docker, "rm", "-f", name])
        if port_was_asked_for:
            raise ProvisionError(
                f"Port {port} is already taken",
                "Pass --memgraph-url with a free port, or leave it out and setup picks one.",
            )
        # Not `-p 127.0.0.1::7687`: Docker assigns that port anew on every start, so after a reboot the
        # config would point at a dead port. A free port chosen here is written into the binding and kept.
        for _ in range(3):
            free = _free_port()
            log(f"Port {port} is taken; using {free}")
            result = _docker_run(docker, name, image, f"127.0.0.1:{free}:7687")
            if result.returncode == 0 or not _port_taken(result.stderr):
                break
            _run([docker, "rm", "-f", name])  # lost a race for that port too; try another
        notes.append(f"port {port} was taken, so it uses {free}")
    if result.returncode != 0:
        _run([docker, "rm", "-f", name])
        raise ProvisionError(f"`docker run` failed: {_last_line(result.stderr)}")
    published = _published_port(docker, name)
    if published is None:
        raise ProvisionError(f"Docker started {name} but didn't publish its Bolt port")
    url = local_url(published)
    version = _wait_ready(url, user, password, ready_seconds, name, docker)
    return Memgraph(url, version, "started", container=name, notes=tuple(notes))


def _docker_run(docker: str, name: str, image: str, publish: str) -> subprocess.CompletedProcess[str]:
    volume = f"{name}-data:/var/lib/memgraph"
    run = [docker, "run", "-d", "--name", name, "--restart", "unless-stopped", "-v", volume, "-p", publish]
    return _run([*run, image, "--schema-info-enabled=True"])


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(url: str, user: str, password: str, seconds: float, name: str, docker: str) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        version = version_at(url, user, password, timeout=2.0)
        if version:
            return version
        time.sleep(0.5)
    raise ProvisionError(
        f"Memgraph in {name} didn't answer at {url} within {seconds:.0f}s",
        f"See `{docker} logs {name}`; it may still be loading, so running setup again often works.",
    )


def _published_port(docker: str, name: str) -> int | None:
    result = _run([docker, "port", name, "7687/tcp"])
    for line in result.stdout.splitlines():
        host_port = line.rsplit(":", 1)[-1].strip()
        if host_port.isdigit():
            return int(host_port)
    return None


def _port_of(url: str) -> int:
    return urlsplit(url).port or DEFAULT_PORT


def _port_taken(stderr: str) -> bool:
    return any(marker in stderr for marker in _PORT_TAKEN)


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)
