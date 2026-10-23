"""setup's Memgraph step against real Docker, and one whole run of setup and uninstall.

Every container and volume is named per test session and removed afterwards; nothing that already runs on
the machine is stopped or reused. A second Memgraph, under a name that isn't setup's, plays someone else's
instance (a dev graph, a test instance, the AI Memory demo) holding the port setup would otherwise choose.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import uuid
from typing import TYPE_CHECKING

import pytest
from neo4j import GraphDatabase

from memgraph_context_graph.cli import main
from memgraph_context_graph.setup import memgraph as mg

if TYPE_CHECKING:
    from typing_extensions import LiteralString

pytestmark = pytest.mark.docker


def _docker_ok() -> bool:
    docker = shutil.which("docker")
    return docker is not None and subprocess.run([docker, "info"], capture_output=True).returncode == 0


if not _docker_ok():
    pytest.skip("needs a running Docker", allow_module_level=True)


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _query(url: str, cypher: LiteralString) -> list[dict]:
    with GraphDatabase.driver(url, auth=("", "")) as driver, driver.session() as session:
        return session.run(cypher).data()


@pytest.fixture(scope="module")
def names():
    """Container names for this run, and the cleanup of whatever setup made under them."""
    prefix = f"cg-setup-test-{uuid.uuid4().hex[:8]}"
    made = {
        "own": f"{prefix}-own",
        "stranger": f"{prefix}-stranger",
        "other": f"{prefix}-other",
        "lost": f"{prefix}-lost",
    }
    yield made
    for name in made.values():
        _docker("rm", "-f", name)
        _docker("volume", "rm", f"{name}-data")


@pytest.fixture(scope="module")
def stranger(names):
    """A Memgraph that isn't setup's, on a port of its own that stands in for the default one."""
    port = _free_port()
    started = _docker("run", "-d", "--name", names["stranger"], "-p", f"127.0.0.1:{port}:7687", mg.IMAGE)
    assert started.returncode == 0, started.stderr
    url = mg.local_url(port)
    mg._wait_ready(url, "", "", mg.READY_SECONDS, names["stranger"], "docker")
    return port


def test_another_memgraph_on_the_default_port_is_never_adopted(names, stranger):
    lines: list[str] = []
    found = mg.find_or_start(configured_url=None, default_port=stranger, container=names["own"], log=lines.append)

    assert found.how == "started"
    assert found.container == names["own"]
    assert found.url != mg.local_url(stranger)
    assert found.notes == (f"port {stranger} was taken, so it uses {found.url.rsplit(':', 1)[1]}",)
    assert _query(found.url, "SHOW VERSION")[0]["version"] == found.version
    inspect = json.loads(_docker("inspect", names["own"]).stdout)[0]
    assert inspect["Mounts"][0]["Name"] == f"{names['own']}-data"
    assert "--schema-info-enabled=True" in inspect["Args"]
    # A fixed host port, not one Docker reassigns on every start: the config must survive a reboot.
    assert inspect["HostConfig"]["PortBindings"]["7687/tcp"][0]["HostPort"] == found.url.rsplit(":", 1)[1]

    again = mg.find_or_start(configured_url=found.url, default_port=stranger, container=names["own"])
    assert (again.how, again.url) == ("configured", found.url)


def test_a_configured_default_url_leads_to_our_container_not_whoever_answers_there(names, stranger):
    """A config left at the default URL (an old install, or by hand) while someone else's Memgraph holds it."""
    ours = mg.find_or_start(configured_url=None, default_port=stranger, container=names["own"], log=lambda _: None)

    found = mg.find_or_start(
        configured_url=mg.local_url(stranger), default_port=stranger, container=names["own"], log=lambda _: None
    )

    assert (found.how, found.url, found.container) == ("found", ours.url, names["own"])


def test_asked_for_by_name_another_memgraph_is_used(names, stranger):
    found = mg.find_or_start(configured_url=None, explicit_url=mg.local_url(stranger), container=names["other"])
    assert (found.how, found.container) == ("found", None)
    assert _docker("inspect", names["other"]).returncode != 0


def test_an_asked_for_url_that_doesnt_answer_is_never_swapped_for_our_container(names, stranger):
    """An explicit URL whose Memgraph is down, or refuses the login, must not land the config on another port."""
    ours = mg.find_or_start(configured_url=None, default_port=stranger, container=names["own"], log=lambda _: None)
    asked = mg.local_url(_free_port())

    with pytest.raises(mg.ProvisionError, match="doesn't answer") as excinfo:
        mg.find_or_start(configured_url=None, explicit_url=asked, container=names["own"], log=lambda _: None)
    assert "No local container was changed" in " ".join(excinfo.value.hints)

    again = mg.find_or_start(configured_url=None, explicit_url=ours.url, container=names["own"], log=lambda _: None)
    assert again.url == ours.url


def test_an_asked_for_port_that_is_taken_fails_and_leaves_nothing_behind(names):
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        with pytest.raises(mg.ProvisionError, match=f"Port {port} is already taken"):
            mg.find_or_start(
                configured_url=None, explicit_url=mg.local_url(port), container=names["other"], log=lambda _: None
            )
    assert _docker("inspect", names["other"]).returncode != 0


def test_our_stopped_container_comes_back_on_its_port_with_its_data(names, stranger):
    url = mg.find_or_start(configured_url=None, default_port=stranger, container=names["own"]).url
    _query(url, "MERGE (:SetupProbe {id: 1})")
    assert _docker("stop", names["own"]).returncode == 0

    found = mg.find_or_start(configured_url=None, default_port=stranger, container=names["own"], log=lambda _: None)

    assert (found.how, found.url) == ("restarted", url)
    assert _query(url, "MATCH (n:SetupProbe) RETURN count(n) AS n") == [{"n": 1}]


def test_a_container_that_lost_its_network_is_reattached_and_published(names):
    port = _free_port()
    volume = f"{names['lost']}-data:/var/lib/memgraph"
    run = ["run", "-d", "--name", names["lost"], "-v", volume, "-p", f"127.0.0.1:{port}:7687", mg.IMAGE]
    assert _docker(*run).returncode == 0
    # Docker keeps the binding in the config but publishes nothing once the container has no network.
    assert _docker("network", "disconnect", "bridge", names["lost"]).returncode == 0
    assert _docker("restart", names["lost"]).returncode == 0
    assert _docker("port", names["lost"], "7687/tcp").stdout.strip() == ""

    found = mg.find_or_start(configured_url=None, default_port=port, container=names["lost"], log=lambda _: None)

    assert found.url == mg.local_url(port)
    assert _query(found.url, "RETURN 1 AS ok") == [{"ok": 1}]


def test_setup_then_uninstall_end_to_end(names, stranger, fake_agents, setup_home, tmp_path, capsys):
    """The plain front end, the way the shim runs it without a terminal: every step, then a re-run, then undo."""
    from memgraph_context_graph.adapters._identity import load_config

    project = tmp_path / "billing"
    project.mkdir()
    argv = [
        "setup", "claude-code", "cursor", "--no-tui", "--no-viewer", "--user-id", "alice",
        "--project-dir", str(project), "--memgraph-container", names["own"],
    ]  # fmt: skip

    main(argv)
    out = capsys.readouterr().out
    assert "OK    Memgraph: using Memgraph" in out
    assert f"(container {names['own']})" in out
    config = load_config()
    assert config.user_id == "alice"
    assert config.memgraph_url == mg.find_or_start(configured_url=None, container=names["own"]).url
    assert "OK    Claude Code: plugin installed" in out
    assert f"OK    Cursor: memory tools registered; wired {project.resolve()}" in out
    assert "OK    Check memgraph:" in out
    assert "OK    Check config:" in out
    assert fake_agents.of("claude")["plugins"] == ["context-graph@context-graph-plugins"]

    main(argv)
    out = capsys.readouterr().out
    assert "OK    Memgraph: using Memgraph" in out
    assert "OK    Claude Code: plugin already installed" in out
    assert f"OK    Cursor: memory tools already registered; rewired {project.resolve()}" in out
    assert len(json.loads((project / ".cursor" / "hooks.json").read_text())["hooks"]["sessionStart"]) == 1

    assert main(["uninstall", "--project-dir", str(project)]) == 0
    out = capsys.readouterr().out
    assert "OK    Claude Code: removed the plugin and marketplace" in out
    assert f"OK    Cursor: unwired {project.resolve()}, the memory tools" in out
    assert not (project / ".cursor" / "hooks.json").exists()
    assert _docker("inspect", names["own"]).returncode == 0  # the graph stays
    assert load_config().memgraph_url  # and so does the config
    assert "uninstall --all" in out

    # Everything, when asked for: the container and its volume (by name), and the config.
    assert main(["uninstall", "--all", "--memgraph-container", names["own"]]) == 0
    out = capsys.readouterr().out
    assert f"OK    Graph: deleted the container {names['own']} and the volume {names['own']}-data" in out
    assert "OK    Config: deleted" in out
    assert "SKIP  Tool: not running as the uv tool" in out  # a development venv is the developer's
    assert _docker("container", "inspect", names["own"]).returncode != 0
    assert _docker("volume", "inspect", f"{names['own']}-data").returncode != 0
    assert not (tmp_path / "config.toml").exists()
    assert "Kept;" not in out  # nothing left to list
