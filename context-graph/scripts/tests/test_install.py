"""Exercise the installer's Memgraph startup against real Docker.

Run with `python3 -m unittest discover -s context-graph/scripts/tests -v`.
Only the database stage runs, so no runtime plugins or user config are changed.
"""

import os
import shutil
import socket
import subprocess
import unittest
import uuid
from pathlib import Path


def docker(*args: str) -> subprocess.CompletedProcess[str]:
    """Run Docker, returning captured output and status."""
    return subprocess.run(["docker", *args], text=True, capture_output=True, check=False)


def free_port() -> int:
    """Reserve an unused local port long enough to select its number."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@unittest.skipUnless(shutil.which("docker"), "requires Docker")
class MemgraphStartupTest(unittest.TestCase):
    """Check collisions and port propagation without faking Docker's bindings."""

    def setUp(self) -> None:
        """Give every test isolated containers and a free preferred port."""
        if docker("info").returncode:
            self.skipTest("requires a running Docker daemon")
        self.container = f"context-graph-install-test-{uuid.uuid4().hex[:12]}"
        self.blocker = f"{self.container}-blocker"
        self.port = free_port()
        self.addCleanup(docker, "rm", "-f", self.container, self.blocker)
        source = (Path(__file__).resolve().parents[1] / "install.sh").read_text()
        self.script = source.split("# ---- 2. uv", 1)[0]
        self.script = self.script.replace("${MEMGRAPH_PORT:-7687}", f"${{MEMGRAPH_PORT:-{self.port}}}")
        # Enter the Docker-start path deterministically: on Docker Desktop a
        # published port can accept TCP even when nothing listens in the container.
        self.script = self.script.replace("if bolt_reachable; then", "if false; then", 1)
        self.script += '\nprintf "SELECTED_PORT=%s\\n" "$MEMGRAPH_PORT"\n'
        self.env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"MEMGRAPH_HOST", "MEMGRAPH_PORT", "SKIP_MEMGRAPH"}
        }
        self.env["MEMGRAPH_HOST"] = "127.0.0.1"
        self.env["CONTEXT_GRAPH_MEMGRAPH_CONTAINER"] = self.container

    def occupy_port(self) -> None:
        """Publish a Docker port without a listener, matching reserved-port failures."""
        result = docker("run", "-d", "--name", self.blocker, "-p", f"{self.port}:7687", "alpine:3", "sleep", "300")
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_stage(self) -> subprocess.CompletedProcess[str]:
        """Execute the real shell database stage with isolated resource names."""
        return subprocess.run(["bash"], input=self.script, env=self.env, text=True, capture_output=True, timeout=240)

    def test_default_port_collision_selects_and_reports_another_port(self) -> None:
        """A Docker-reserved default port must not abort or change its owner."""
        self.occupy_port()
        result = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        selected = docker(
            "inspect", "--format", '{{(index (index .NetworkSettings.Ports "7687/tcp") 0).HostPort}}', self.container
        ).stdout.strip()
        self.assertTrue(selected.isdigit(), result.stdout + result.stderr)
        self.assertNotEqual(selected, str(self.port))
        self.assertIn(f"SELECTED_PORT={selected}", result.stdout)
        self.assertEqual(docker("inspect", "--format", "{{.State.Running}}", self.blocker).stdout.strip(), "true")

    def test_explicit_busy_port_has_actionable_error(self) -> None:
        """An explicit port is never silently replaced by another port."""
        self.occupy_port()
        self.env["MEMGRAPH_PORT"] = str(self.port)
        result = self.run_stage()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MEMGRAPH_PORT", result.stderr)
        self.assertIn("already allocated", result.stderr)

    def test_new_container_publishes_only_bolt(self) -> None:
        """An unused monitoring port must not create a second collision."""
        result = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        bindings = docker("inspect", "--format", "{{json .HostConfig.PortBindings}}", self.container).stdout
        self.assertIn("7687/tcp", bindings)
        self.assertNotIn("7444/tcp", bindings)
