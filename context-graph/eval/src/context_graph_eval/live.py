"""Drive a real Claude Code session so the gold slice has a fixture to score.

Tier 1 injects fixtures straight into the graph, which never exercises the
capture layer. The gold slice instead runs a genuine session with hooks live, so
what gets scored is what `memgraph-context-graph` actually recorded -- the only
coverage `skills-graph` and subagent nesting get at all (#308).

Ordering matters and is the whole reason this is separate from injection:

    wipe -> inject Tier 1 -> drive live session -> reconcile -> retrieve

The live session plants ON TOP of an already-injected graph, so the gold-slice
fact sits among Tier 1's distractors. Running it before injection would simply
have the wipe destroy it.

Hooks read configuration only from a config file (ADR 0002), so the session
is handed a throwaway one through `CONTEXT_GRAPH_CONFIG` (ADR 0003) rather
than rewriting the global file every other session reads.
"""

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

#: Names the config file a hook subprocess should read (ADR 0003). Selects the
#: file, never its contents -- values still come only from the file.
CONFIG_PATH_ENV = "CONTEXT_GRAPH_CONFIG"

#: Every connector the gold slice needs recording. `hook run` silently does
#: nothing without explicit --connector flags -- a real gap `test-graph-model`
#: discovered the hard way.
CONNECTORS = ("skills-graph", "actions-graph", "sessions-graph")


class LiveSessionError(RuntimeError):
    """The session could not be driven, so nothing should be scored from it."""


@contextmanager
def hooks_pointed_at(memgraph_url: str, *, user_id: str = "context-graph-eval"):
    """Yield an environment whose hooks write to ``memgraph_url``.

    Writes a throwaway config file and points ``CONTEXT_GRAPH_CONFIG`` at it
    (ADR 0003). Nothing outside the yielded environment is modified, so a
    crash cannot leave other sessions pointed at the eval instance.
    """
    with tempfile.TemporaryDirectory(prefix="context-graph-eval-") as workdir:
        config = Path(workdir) / "config.toml"
        config.write_text(
            "\n".join(
                [
                    "[identity]",
                    # sessions-graph only writes (:User)-[:HAD_SESSION]->(:Session)
                    # when a user_id resolves; without it the session is half recorded.
                    f'user_id = "{user_id}"',
                    "",
                    "[memgraph]",
                    f'url = "{memgraph_url}"',
                    'user = ""',
                    'password = ""',
                    'database = "memgraph"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        env = {**os.environ, CONFIG_PATH_ENV: str(config)}
        # The judge's ANTHROPIC_API_KEY would override the CLI's own login and
        # make it refuse to start.
        env.pop("ANTHROPIC_API_KEY", None)
        yield env


def hooks_settings(path: Path) -> Path:
    """Write a Claude Code settings file wiring every hook to this checkout.

    Uses ``uv run`` rather than a bare binary, so a globally installed copy
    cannot shadow the checkout under test and have the session exercise the
    wrong version of the code.

    Deliberately without ``--package memgraph-context-graph``: that narrows the
    environment to exclude the connector packages, and hooks swallow the
    resulting ImportError, so a billed session would record nothing.
    """
    from memgraph_context_graph.adapters.claude_code import PLUGIN

    connectors = " ".join(f"--connector {name}" for name in CONNECTORS)
    command = f"uv run memgraph-context-graph hook run claude-code {connectors}"
    path.write_text(json.dumps({"hooks": PLUGIN.build_hooks_config(command)}), encoding="utf-8")
    return path


def drive_session(
    prompt: str,
    *,
    repo_root: Path,
    env: dict[str, str] | None = None,
    allowed_tools: str = "Task",
    timeout: int = 600,
) -> tuple[str, str]:
    """Run one real, non-interactive Claude Code session.

    Returns ``(session_id, transcript)``. The transcript is returned rather than
    discarded because a zero exit code says only that the CLI did not crash --
    not that hooks fired, nor that the model used the tool it was told to. When
    a run plants nothing, what the session actually did is the only way to tell
    those apart, and re-running to find out costs another billed session.

    ``allowed_tools`` defaults to Task alone, which is what forces genuine
    delegation: with no other tool available the model cannot do the work
    itself, so the fact lands inside a subagent rather than at top level.
    """
    if not shutil.which("claude"):
        raise LiveSessionError("the `claude` CLI is not on PATH; the gold slice needs a real session")

    session_id = str(uuid.uuid4())
    with tempfile.TemporaryDirectory() as workdir:
        settings = hooks_settings(Path(workdir) / "settings.json")
        result = subprocess.run(
            [
                "claude",
                "-p",
                prompt,
                "--settings",
                str(settings),
                "--setting-sources",
                "project",
                "--session-id",
                session_id,
                "--output-format",
                "json",
                "--permission-mode",
                "bypassPermissions",
                f"--allowedTools={allowed_tools}",
            ],
            # Hook commands embed `uv run`, which resolves against the hook
            # subprocess's cwd -- wherever `claude` was launched from.
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or {**os.environ},
        )

    if result.returncode != 0:
        raise LiveSessionError(f"claude exited {result.returncode}: {result.stderr[-600:] or result.stdout[-600:]}")
    return session_id, result.stdout
