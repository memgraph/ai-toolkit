"""Drive a real Claude Code session so the gold slice has a fixture to score.

Tier 1 injects fixtures straight into the graph, which never exercises the
capture layer. The gold slice instead runs a genuine session with hooks live, so
what gets scored is what `agent-context-graph` actually recorded -- the only
coverage `skills-graph` and subagent nesting get at all (#308).

Ordering matters and is the whole reason this is separate from injection:

    wipe -> inject Tier 1 -> drive live session -> reconcile -> retrieve

The live session plants ON TOP of an already-injected graph, so the gold-slice
fact sits among Tier 1's distractors. Running it before injection would simply
have the wipe destroy it.

Hooks resolve configuration only from a config file (ADR 0002), and that file's
path was a single global -- so pointing this session at the eval instance by
rewriting it would have redirected *every* Claude Code session on the machine.
That is not hypothetical: it happened here, and an unrelated session's activity
landed in the graph under test. ADR 0003 added `CONTEXT_GRAPH_CONFIG` so a
throwaway config can be handed to one session alone, which is what this module
uses.
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
    (ADR 0003), rather than rewriting the user's real one.

    That distinction is the whole point. The config path is otherwise a single
    global, so repointing it redirects **every** Claude Code session on the
    machine, not just the one being driven. Found the hard way: an unrelated
    session's activity was recorded into the eval graph mid-run, which is
    exactly the contamination a dedicated eval instance exists to prevent.

    Nothing outside the yielded environment is modified, so there is nothing to
    restore and no window in which a crash leaves the machine misconfigured.
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
        # The driven session authenticates the way the user's CLI normally
        # does. An ANTHROPIC_API_KEY inherited from the eval process takes
        # precedence over that login and makes the CLI refuse to start -- and
        # the eval process legitimately has one, because #304 runs the judge on
        # Anthropic. The judge's credential has no business steering the
        # session under test.
        env.pop("ANTHROPIC_API_KEY", None)
        yield env


def hooks_settings(path: Path) -> Path:
    """Write a Claude Code settings file wiring every hook to this checkout.

    Uses ``uv run`` rather than a bare binary, so a globally installed copy
    cannot shadow the checkout under test and have the session exercise the
    wrong version of the code.

    Deliberately **without** ``--package agent-context-graph``. That flag
    re-resolves the environment to that one package's closure, which excludes
    the connector packages -- so every hook raised
    ``ImportError: skills-graph is required`` and, because the runner swallows
    hook errors to avoid breaking the harness, exited 0 having written nothing.
    A whole billed session recorded no data and reported success. The workspace
    venv already has every package; asking for a subset is what broke it.
    """
    from agent_context_graph.adapters.claude_code import build_hooks_config

    connectors = " ".join(f"--connector {name}" for name in CONNECTORS)
    command = f"uv run agent-context-graph hook run claude-code {connectors}"
    path.write_text(json.dumps({"hooks": build_hooks_config(command)}), encoding="utf-8")
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
