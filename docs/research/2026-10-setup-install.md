# Interactive setup: launching Textual from `curl | bash`, and provisioning Memgraph from Python

Research for map #532: #534 (launch and no-TTY fallback) and #535 (Memgraph provisioning).
Measured 2026-10-13 on macOS 14 (Darwin 23.6), Docker Desktop, Textual 8.2.8, `memgraph/memgraph-mage:3.13.1`.
The scripts that produced every number here are in [`setup-install/`](setup-install/); see "Reproducing" at the end.

## Answers

| Question | Answer |
|---|---|
| Can a Textual app run from `curl … \| bash`? | **Yes, only if the shim reopens the terminal**: `exec memgraph-context-graph setup "$@" </dev/tty`. |
| What if it doesn't? | **It hangs.** Textual mounts and draws, but keys never arrive, because stdin is the rest of the script. Nothing times out. |
| How does the shim know there's a terminal? | `(exec </dev/tty) 2>/dev/null` succeeds only with a controlling terminal. It is false in CI, `ssh` without `-t`, and Docker builds. |
| How does `setup` know? | Interactive only when **both** `sys.stdin.isatty()` and `sys.stdout.isatty()`. Otherwise plain output, never the TUI. That rules out the hang even when setup is started some other way. |
| Textual in the base install or an extra? | **Base.** It adds ~6 MB (rich is already installed) and costs hooks nothing: `import memgraph_context_graph.cli` takes 38 ms and doesn't load Textual. Textual is imported only when the TUI starts (~100 ms warm). |
| Docker SDK or CLI? | **The `docker` CLI via subprocess.** It needs no new dependency and works wherever `docker` works (Desktop, Colima, OrbStack, Podman's `docker` shim). |
| What does a busy port look like? | `docker run` exits 125 with `Bind for 0.0.0.0:7687 failed: port is already allocated` (`address already in use` when a non-Docker process holds it). **Docker keeps the failed container**, so it has to be removed before retrying under the same name. |
| How do you get a free port? | ~~`-p 127.0.0.1::7687` (empty host port), then `docker port`~~ **Corrected while implementing:** Docker reassigns that port on every container start, so after a restart or reboot the config points at a dead port. Pick a free port in Python (bind port 0) and publish it explicitly; the binding is kept across restarts. setup's Docker test asserts the port survives a stop/start. |
| How long until it answers? | **12.9 s** from `docker run` to a real query on a cached MAGE image. A TCP accept comes much earlier; the Bolt handshake fails until Memgraph is ready. |
| How do you know it's Memgraph? | `SHOW VERSION` returns `{'version': '3.13.1'}`. **It has to be sent auto-commit** (`session.run`). Through the driver's managed transactions (`execute_query`) Memgraph rejects it with a `ClientError`, so a readiness loop written that way never succeeds. |
| Which image? | Pin `memgraph/memgraph-mage:3.13.1`: the newest release, what `latest` resolves to today, and what CI tests. The demo pins 3.12.0; `install.sh` used `latest`. |

## Launch: the three cases

`setup-install/launch.py` runs a Textual app behind a two-line shim, the way an installer piped into bash would:

```
--- pty, shim reopens /dev/tty: exit 0
MOUNTED stdin_tty=True stdout_tty=True size=Size(width=80, height=25)
--- pty, stdin left as the pipe: exit 0
MOUNTED stdin_tty=False stdout_tty=True size=Size(width=80, height=25)
--- no terminal, shim probes /dev/tty: exit 0
NO TTY: plain mode
```

`setup-install/keys.py` then types `q` into a real pty while the app runs:

```
--- shim reopens /dev/tty: MOUNTED stdin_tty=True stdout_tty=True size=Size(width=80, height=25)
GOT KEY q
--- stdin left as the pipe: MOUNTED stdin_tty=False stdout_tty=True size=Size(width=80, height=25)  -> key never arrived, killed
```

The middle case is the trap. The screen looks right, so a user would sit at an installer that can't hear them.

## Provisioning: what setup does, in order

1. **The configured URL answers `SHOW VERSION`**: use it. That covers re-runs and a Memgraph the user runs themselves.
2. **A configured remote URL doesn't answer**: fail that step and touch no local container.
3. **Our container exists** (`context-graph-memgraph`): `docker start` it and read its published port with `docker port`, rather than failing when the port differs from the config.
4. **Something else answers on `localhost:7687`**:
   - a container labelled disposable (the AI Memory demo, see #536): don't adopt it; start our own;
   - otherwise: use it, as `install.sh` did.
5. **Nothing**: `docker run -d --name context-graph-memgraph -v context-graph-memgraph-data:/var/lib/memgraph -p 127.0.0.1:7687:7687 memgraph/memgraph-mage:3.13.1 --schema-info-enabled=True`.
   - On a busy port: remove the failed container and retry on a free port chosen in Python, published explicitly (not `127.0.0.1::7687`, which changes on every start).
   - An explicitly requested port that's busy fails with instructions.
   - Never stop what owns the port.
   - Then poll an auto-commit `SHOW VERSION` for up to 180 s.
6. **Write `memgraph.url`** to the config file with the port that actually answered.

A **named volume** is new. `install.sh` kept data only inside the container, so `docker rm` lost the graph.

## Found on the way

**The AI Memory demo's container is holding `localhost:7687` on this machine.** The real `context-graph-memgraph` container exited, so the developer's own Claude Code capture (config `memgraph.url = bolt://localhost:7687`) has been writing into the disposable demo database since 2026-10-12. That's the teardown-order problem in #536, live: `ai-memory.sh clean` would silently drop the capture target. It's why step 4 refuses a disposable container.

## Reproducing

From a checkout with a venv that has `textual` and `neo4j` (e.g. `uv pip install -e context-graph/viewer`):

```bash
python docs/research/setup-install/launch.py      # macOS/Linux; uses `script` and a pty
python docs/research/setup-install/keys.py
python docs/research/setup-install/docker_probe.py  # needs Docker; occupies and frees a random port
```

`docker_probe.py` assumes something already holds host port 7687, to show the conflict; it prints `busy port exit 0` otherwise. It removes its own `cg-setup-probe` container and never touches others.
