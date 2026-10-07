# Capture-failure log: which paths write to it, what a line holds, where it lives

Research findings for [#378](https://github.com/memgraph/ai-toolkit/issues/378) (re-scoped by [#381](https://github.com/memgraph/ai-toolkit/issues/381)), part of [Map: Context-graph visualisation & observability](https://github.com/memgraph/ai-toolkit/issues/374).

Investigated 2026-10-06 against `main` at `a19bdf0`. Repo claims cite `path:line` at that commit. External claims cite primary docs or source; Codex source is pinned to [`openai/codex@588f616e`](https://github.com/openai/codex/tree/588f616e8b6aee417d4b2f480e48b7e6d8741993). Results I **measured locally** are marked *(measured)* and described in [Reproduction](#reproduction). Anything I could not verify is marked **unverified**.

## Question

Settled by #381 (not reopened here): capture health in the viewer comes from **a local capture-failure log** (every capture failure, never synced into the graph, bounded) plus **the graph** (only what came in). Hooks must never block or fail the host session, and hook subprocesses take config values only from the config file, with `CONTEXT_GRAPH_CONFIG` choosing which file (ADR 0003).

This note answers what's left: which failure paths must write to the log, what one line holds, where the file lives, how it stays bounded under concurrent hook processes, what it costs a hook, and how the viewer and `doctor` read it.

## TL;DR recommendation

| Decision | Recommendation | Main evidence |
|---|---|---|
| **Format** | JSONL, one object per failure, versioned (`"v":1`), **≤ 4 KiB per line**. Identifiers and a redacted, truncated error message only. Never payloads, tool input/output, prompts, or credentials. | The graph and the transcript already hold content. The log only needs to say *what broke, where, and for which session*. See [Q3](#q3-log-shape). |
| **Location** | Default config: `~/.local/state/context-graph/capture-failures.jsonl` (hard-coded under `Path.home()`, `XDG_STATE_HOME` **not** read). With `CONTEXT_GRAPH_CONFIG` set: a sibling of that file, `<config-stem>.capture-failures.jsonl`. | Logs are state per the XDG spec. The config dir already ignores `XDG_*` (`_identity.py:46`), and reading an ambient env var at hook time is the drift ADR 0002 removed. A sibling path gives each config its own log, so tests, eval and `live-hooks-e2e` (all of which set `CONTEXT_GRAPH_CONFIG` to a temp file) never touch the user's log. |
| **Append** | One `os.open(O_WRONLY\|O_APPEND\|O_CREAT\|O_CLOEXEC\|O_NONBLOCK, 0o600)` + one `os.write` of the whole line + `close`. No lock, no `logging` module, no buffered text file. | POSIX positions every `O_APPEND` write at end-of-file with no intervening modification. 40 concurrent writers × 4 KiB lines gave 0 torn lines *(measured)*. The Python logging cookbook says multi-process logging to one file "is *not* supported". |
| **Bound** | Size check on the open fd. At ≥ 1 MiB, `os.replace(log, log + ".1")` and keep exactly one old file, so ≤ ~2 MiB total. Races lose old lines, never corrupt new ones. | No daemon needed. Measured under a 32 MiB burst: 2 files, 0 corrupt lines *(measured)*. |
| **Never raise, never block** | The whole writer is inside `try/except Exception: pass`. `O_NONBLOCK` stops a FIFO at the path from hanging the open. | ~35 µs per append, against ~130 ms for a whole failing hook run *(measured)*. Returns quietly when the dir is unwritable *(measured)*. |
| **Must-fix alongside** | (1) Lower the Memgraph driver's `connection_timeout` in hooks (default **30 s** = the hook timeout). (2) Keep argparse/unknown-runtime errors from exiting **2**. (3) Isolate connectors in `AgentLink.emit`. | A blackholed Memgraph host made one hook run take **30.18 s** *(measured)*, so the runtime kills it before anything is logged. Exit 2 is a *blocking* error on Claude Code `PreToolUse`. Today one connector's exception stops every later connector and event. |

## Q1. Failure paths inventory

### How one hook invocation flows today

`agent-context-graph hook run <runtime> …` → `hooks/cli.py:_run_runtime` (`hooks/cli.py:56-66`) → `runner.run_hook` (`hooks/runner.py:103-167`):

1. `argparse` parses flags (`runner.py:111-137`). **Outside any try.**
2. `resolve_memgraph_env` → `load_config` → `_read_config_file` (`runner.py:143-150`, `adapters/_identity.py:100-112,316-344`). **Outside the try.**
3. Inside `try` (`runner.py:153-160`): `load_payload` (stdin JSON), `create_link` (constructs each connector's graph class, and each `Memgraph()` calls `verify_connectivity()`, `memgraph-toolbox/src/memgraph_toolbox/api/memgraph.py:124-141`), `adapter.handle_payload` (`adapters/_spec.py:168-185`), then `_print_response`.
4. `except Exception` (`runner.py:161-166`): if `--strict` or `AGENT_CONTEXT_GRAPH_<RUNTIME>_STRICT=1`, re-raise. Otherwise print the normal response and call `_debug_log`, which writes to stderr **only** when `AGENT_CONTEXT_GRAPH_<RUNTIME>_DEBUG=1` (`runner.py:98-100`). Return 0.

So with the shipped plugin configs (no `--strict`; `context-graph/plugins/agent-context-graph-claude/hooks/hooks.json`, `…-codex/hooks/hooks.json`), every caught failure is **completely silent**: exit 0, nothing on stderr. *(Measured: against an unreachable Memgraph the installed 0.2.0 CLI printed `{"continue": true}`, wrote 0 bytes to stderr, and exited 0.)*

### Strict mode

`--strict` (help: "Return a non-zero status if the hook payload cannot be recorded", `runner.py:132-136`) re-raises (`runner.py:163-164`). The interpreter prints a traceback to stderr and exits **1**. In Claude Code that's a non-blocking error whose transcript notice shows only the *first* stderr line ([Q2](#q2-what-the-runtimes-do-with-hook-stderr-and-exit-codes)). For a Python traceback that line is `Traceback (most recent call last):`, which tells the user nothing. Strict mode is a developer tool. It should still write the log line **before** re-raising, so the log is complete in both modes.

### Failure-path table

"Stage" is the proposed value of the log's `stage` field ([Q3](#q3-log-shape)).

| # | Location | Failure kind | Current handling | Should it log? (stage) |
|---|---|---|---|---|
| 1 | `hooks/runner.py:137` `parser.parse_args` | Bad/unknown flag in a hooks.json (e.g. stale `--connector` spelling, removed option) | argparse prints usage to stderr and **exits 2** ([argparse docs](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.error)). Outside the try. On Claude Code `PreToolUse`, exit 2 **blocks the tool call** | **Yes** (`args`). Also **fix**: override `ArgumentParser.error`/`exit_on_error=False` in the hook path so it can never exit 2 |
| 2 | `hooks/cli.py:60-64`, `:42-44` | Unknown runtime name / missing runtime arg | Message to stderr, **return 2**. Same blocking hazard as #1 | **Yes** (`args`). Same fix: return 0 (or 1) from `hook run` |
| 3 | `hooks/runtime_plugin.py:66-67` `entry_point.load()` | A registered runtime plugin fails to import (broken install, version skew) | Not caught. Traceback, exit 1 → Claude Code "hook error" notice, Codex "Hook failed" | **Yes** (`args`). Catch in `_run_runtime` |
| 4 | `adapters/_identity.py:318-324` `_read_config_file` | Malformed config file | `except Exception: return HookConfig()`, silently uses defaults (`bolt://localhost:7687`, no user). Captures then go to the wrong place or arrive unattributed, with no signal | **Yes**, as a warning (`config`). This is the most misleading silent path |
| 5 | `hooks/runner.py:143-150` | Config read raising (e.g. permission error on `is_file`/`read_text`) | Outside the try → traceback, exit 1 | **Yes** (`config`). Move inside the try |
| 6 | `hooks/runner.py:34-37` `load_payload` | Non-JSON or non-object stdin | `json.JSONDecodeError`/`TypeError` → caught at `runner.py:161`, silent | **Yes** (`payload`). Record `error_type` + length, never the bytes |
| 7 | `hooks/runner.py:59-60`, `:195-233` `create_link` | Unsupported connector name; connector package not installed (`ImportError`, `runner.py:199-201,212-214,227-229`) | Caught at `runner.py:161`, silent. **No connector runs** | **Yes** (`setup`), with `connector` set |
| 8 | `memgraph_toolbox/api/memgraph.py:134-141` (via `runner.py:204,219,232`) | Memgraph unreachable (`ServiceUnavailable` → `ValueError`) or auth failure (`AuthError` → `ValueError`) | Caught at `runner.py:161`, silent. First connector's failure aborts the hook | **Yes** (`connect`). Keep `cause_type` (`neo4j.exceptions.AuthError`) because the `ValueError` wrapper hides it |
| 9 | Same, blackholed host | TCP connect hangs for the driver's `connection_timeout` (**30 s default**, [neo4j driver API](https://neo4j.com/docs/api/python-driver/current/api.html)) | Hook takes 30.18 s *(measured)*, which equals the plugin's `"timeout": 30`. The runtime kills it (Codex sends `SIGKILL` to the process group, [`command_runner.rs:331-339`](https://github.com/openai/codex/blob/588f616e8b6aee417d4b2f480e48b7e6d8741993/codex-rs/hooks/src/engine/command_runner.rs#L331-L339) + [`process_group.rs:269-283`](https://github.com/openai/codex/blob/588f616e8b6aee417d4b2f480e48b7e6d8741993/codex-rs/utils/pty/src/process_group.rs#L269-L283)), so **nothing can be logged** | **Yes** (`connect`), but only reachable if the hook's driver gets a short `connection_timeout` (e.g. 3–5 s) via `driver_config` (`memgraph.py:94,121-124`). See [Open items](#open-items) |
| 10 | `adapters/_spec.py:181` `rule(context)` | Adapter bug or unexpected payload shape (`KeyError`, `TypeError`, dataclass validation) | Caught at `runner.py:161`, silent | **Yes** (`translate`), with `where` (innermost frame) so adapter bugs are diagnosable |
| 11 | `adapters/_spec.py:170-172` | Unknown event name, or foreign payload (e.g. Grok reading `.claude/settings.json`) | Returns `[]`. Intentional, not a failure | **No**. It's a normal path; logging it would flood the log |
| 12 | `adapters/_spec.py:187-191` | Payload has no session id → `session_id=""` | Proceeds and writes under the empty id | **Yes**, as a warning (`translate`, `error_type: "MissingSessionId"`). Arguably also skip the write; out of scope here |
| 13 | `link.py:46-50` `AgentLink.emit` | Any connector's `on_event` raises (Cypher error, constraint violation, lost connection mid-hook) | **No isolation**: the exception propagates to `runner.py:161`, so later connectors **and later events from the same payload** (e.g. `turn_end` emits Message + TurnEnd, `_spec.py:397-419`) are dropped silently | **Yes** (`emit`), one line per failing connector. **Fix**: catch per connector in `emit`, report through a callback, continue, and re-raise an aggregate only in strict mode |
| 14 | `actions-graph/src/actions_graph/connector.py:229-240` | Unknown `ActionStatus`/`MessageRole` → coerced to `COMPLETED`/`USER` | Silent normalisation | **No**. Data normalisation, not a capture failure |
| 15 | `skills-graph/src/skills_graph/connector.py:262-264, 296-298, 316-318` | Unparsable JSON result, `shlex` failure, unreadable `SKILL.md` | Fallback values | **No**. Same reason as #14 |
| 16 | `sessions-graph/src/sessions_graph/connector.py:142-156, 170-174` | `MERGE` Session/User or `SET reconciliation_status='pending'` fails | Propagates → #13 | **Yes** (`emit`, `connector: sessions-graph`) |
| 17 | `sessions-graph/src/sessions_graph/connector.py:182-192` `_spawn_detached` | `Popen` of `sessions-graph embed`/`reconcile` fails (`OSError`, e.g. executable missing) | `logger.warning(...)`. No logging config exists in the hook process, so it goes to `logging.lastResort` → **stderr** ([logging docs](https://docs.python.org/3/library/logging.html#logging.lastResort)), which on exit 0 Claude Code keeps in its debug log only | **Yes** (`spawn`). Replace the warning with the failure-log writer |
| 18 | `sessions-graph/src/sessions_graph/cli.py:112-119, 156-163` (detached child) | The child can't reach Memgraph, or fails to import LightRAG, **before** it can record anything on the Session | Child stderr is `DEVNULL` (`connector.py:185-187`), so it is invisible anywhere | **Yes** (`embed`/`reconcile`). The child inherits `CONTEXT_GRAPH_CONFIG` through `_child_env`'s copy of `os.environ` (`connector.py:213`), so it resolves the same log. See [Open items](#open-items) |
| 19 | `sessions-graph/src/sessions_graph/core.py:388-395, 561-562, 781-785, 813-820` | Reconciliation/embedding fails **after** the Session is reachable | Written to the graph: `reconciliation_status='failed'` + `reconciliation_error`, `embedding_status='failed'` + `embedding_error` | **No**. #381 rule: memory state stays on the Session. Only #18 (nothing to attach to) goes to the log |
| 20 | `hooks/runner.py:186-189` `session_start_context` | Memory-tool hints fail to load on `SessionStart` | `except Exception: return {}`, silent. The model isn't told recall exists | **Yes**, as a warning (`respond`) |
| 21 | `hooks/runner.py:165` `_print_response` inside the `except` | Building the response raises | Escapes the handler → traceback, exit 1 | Covered by an outer guard. The writer call must come **before** `_print_response` in the handler |
| 22 | `agent-context-graph` not on `PATH` / Python won't start | The command never runs | Shell exit 127 → runtime error notice | **Can't**: no process of ours runs. Only the runtime shows it (Claude Code notice; Codex "Hook failed") |
| 23 | `cli.py:640-651` `doctor` → `_check_runtime` | Probe failure | Reported in doctor output | **No**. `doctor` reports its own result directly; it *reads* the log ([Q7](#q7-how-the-viewer-and-doctor-read-it)) |

### Where the writes should go

The fewest call sites that cover the table:

- **`run_hook` outer guard.** One `record_failure(stage, exc, …)` in the existing `except` (`runner.py:161`), written before strict re-raise or `_print_response`. Wrap the steps outside the try today (argparse, config read) in the same guard. Covers #1, 4–8, 10, 12, 20, 21.
- **`AgentLink.emit` per connector** (#13, #16), so the line names the failing connector. `link.py` stays I/O-free if `emit` takes an `on_error(connector, event, exc)` callback that the runner wires to the writer.
- **`hooks/cli.py:_run_runtime`** for #2–3.
- **`sessions_graph.connector._spawn_detached`** (#17) and the **`sessions-graph` CLI entry** (#18), importing the writer from `agent_context_graph` (sessions-graph's connector already depends on it, `connector.py:42-43`).

The writer belongs in `agent-context-graph` (e.g. `agent_context_graph/capture_log.py`, next to `_identity.config_file()`, which it derives its path from).

## Q2. What the runtimes do with hook stderr and exit codes

### Claude Code ([hooks reference](https://code.claude.com/docs/en/hooks), [hooks guide](https://code.claude.com/docs/en/hooks-guide))

- **Exit 0:** "Stderr from a hook that exits 0 goes to the debug log only, never the transcript, and Claude never sees it." This is our only exit code in non-strict mode, so **nothing surfaces to the user today**.
- **Exit 2:** "Exit 2 means a blocking error. On events that can block, exit 2 blocks whether or not you print JSON." On `PreToolUse` that **blocks the tool call** and shows Claude the stderr. Paths #1–2 above can exit 2.
- **Other non-zero:** "a non-blocking error for most hook events: the action proceeds, and the transcript shows a `<hook name> hook error` notice followed by the first line of stderr, prefixed with `Failed with non-blocking status code:`." Full stderr needs debug logging.
- **Seeing stderr:** "For full execution details including which hooks matched, their exit codes, stdout, and stderr, read the debug log. Start Claude Code with `claude --debug-file /tmp/claude.log` … If you started without that flag, run `/debug` mid-session" (guide, *Debug techniques*). `Ctrl+O` opens the transcript view, which shows hook error notices.
- **Timeout:** Claude Code "cancels a `command` … hook that reaches its `timeout`, discarding the hook's output", and "A timed-out … hook doesn't block the tool call." Which signal it sends is **unverified**. Our plugin sets `"timeout": 30` per hook (`hooks.json`).
- **Concurrency:** "All matching hooks run in parallel."
- **`systemMessage`:** "Warning message shown to the user." This is the channel for the parked in-session alert (map #374, *Not yet specified*).

So today a user sees nothing unless they set **both** `AGENT_CONTEXT_GRAPH_CLAUDE_CODE_DEBUG=1` in the environment Claude Code was launched from **and** run `claude --debug`/`/debug`. Even then the debug line holds only `str(exc)`, with no event or connector.

### Codex ([hooks docs](https://learn.chatgpt.com/docs/hooks), source at `588f616e`)

- **Exit 0:** stdout is parsed. On this path the `PreToolUse` result parser never reads stderr ([`pre_tool_use.rs:213-260`](https://github.com/openai/codex/blob/588f616e8b6aee417d4b2f480e48b7e6d8741993/codex-rs/hooks/src/events/pre_tool_use.rs#L213-L260)). The TUI hides quiet successes: "A hook that starts and finishes successfully without user-facing output should not leave a transcript artifact" ([`hook_cell.rs:1-12`](https://github.com/openai/codex/blob/588f616e8b6aee417d4b2f480e48b7e6d8741993/codex-rs/tui/src/history_cell/hook_cell.rs#L1-L12)).
- **Other non-zero:** status `Failed` with the entry text `hook exited with code {exit_code}`. **The stderr text is not included** ([`pre_tool_use.rs:278-284`](https://github.com/openai/codex/blob/588f616e8b6aee417d4b2f480e48b7e6d8741993/codex-rs/hooks/src/events/pre_tool_use.rs#L278-L284)). Non-success runs persist in history as "Hook failed" ([`hook_cell.rs:296-300, 427-433`](https://github.com/openai/codex/blob/588f616e8b6aee417d4b2f480e48b7e6d8741993/codex-rs/tui/src/history_cell/hook_cell.rs#L296-L300)).
- **Exit 2:** blocks with stderr as the reason (docs; `pre_tool_use.rs:261-277`).
- **Timeout:** `hook timed out after {n}s` ([`command_runner.rs:276-320`](https://github.com/openai/codex/blob/588f616e8b6aee417d4b2f480e48b7e6d8741993/codex-rs/hooks/src/engine/command_runner.rs#L276-L320)), then `SIGKILL` to the process group (row #9). Detached helpers survive *successful* hooks: "Successfully completed hooks may intentionally leave detached helpers running" (`command_runner.rs:281`). Our `start_new_session=True` children (`sessions-graph/.../connector.py:188`) are in their own group either way.
- **Concurrency:** "Multiple matching command hooks for the same event are launched concurrently."
- **Debug log for hook stderr:** none documented. **Unverified** whether `RUST_LOG`/tracing captures it. `command_runner.rs` only instruments outcome fields (`:199-211`).

**Net:** neither runtime shows a non-strict failure of ours anywhere a user would look, and Codex never shows our stderr text even on failure. A local log is the only reliable channel. That matches #381.

## Q3. Log shape

One JSON object per line, UTF-8 with `ensure_ascii=True` (keeps lines byte-stable and trivially splittable), compact separators, ≤ 4096 bytes including `\n`.

| Field | Type | Why it's there | Why it's safe |
|---|---|---|---|
| `v` | int | Schema version, so the viewer can evolve without guessing | constant |
| `at` | string, UTC ISO-8601 ms, `Z` | Ordering, "since" filters, "last failure N min ago". UTC so viewer and hook agree regardless of TZ env | timestamp |
| `runtime` | string | `plugin.name` (`claude-code`, `codex`, …). Health per runtime | registry name |
| `hook_event` | string ≤ 64 | `hook_event_name` from the payload (e.g. `PostToolUse`). Shows *which* events fail | runtime enum value |
| `event_type` | string \| null | Event Protocol type being emitted (`tool_end`) when the failure was in `emit`. Distinguishes "payload broke" from "write broke" | enum |
| `stage` | enum | `args`, `config`, `payload`, `setup`, `connect`, `translate`, `emit`, `respond`, `spawn`, `embed`, `reconcile`. **The main grouping key in the viewer**: it maps directly onto the table in Q1 | enum |
| `connector` | string \| null | `actions-graph` / `skills-graph` / `sessions-graph`, or null before connectors exist | registry name |
| `session_id` | string \| null | Joins a failure to the graph's `(:Session {session_id})`, so the viewer can mark "this session has failures" and show "session started, no actions" next to its cause | opaque id the graph already stores |
| `tool_name` | string ≤ 128 \| null | Failures that only hit one tool (huge `Read` results, an MCP tool's odd shape) show up as a pattern | name only, never input/output |
| `cwd` | string \| null | Lets the viewer's project filter (#381: `Session.working_directory`) apply to failures too, including failures with no Session in the graph | a path the graph already stores as `working_directory` |
| `error_type` | string | Qualified class, e.g. `ValueError`, `neo4j.exceptions.ClientError` | class name |
| `cause_type` | string \| null | `type(exc.__cause__)`. Needed because memgraph-toolbox wraps `ServiceUnavailable`/`AuthError` in `ValueError` (`memgraph.py:136-141`) | class name |
| `error` | string ≤ 512 | `str(exc)`, after **redaction**: the configured Memgraph password and LLM keys are replaced with `***` by literal match (they're in hand from `load_config()`), then truncated | see note below |
| `where` | string \| null | Innermost traceback frame as `module/file.py:line in func`, relative to `site-packages`. Makes adapter bugs (#10) actionable without a full traceback | code location only; no locals |
| `memgraph` | string \| null | `host:port/database` from `urlsplit(url)`, with userinfo dropped. Shows a hook pointed at the wrong instance (row #4) | no username, no password |
| `versions` | object | `agent-context-graph` plus the failing connector's package, via `importlib.metadata.version` (only computed on failure). Version skew between plugin and connectors is a real failure mode (row #3) | public version strings |
| `pid` | int | Correlates lines from one invocation (several connectors failing in one emit) | process id |
| `elapsed_ms` | int | Time from hook start to failure. Values near 30 000 point at timeouts and slow connects | number |
| `level` | `"error"` \| `"warning"` | Warnings are degraded-but-captured cases (#4, #12, #20). The viewer can show them separately from hard failures | enum |

**Deliberately excluded:** raw payload or any of its bytes, `tool_input`, tool results, prompts, assistant text (all can contain secrets and are the graph's job anyway); `transcript_path`; Memgraph username (it appears in the `AuthError` message `memgraph.py:139-141`, so redaction covers the user too if it's a concern, see Open items); full tracebacks; environment variables; the config file path (implied by the log's own location).

**On `error`:** Memgraph error messages can echo values (constraint violations, type errors). Our queries are parameterised, so parameter values *usually* stay out of messages, but that isn't guaranteed. Redacting known secrets and truncating to 512 chars caps the exposure. The log is also `0600` in a `0700` directory, the same trust level as the config file, which holds the plaintext password (`_identity.py:244-246`).

### Example line

```json
{"v":1,"at":"2026-10-06T12:41:07.512Z","level":"error","runtime":"claude-code","hook_event":"PostToolUse","event_type":"tool_end","stage":"emit","connector":"actions-graph","session_id":"3f1c9a2e-7b0d-4c51-9a8e-2f6d1e0b9c44","tool_name":"Bash","cwd":"/Users/ante/repos/ai-toolkit","error_type":"neo4j.exceptions.ClientError","cause_type":null,"error":"Unable to commit due to existence constraint violation on :Action(action_id)","where":"actions_graph/core.py:212 in record_action","memgraph":"localhost:7687/memgraph","versions":{"agent-context-graph":"0.3.1","actions-graph":"0.3.0"},"pid":48213,"elapsed_ms":184}
```

## Q4. Location and config interaction

**Options considered:**

| Option | For | Against |
|---|---|---|
| A. Always next to the config file | One rule; follows `CONTEXT_GRAPH_CONFIG` for free | Puts state in a config dir (XDG says logs are state). People version `~/.config` in dotfile repos, so a log would land in them |
| B. Platform dirs (`platformdirs.user_log_dir`) | Native per-OS | New dependency. Differs by OS (`~/Library/Logs/context-graph` on macOS, `~/.local/state/context-graph/log` on Linux *(measured, platformdirs 4.12.3)*, `%LOCALAPPDATA%\…` on Windows) while the config path is the same `~/.config/context-graph` everywhere (`_identity.py:46`). Honours `XDG_STATE_HOME`, which is an ambient env read |
| **C. Default under `~/.local/state/context-graph/`; under an override, beside the override file** | State in the XDG state location ("actions history (logs, history, recently used files, …)", [XDG Base Directory spec 0.8](https://specifications.freedesktop.org/basedir/latest/)); a separate log per config | Two rules. Small cost |

**Recommend C**, with these exact rules:

```text
CONTEXT_GRAPH_CONFIG unset  -> Path.home()/".local/state/context-graph/capture-failures.jsonl"
CONTEXT_GRAPH_CONFIG=<p>    -> <p>.parent / f"{<p>.stem}.capture-failures.jsonl"
```

- **Don't read `XDG_STATE_HOME`.** The spec's default for an unset variable is `$HOME/.local/state`. Reading the variable means the hook (spawned by a harness, perhaps launched from a GUI or IDE without the shell profile) and the viewer (an interactive shell) could resolve different files. That's exactly the drift ADR 0002 removed for config values. The config dir already ignores `XDG_CONFIG_HOME` (`_identity.py:46`), so this is consistent. `CONTEXT_GRAPH_CONFIG` stays the one explicit, parent-to-child selector (ADR 0003), and the log follows it.
- **Per-config isolation.** Each consumer that sets the override gets its own log with no extra wiring: eval's `hooks_pointed_at` writes `config.toml` in a `TemporaryDirectory` (`context-graph/eval/src/context_graph_eval/live.py:64-65`), `live-hooks-e2e.sh` uses `$WORK/config.toml` (`context-graph/scripts/live-hooks-e2e/live-hooks-e2e.sh:84`), and package tests monkeypatch it to `tmp_path` (e.g. `context-graph/agent-context-graph/tests/test_identity.py:23`, `sessions-graph/tests/conftest.py:32`). Their failures land in temp dirs and disappear with them, and never pollute the user's log.
- **Viewer and `doctor` resolve the path through the same function** (`capture_log.path()` built on `_identity.config_file()`), so "doctor and the hooks it inspects agree by passing the same variable" (ADR 0003) carries over to the log.
- **Edge:** if `CONTEXT_GRAPH_CONFIG` names the default path explicitly, use the default log, to avoid two logs for one config.
- **Platforms:** `Path.home()` gives `~/.local/state/context-graph/` on macOS and Linux, and `C:\Users\<u>\.local\state\context-graph\` on Windows. That mirrors how `~/.config/context-graph` already lands on Windows. Directory created `0o700` on first write (mode ignored on Windows), file `0o600`.

## Q5. Bounding and concurrency

**Concurrency facts:**

- POSIX `write()`: "If the O_APPEND flag of the file status flags is set, the file offset shall be set to the end of the file prior to each write and no intervening file modification operation shall occur between changing the file offset and the write operation" ([POSIX.1-2024 write](https://pubs.opengroup.org/onlinepubs/9799919799/functions/write.html)). Linux `open(2)`: "The modification of the file offset and the write operation are performed as a single atomic step" ([open(2)](https://man7.org/linux/man-pages/man2/open.2.html)).
- **`PIPE_BUF` does not apply.** POSIX's "shall not be interleaved" guarantee for writes ≤ `PIPE_BUF` covers **pipes and FIFOs** only. Regular files have no stated size threshold. Two concurrent `O_APPEND` writes can't overwrite each other, because each starts at the then-current end of file. In practice a small single `write()` to a local filesystem goes out in one piece. The 4 KiB line cap keeps us in that regime and bounds growth per event. It is a convention, not a spec guarantee.
- **Partial writes** are allowed ("only as many bytes as there is room for shall be written", POSIX; "may be less than count if … insufficient space", [write(2)](https://man7.org/linux/man-pages/man2/write.2.html)). A short write only happens on a full disk, and leaves at most one torn last line. **The reader must skip lines that don't parse** rather than fail.
- **NFS:** "O_APPEND may lead to corrupted files on NFS filesystems if more than one process appends data to a file at once" ([open(2)](https://man7.org/linux/man-pages/man2/open.2.html)). A home directory on NFS can get torn lines, and the reader's skip rule covers it. The config file is read from the same home, so the log adds no new hang exposure.
- **Why not `logging`/`RotatingFileHandler`:** "logging to a single file from *multiple processes* is *not* supported, because there is no standard way to serialize access to a single file across multiple processes in Python" ([logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes)). Its rotation renames under an in-process lock only. Buffered text files can also split one line across several `write()` calls. So use `os.open` + one `os.write(fd, line_bytes)`.

**Rotation without a daemon or lock:**

```text
fd = os.open(path, O_WRONLY|O_APPEND|O_CREAT|O_CLOEXEC|O_NONBLOCK [|O_BINARY on Windows], 0o600)
if os.fstat(fd).st_size >= 1 MiB:
    try: os.replace(path, path + ".1")   # atomic rename; old ".1" dropped
    except OSError: pass                 # e.g. Windows sharing violation: skip rotation this time
os.write(fd, line)                       # still lands in the file we opened (now ".1"), whole
os.close(fd)
```

- **Races are lossy, never corrupting.** If two processes rotate at once, the second rename can move a nearly-empty fresh file over `.1`, dropping the just-rotated megabyte. That's acceptable for a bounded diagnostic log. The total stays ≤ ~2 MiB + one line per concurrent writer. *Measured:* 40 processes × 200 lines × ~3.9 KiB (≈ 32 MiB through a 1 MiB cap) left `log.jsonl` + `log.jsonl.1` with **0 corrupt lines**. Below the cap, 40 × 6 lines all landed (240/240) with 0 corrupt.
- **Size:** at ~400–600 B per typical line, 1 MiB holds ~2,000 failures per file. When Memgraph is down, each hook writes one `connect` line (the first connector's constructor fails, `runner.py:47-60`), so an hour of heavy tool use (~600 tool calls → ~1,200 Pre/Post hooks) fits. The viewer aggregates repeats ([Q7](#q7-how-the-viewer-and-doctor-read-it)), so no write-side dedup is needed in v1.
- **Windows:** the CRT's `_O_APPEND` "Moves the file pointer to the end of the file before every write operation" ([MS `_open`](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/open-wopen)). That's a seek then a write, with no documented cross-process atomicity, so concurrent lines **may** overwrite each other (**unverified** in practice). Also pass `os.O_BINARY` to avoid `\r\n` translation. `os.replace` on a file another process has open fails with a sharing violation; the `except OSError` skips rotation that time, so the file can grow past the cap until a quiet moment. Treat Windows as best-effort. The plugin bootstrap scripts are bash-only today (`plugins/*/scripts/bootstrap.sh`).

## Q6. Writer cost and failure safety

- **Cost:** ~35 µs per append (open + fstat + write + close, 300-byte error) on macOS 14.6.1 / APFS / Python 3.13 *(measured)*. A whole failing hook run against an unreachable Memgraph takes ~130 ms wall time, dominated by interpreter start and imports *(measured)*. `json` is already imported by the runner (`runner.py:14`). `importlib.metadata.version` lookups for `versions` only run on the failure path. The success path pays **nothing**: nothing is written when nothing fails.
- **Can't raise:** the writer is one function whose whole body is inside `try: … except Exception: return`, including JSON encoding (`default=str` for odd objects), redaction, `makedirs`, open, rotate, write, close. Probe: unwritable parent dir and unwritable target dir both return `False` without raising *(measured)*. Disk full and `EACCES` are both `OSError`, and both are swallowed.
- **Can't block:** no locks. `O_NONBLOCK` makes `open()` on a FIFO planted at the path fail with `ENXIO` instead of waiting for a reader (it has no effect on regular files). The one unbounded wait left is a hard-mounted, unresponsive NFS home, which the config read already shares.
- **Ordering in the handler:** write the log line first, then `_print_response` / strict re-raise, so a crash in response building (row #21) can't lose the record.
- **The real latency risk is the driver, not the log.** Row #9: the Memgraph driver's 30 s default `connection_timeout` equals the hook timeout. In that case the runtime kills the hook, and neither the response nor the log line is written. The writer is only as good as the hook's own deadline.

## Q7. How the viewer and `doctor` read it

**Viewer (Textual, polling every few seconds per #380/#382):**

- `stat()` both files each tick and re-read only when `(size, mtime)` changed. That's cheap enough for the poll interval.
- **Tail-last-N:** seek to `max(0, size − 256 KiB)` in the current file (and `.1` if more is needed), drop the first partial line, and parse each line, **skipping any that fail to parse**. No index or offset file is needed at ≤ 2 MiB total.
- **Since-timestamp:** lines are appended in roughly time order (concurrent writers can swap neighbours by milliseconds), so filter on `at >= since` after parsing the tail. Don't binary-search.
- **Landing-screen health tile:** "last failure N min ago" + count in the last 24 h, grouped by `(stage, connector, error_type)` with count, first/last `at`, and the latest `error`. One row instead of 1,200 identical `connect` lines.
- **Session drill-down:** filter on `session_id` to show failures next to that Session's graph-derived hints ("started, no actions"). Failures whose `session_id` has no Session node are themselves a signal ("captures lost entirely").
- **Project filter (#381):** apply to `cwd`.

**`doctor`:** add one check, `capture-failures`, that resolves the same path and reports `N failures in last 24h; last: <stage>/<connector> <error_type> at <at>` plus the path. Keep it **reporting-only** (it doesn't flip `ok`): doctor's own runtime probe (`cli.py:640-651`) already gives the hard gate on *current* health, and the log is history. A failure an hour ago that has since been fixed shouldn't fail doctor. `--json` gets the grouped summary. Doctor's own probe should **not** write to the log; it reports directly.

## Open items

1. **Hook driver deadline (row #9).** Pick a hook-only `connection_timeout` (3–5 s?) passed through `driver_config`. Without it a blackholed host makes every hook hit the runtime timeout with nothing logged. Note that three connectors connect in sequence (`runner.py:47-60`), so the worst case is a multiple of whatever is chosen. One shared `Memgraph` per hook would avoid that. This is a capture-reliability change, not observability, but the log's coverage depends on it.
2. **Exit-2 hazard (rows #1–2).** Not strictly #378's scope, but it's a live "hook blocks the session" bug on Claude Code `PreToolUse`. Fix it in the same task as the writer.
3. **`AgentLink.emit` isolation (row #13).** A behaviour change: later connectors and events would run after one connector fails. That's the intended semantics ("never lose more than necessary"), but it changes what `--strict` means, which should become "raise an aggregate after all connectors ran".
4. **Detached-child failures (row #18).** Confirm the boundary with #381's rule: failures *before* the Session can be updated go to the log (`stage: embed|reconcile`), and failures recorded on the Session stay in the graph only.
5. **Runtime-killed hooks** (timeouts, row #22) can't be logged from inside. If the viewer's graph hints aren't enough, a "started" marker per hook would need a write on the success path, which is rejected here on cost and size grounds. Revisit only if #383 shows a gap.
6. **Username in `AuthError` messages** (`memgraph.py:139-141`). Redact it too, or treat it as non-secret? Recommended: redact, since it's in hand from config and costs nothing.
7. **Env-var-gated diagnostics** (`AGENT_CONTEXT_GRAPH_<RT>_DEBUG/_STRICT/_CONNECTORS`, `runner.py:89-100,162`) are read at hook runtime. They're flags rather than config values, but they have the same reachability problem ADR 0002 describes. Once the log exists, `_DEBUG` is redundant and can go; `_STRICT`/`_CONNECTORS` are worth a separate look.
8. **Aside (out of scope):** the Codex docs now list a `SessionEnd` event ("runs when a session ends … but not for subagents", default timeout 1 s, max 3 s), while `adapters/codex.py:41-42` says "Codex has no session-end hook". Worth a ticket. A 1–3 s budget is also below the driver timeouts discussed in item 1.
9. **Windows append atomicity** is unverified (Q5). It needs a probe on a Windows box if Windows becomes a supported plugin target.

## Reproduction

All throwaway, not committed:

- **Append probe** (`/tmp/cfl/writer.py`): implements the Q5 algorithm. `bench` times 2,000 appends; `concurrency <path> <n>` spawns 40 Python processes each appending `n` ~3.9 KiB lines, then counts lines that fail `json.loads`; `unwritable` runs the writer against a `chmod 500` dir. macOS 14.6.1, APFS, CPython 3.13.1.
- **Silent failure:** `CONTEXT_GRAPH_CONFIG=/tmp/cfl/none.toml agent-context-graph hook run claude-code --connector actions-graph --memgraph-url bolt://127.0.0.1:1 < payload.json` with payload `{"hook_event_name":"Stop","session_id":"probe"}`. Stdout `{"continue": true}`, 0 bytes stderr, exit 0. With `AGENT_CONTEXT_GRAPH_CLAUDE_CODE_DEBUG=1`, one stderr line: `agent-context-graph claude-code hook skipped: Could not connect to Memgraph database…`. Installed tool was agent-context-graph 0.2.0; the runner's except path (`runner.py:161-167`) is the same on `main`.
- **Connect timeout:** same command with `--memgraph-url bolt://10.255.255.1:7687` (non-routable). `/usr/bin/time` reported `real 30,18`.
- **platformdirs paths:** `platformdirs` 4.12.3 in a scratch venv, `Unix(...)`/`MacOS(...)` `user_state_dir`/`user_log_dir` for appname `context-graph`.
