# Command Hook Reference

This reference covers local command-hook setup for source development and per-project experiments. Most users should use runtime plugins and `agent-context-graph bootstrap`.

## Local Codex Config

Local `.codex/` files remain useful for source development and per-project experiments. This repository ignores `.codex/`.

If Memgraph is running locally with defaults:

```bash
agent-context-graph setup codex --project-dir "$PWD" --setup-schema
```

`--setup-schema` connects to Memgraph immediately and runs `SkillGraph().setup()`.

If you need non-default Memgraph connection values:

```bash
agent-context-graph setup codex \
  --project-dir /path/to/your/repo \
  --memgraph-url bolt://localhost:7687 \
  --memgraph-user "" \
  --memgraph-password "" \
  --memgraph-database memgraph \
  --setup-schema
```

The `--memgraph-*` options are used for `--setup-schema`, but they are not written into `.codex/hooks.json`.

For source development in this workspace:

```bash
uv run --package skills-graph --extra agent-context-graph \
  python -m agent_context_graph.cli setup codex \
  --project-dir /path/to/your/repo \
  --memgraph-url bolt://localhost:7687 \
  --setup-schema
```

The command writes local, ignored files:

```text
.codex/config.toml
.codex/hooks.json
```

It refuses to overwrite existing generated files unless you pass `--force`.

The generated hook command does not embed any Memgraph connection values. At runtime, Codex must run with the needed `MEMGRAPH_*` variables in its process environment, or the hooks will use `memgraph-toolbox` defaults.

If Memgraph requires a password, provide `MEMGRAPH_PASSWORD` to the Codex process environment. `.codex/hooks.json` should not contain Memgraph credentials.

To smoke test the generated command, copy the `"command"` value from `.codex/hooks.json` and run:

```bash
printf '{"hook_event_name":"Stop","session_id":"test"}' | COMMAND
```

The expected output is:

```json
{"continue": true}
```

## Manual Setup

Manual setup is mainly useful when developing from source or when you do not want `uv tool install`.

1. Make `skills-graph` able to reach Memgraph, then initialize your skill graph once:

```bash
export MEMGRAPH_URL="bolt://localhost:7687"
export MEMGRAPH_USER=""
export MEMGRAPH_PASSWORD=""
```

```python
from skills_graph import SkillGraph

skills = SkillGraph()
skills.setup()
```

2. Install the hook command and graph connector in the same Python environment:

```bash
python -m venv ~/.venvs/agent-context-graph-hooks
~/.venvs/agent-context-graph-hooks/bin/python -m pip install \
  "agent-context-graph" \
  "skills-graph[agent-context-graph]"
```

For source development in this workspace, use this command instead of the venv binary:

```bash
cd /path/to/ai-toolkit
uv run --package skills-graph --extra agent-context-graph \
  python -m agent_context_graph.cli hook run codex --connector skills-graph
```

3. Generate private Codex hook config in the workspace:

```bash
agent-context-graph hook init codex --connector skills-graph
```

For source development in this workspace:

```bash
uv run --package skills-graph --extra agent-context-graph \
  python -m agent_context_graph.cli hook init codex --connector skills-graph
```

## Codex Hook JSON Shape

The generated config enables Codex hooks and points all supported Codex hook events at a command like:

```bash
agent-context-graph hook run codex --connector skills-graph
```

The resulting `.codex/hooks.json` has this shape:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear",
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "PermissionRequest": [
      {
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ],
    "Stop": [
      {
        "hooks": [{ "type": "command", "command": "COMMAND", "timeout": 30 }]
      }
    ]
  }
}
```

The Codex adapter records `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PermissionRequest`, and `Stop` payloads. MCP tool names such as `mcp__skills__get_skill` are normalized by `skills-graph` to the underlying `get_skill` operation.
