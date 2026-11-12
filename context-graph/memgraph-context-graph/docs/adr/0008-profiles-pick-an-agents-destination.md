# Profiles in the one config file pick each agent's destination

ADR 0003 lets `CONTEXT_GRAPH_CONFIG` select which config file a process reads. ADR 0007 hands that path to the processes a hook spawns. Both still assume one destination per file. This ADR adds a second selector, inside the file: which **profile** applies.

## Context

The config file decided where every agent on a machine captured, recalled and wrote memory. `join` and `setup` both wrote to it. So moving one agent to a team's Enterprise Memgraph also redirected capture for every other agent with Context Graph hooks, while `setup <runtime>` switched built-in memory off only in the agent it was run for. Users asked to keep, say, Claude Code on a local Memgraph and Codex on the team's (#692).

## Decision

- **A profile is a named destination in the same file.** `[profiles.<name>]` holds `edition`, and `[profiles.<name>.memgraph]` and `[profiles.<name>.memory]` hold that destination's connection and memory backend. `[agents]` maps a runtime to a profile; an agent not listed uses the top level.
- **A profile replaces the destination whole.** What it leaves out is the default, not the top level's value. An Enterprise profile also drops `identity.user_id`. Either way the login decides the user and the database, and a local `memgraph.database` can't override them.
- **Only the destination differs.** Everything else is shared by all agents, including `[capture]`, `[redact]` and `[recall]`. So pointing an agent elsewhere can't quietly turn redaction off. Any other key in a profile is reported as a config problem and ignored.
- **The process chooses the profile; no value is written to choose it.** The first of these that applies wins:
  1. a command's `--profile`;
  2. `CONTEXT_GRAPH_PROFILE`;
  3. the `[agents]` mapping for the agent the process runs for.

  Hooks know their agent from `hook run <runtime>`. The MCP server knows it from `mcp --agent <runtime>`, which setup writes into each agent's MCP config, or from `CONTEXT_GRAPH_AGENT`, which the plugins set. `doctor --runtime` and `setup <runtime>` use the agent they name.
- **Spawned processes inherit the profile like the path.** `spawn_env()` sets `CONTEXT_GRAPH_PROFILE` to the profile the hook resolved. For the top level it sets the reserved name `default`, so an ambient value can't redirect the child. Like `CONTEXT_GRAPH_CONFIG`, this variable selects *which* values apply, never what they are, so ADR 0002's guarantee holds.
- **Writes follow reads.** While a profile applies, `write_config` and `config set` write the destination keys into that profile, creating it if it is new, and everything else to the top level. So `setup codex --memory-backend context-graph` and `join --agent codex` change only Codex's destination.

## Considered

- **One config file per agent,** with `CONTEXT_GRAPH_CONFIG` written into each hook command. Plugin hook files are fixed and can't vary per user. Inline environment assignments don't work in native Windows hook commands. And shared settings such as redaction would be copied into each file and drift apart.
- **One agent writing to several Memgraphs.** This needs duplicate writes, recall merged across instances, and handling for partial failures. Private memory plus team memory already lives in one Enterprise Memgraph, as a personal space plus shared spaces.

## Consequences

- A file without `[profiles.*]` and `[agents]` resolves exactly as before. No migration.
- Agents on different profiles don't share memory. `setup` names the profile it acts on, and `doctor --runtime` shows each agent's profile.
- `setup` with agents on different profiles refuses and asks for one run per profile, since every step after it reads and writes a single destination.
- The Claude Code and Codex plugins start their MCP server from a fixed `.mcp.json`, and a plugin update doesn't upgrade the installed CLI. An older CLI would refuse `--agent` and the server wouldn't start, so the plugins name their agent in `CONTEXT_GRAPH_AGENT` instead. An older CLI ignores it and serves the top level, as it did before profiles. Like `CONTEXT_GRAPH_PROFILE`, it selects which values apply, never what they are.
