# Codex CLI: config layers, native memories, MCP, hooks, AGENTS.md (Oct 2026)

Researched 2026-10-10. Sources are official OpenAI docs (developers.openai.com, which now 308-redirects to learn.chatgpt.com/docs), the `openai/codex` repo at HEAD `de8fab6` (2026-10-10), and a read-only look at `~/.codex` on this machine (installed `codex-cli 0.162.0`). Repo paths are relative to `codex-rs/`.

Legend: **[doc]** official docs, **[src]** repo source, **[local]** observed on this machine, **[UNVERIFIED]** inferred and not confirmed by docs or a live run.

---

## 1. User vs project config

- **User config:** `~/.codex/config.toml`. The Codex home is `$CODEX_HOME` when set; when it is set, it must already exist and be a directory. Otherwise it is `~/.codex`. [doc config-basic; src `utils/home-dir/src/lib.rs`]
- **Project config:** `.codex/config.toml`, read in every directory from the project root down to the cwd. The closest file wins. [doc]
- **Precedence**, highest first [doc config-basic]:
  1. CLI flags and `-c/--config`
  2. Project `.codex/config.toml` (trusted projects only)
  3. Profile (`--profile`, `~/.codex/<name>.config.toml`)
  4. User `~/.codex/config.toml`
  5. Cloud-managed defaults
  6. System `/etc/codex/config.toml`
  7. Built-in defaults

  Source numbers [src `config/src/config_layer_source.rs`]: PackagedDefaults -10, Mdm 0, System 10, EnterpriseManaged 15, User 20 (21 with profile), Project 25, SessionFlags 30, legacy managed 40/50.
- **Trust is required.** The project `.codex/` layer loads only for a trusted project. That covers config, `hooks.json`, and rules. In an untrusted project the layer is kept but disabled with the reason "To load project-local config, hooks, and exec policies, add <path> as a trusted project in <user config>". [doc config-basic; src `config/src/loader/mod.rs` `disabled_reason_for_decision`]
- **Where trust is recorded:** the user config, as `[projects."<abs path>"] trust_level = "trusted" | "untrusted"`. [src `config/src/config_toml.rs`; local `~/.codex/config.toml`]
- **Keys a project layer can never set**, even when trusted (`PROJECT_LOCAL_CONFIG_DENYLIST`): `openai_base_url`, `chatgpt_base_url`, `model_provider(s)`, `notify`, `profile(s)`, `otel`, `responses_api_metadata`, `apps_mcp_product_sku`, the realtime base URLs, and a few `features.*` proxy and credential keys. `[memories]`, `[features] memories` and `[mcp_servers]` are **not** on the denylist. [src `config/src/loader/mod.rs:89`]
- **Hook trust is separate.** Each non-managed hook also needs a per-hook trust hash in the user config: `[hooks.state."<file>:<event>:<i>:<j>"] trusted_hash = "sha256:..."`. A hook whose contents change is skipped until it is reviewed again. [doc hooks; local]

## 2. Native memory controls

Feature flag `[features] memories`. The internal id is `Feature::MemoryTool`, stage Stable, **default `false`**. [src `features/src/lib.rs`; doc memories; local `codex features list` shows `memories stable false`]

The `[memories]` table (`MemoriesToml`, `config/src/types.rs:322`):

| Key | Default in src | Notes |
|---|---|---|
| `generate_memories` | `true` | When `false`, new threads are stored with `memory_mode = "disabled"` and never feed generation |
| `use_memories` | `true` | When `false`, the memory read-path developer instructions are not injected |
| `disable_on_external_context` (alias `no_memories_if_mcp_or_web_search`) | `false` | When `true`, a thread that used MCP tools, web search or tool search is marked `"polluted"` and kept out of generation |
| `dedicated_tools` | `false` | Exposes dedicated memory tools. Not in the public docs |
| `version` | `v1` | `v1` stores files in `memories/`, `v2` in `memories_v2/`. Not in the public docs |
| `dual_write` | `false` | Writes both versions. Not in the public docs |
| `max_raw_memories_for_consolidation` | 256 (1–4096) | |
| `max_unused_days` | 30 (0–365) | |
| `max_rollout_age_days` | **10** in src, **30** in docs (0–90) | Docs and source disagree |
| `max_rollouts_per_startup` | **2** in src, **16** in docs (max 128) | Docs and source disagree |
| `min_rollout_idle_hours` | 6 (1–48) | |
| `min_rate_limit_remaining_percent` | 25 | |
| `extract_model`, `consolidation_model` | unset | |

- **Two gates control injection.** Memories are injected only when `features.memories` **and** `memories.use_memories` are both true (`ext/memories/src/extension.rs` `from_config`). Generation also requires the feature flag, a root session that is neither ephemeral nor a sub-agent, and an available state DB. [src `memories/README.md`]
- **User level works.** These are ordinary config keys, so they can be set in `~/.codex/config.toml`. They can also be set in a trusted project config (they are not on the denylist), and per run with `--enable memories` or `-c memories.use_memories=false`. [src; CLI `--help`]
- **Per-chat control** in the TUI: the `/memories` command sets whether the current chat can use memories and whether it can contribute to them. This does not change global settings. [doc memories]
- **Desktop app toggles** under Settings > Personalization: "Enable Codex memories" and "Allow memories from tool-assisted chats". [doc memories]

## 3. Native memory storage and injection

**Location.** Memories live in one global store at `$CODEX_HOME/memories/` (`memories_v2/` for v2). The path comes from `memory_root()` in `memories/read/src/lib.rs` and `MemoryVersion::directory_name` in `protocol/src/memory_version.rs`. There is no per-project directory. Project scoping exists only inside the content: each `MEMORY.md` block carries `applies_to: cwd=...`. [src]

**Pipeline** [src `memories/README.md`; `core/src/memories/`]:

- **Phase 1** runs in the background when a root session starts. It picks idle, eligible past rollouts and asks a model to extract `raw_memory`, `rollout_summary` and `rollout_slug` from each one. Secrets are redacted. The results go to SQLite: `$CODEX_HOME/memories_1.sqlite`, table `stage1_outputs` (`thread_id`, `raw_memory`, `rollout_summary`, `usage_count`, `last_usage`, `selected_for_phase2`, ...), with `jobs` and `consolidation_progress` tables beside it. [local schema confirmed]
- **Phase 2** takes a global lock, syncs files into the memories root, writes a git-style diff, and then runs an internal consolidation sub-agent. That sub-agent has no approvals, no network, and local write access only. It rewrites the consolidated files.

**Files in `~/.codex/memories/`** [src README, `ext/memories/templates/memories/read_path.md`, `memories/write/templates/memories/consolidation.md`, `memories/write/src/storage.rs`]. All are Markdown:

- `memory_summary.md`: the compact, prompt-loaded summary. Its first line must be exactly `v1`, followed by `## User Profile` and further sections.
- `MEMORY.md`: a searchable handbook. It has a strict block format:
  ```
  # Task Group: <cwd / project / workflow family>
  scope: ...
  applies_to: cwd=<path>; reuse_rule=...
  ## Task 1: <description, outcome>
  ### rollout_summary_files
  - rollout_summaries/<file>.md (cwd=..., rollout_path=..., updated_at=..., thread_id=...)
  ### keywords
  - k1, k2, ...
  ## User preferences / ## Reusable knowledge / ## Failures and how to do differently
  ```
- `raw_memories.md`: merged Phase 1 raw memories, ordered by thread id.
- `rollout_summaries/<timestamp>-<shorthash>[-<slug>].md`: one recap per selected rollout. Each recap points at the original session `rollout_path`, which is a jsonl file under `~/.codex/sessions/`.
- `skills/<name>/SKILL.md` (with optional `scripts/`, `examples/`, `templates/`): skills the consolidator promotes.
- `extensions/ad_hoc/notes/<timestamp>-<slug>.md`: the only place the model may write. It does so only when the user explicitly asks to update memory, and Phase 2 folds the note in later.
- `phase2_workspace_diff.md` (transient) and `.git/`: the git baseline that Phase 2 diffs against.

**Injection** [src `ext/memories/src/prompts.rs`, `extension.rs`, `core/src/session/mod.rs`]:

- Codex reads `memory_summary.md` and truncates it to **2,500 tokens** (`MEMORY_TOOL_DEVELOPER_INSTRUCTIONS_SUMMARY_TOKEN_LIMIT`).
- The summary is rendered into the `read_path.md` template as a **developer-policy prompt fragment** (`ContentItemKind "memories.instructions"`). The template has a "## Memory" header, a decision boundary for when to use memory, the file layout, a "quick memory pass" of 4–6 search steps, rules about stale data and verification, and a required `<oai-mem-citation>` block. The summary itself sits between `MEMORY_SUMMARY BEGINS/ENDS` markers.
- `MEMORY.md`, rollout summaries and skills are **not** injected. The model reaches them on demand with normal file and shell tools, unless `dedicated_tools = true`.
- The fragment is part of `build_initial_context_with_world_state`. Compaction calls the same function (`core/src/compact.rs` `build_compaction_replacement_history`), so the memory summary is **re-injected after compaction**. [src; not observed in a live run]

**This machine** [local]:

- `~/.codex/memories/` exists and is empty.
- `memories_1.sqlite` has the schema above and 0 `stage1_outputs` rows.
- `[features]` contains only `hooks = true`, so memories are off.
- `~/.codex/AGENTS.md` exists and is 0 bytes.

## 4. MCP server config: `[mcp_servers.<name>]`

The keys come from `RawMcpServerConfig` in `config/src/mcp_types.rs` and the docs pages `mcp` and `config-reference`:

- **Launch:** `command`, `args`, `env` (a literal map), `env_vars`, `cwd`, `environment_id` / `experimental_environment`.
- **HTTP:** `url`, `bearer_token_env_var`, `http_headers`, `env_http_headers`, `http_headers_helper`, `auth`, `oauth`, `oauth_resource`, `scopes`.
- **Startup and timeouts:** `startup_timeout_sec` / `startup_timeout_ms`, `tool_timeout_sec`, `startup_readiness`.
- **Control:** `enabled` (`false` disables the server), `required` (startup fails if the server can't start), `supports_parallel_tool_calls`, `tool_input_schema_max_bytes` (default 5000), `omit_tools_from`, `default_tools_approval_mode` (`auto` | `prompt` | `writes` | `approve`), and per-tool `[mcp_servers.<name>.tools.<tool>]`.

**`enabled_tools` / `disabled_tools`** [src `codex-mcp/src/tools.rs` `ToolFilter::allows`; doc]:

- `enabled_tools` is an allow-list. When set, only the named tools are registered. When unset, every tool is allowed.
- `disabled_tools` is a deny-list applied **after** `enabled_tools`, so a tool in both lists is denied.
- Names are exact strings: a `HashSet` lookup with no globbing.
- Both keys also exist under `plugins.<plugin>.mcp_servers.<server>.*` for MCP servers that plugins provide.

**`env_vars`** [doc mcp; src `McpServerEnvVar`]:

- Type: `array<string | { name, source = "local" | "remote" }>`.
- It allow-lists environment variables to **forward** from Codex's own environment to a stdio server.
- A plain string means `source = "local"`. `"remote"` applies only to remote stdio servers backed by an executor.
- `env` is different: it sets literal values.

## 5. Hooks

Hook events (`HOOK_EVENT_NAMES` in `hooks/src/lib.rs`): `PreToolUse`, `PermissionRequest`, `PostToolUse`, **`PreCompact`**, **`PostCompact`**, **`SessionStart`**, `SessionEnd`, `UserPromptSubmit`, `SubagentStart`, `SubagentStop`, `Stop`, `Interrupt`.

**Where hooks are configured** [doc hooks]:

- `~/.codex/hooks.json`
- inline `[hooks]` in `~/.codex/config.toml`
- `<repo>/.codex/hooks.json` and `<repo>/.codex/config.toml` (trusted projects only)
- plugin `hooks/hooks.json`

Hooks are on by default (`[features] hooks`, stable, `true`).

**SessionStart**

- The `matcher` is a regex tested against `source`. The documented values are `startup | resume | clear | compact`.
- The source enum also has `fork` (`hooks/src/events/session_start.rs`), which is undocumented. For spawned subagents, `startup` and `fork` dispatch `SubagentStart` instead (`core/src/hook_runtime.rs`). **[UNVERIFIED]**: whether `fork` reaches SessionStart for a root session.

**Does SessionStart fire after compaction? Yes.**

- After compaction the root session queues `SessionStartSource::Compact` (`core/src/session/mod.rs:4231`).
- Matching `SessionStart` hooks then run before the next model request. This includes the continuation after an automatic compaction in the middle of a turn. [doc hooks; src]
- Subagents don't get this: `SessionSource::SubAgent` returns early.

**`additionalContext`: supported.**

- The output shape is `{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"..."}}`. Plain text on stdout also works.
- Either form is added as **developer** context.
- The common output fields `continue`, `stopReason`, `systemMessage` and `suppressOutput` are also accepted.
- `additionalContextLimit` on a handler sets the cap. The default is about 2,500 tokens. [doc hooks; src `hooks/schema/generated/session-start.command.output.schema.json`]

**PreCompact and PostCompact**

- Both exist. The matcher is tested against `trigger`, which is `manual` or `auto`.
- Plain stdout is ignored, and neither event supports `additionalContext`. They accept only the universal fields.
- `continue: false` on PreCompact stops before compacting. On PostCompact it stops after compacting.
- PreCompact does not support `decision: "block"`. [doc hooks; src `hooks/src/schema.rs`, `events/compact.rs`]
- Context therefore has to be re-injected after compaction through `SessionStart` with matcher `compact`, not through PostCompact.
- Caveat [doc]: a `SessionStart` hook can run before the MCP servers are ready.

## 6. User-level AGENTS.md

[doc guides/agents-md; src `codex-home/src/instructions/mod.rs`, `core/src/agents_md.rs`]

**Global file.** `$CODEX_HOME/AGENTS.override.md` if it is non-empty, otherwise `$CODEX_HOME/AGENTS.md`, so by default `~/.codex/AGENTS.md`. Only the first non-empty file is used.

**Project files.**

- Codex walks from the project root (marker `.git` by default; see `project_root_markers`) down to the cwd.
- In each directory it takes one file, checked in this order: `AGENTS.override.md`, `AGENTS.md`, then names from `project_doc_fallback_filenames`.
- Files are concatenated from the root down, so closer files come later.
- The total is capped by `project_doc_max_bytes` (32 KiB default).

**Merge.**

- The global text comes first and the project docs follow.
- In source, the separator is `"\n\n--- project-doc ---\n\n"` (`AGENTS_MD_SEPARATOR`). The docs only say the files are joined "with blank lines".
- **Untrusted project:** the project AGENTS.md files are skipped (`config.active_project.is_untrusted()`), but the global file is still loaded.

**Delivery.**

- The combined text becomes a single contextual user fragment (`ContentItemKind "agents_md.instructions"`) held in the world-state section `AgentsMdState` (`core/src/context/world_state/agents_md.rs`).
- Discovery runs once per run or session. [doc]

**Re-sent after compaction: yes, per source.**

- Compaction rebuilds the initial context from world state (`render_full()` inside `build_initial_context_with_world_state`, called by `build_compaction_replacement_history`), so AGENTS.md comes back.
- When the instructions change mid-session, Codex emits a replacement notice: "These AGENTS.md instructions replace all previously provided AGENTS.md instructions."
- **[UNVERIFIED in a live run]**: only source reading supports this. The docs do not state it explicitly.

---

## Unverified / discrepancies

- **Two memory defaults conflict.** For `max_rollout_age_days` and `max_rollouts_per_startup`, the docs say 30 and 16, but HEAD source constants say 10 and 2 (`config/src/types.rs:55-56`). Trust the source for the build you run, and check the installed 0.162.0 if it matters.
- **Some memory keys are undocumented.** `memories.version`, `dual_write` and `dedicated_tools` are in source only.
- **Post-compaction behaviour was read from source only.** No live session was run, so the claims that AGENTS.md and the memory summary are re-injected after compaction are unconfirmed in practice.
- **The `fork` SessionStart source** behaviour for root sessions is unconfirmed.
- **HEAD may be ahead of the installed CLI.** The repo was read at HEAD (2026-10-10). Local `codex-cli 0.162.0` may lag in small details. The flag list from `codex features list` on this machine does match: `memories` stable/false, `hooks` stable/true.

## Sources

- https://developers.openai.com/codex/memories (.md) — memory overview, storage, `[features] memories`, `memories.*` keys
- https://developers.openai.com/codex/config-reference → https://learn.chatgpt.com/docs/config-file/config-reference — `features.memories`, `memories.*`, `mcp_servers.<id>.*`
- https://developers.openai.com/codex/config-basic — precedence, trust, `~/.codex/config.toml`
- https://developers.openai.com/codex/config-advanced — `CODEX_HOME`, project-instruction discovery
- https://developers.openai.com/codex/hooks — events, matchers, SessionStart `compact`, `additionalContext`, Pre/PostCompact
- https://developers.openai.com/codex/guides/agents-md — global and project AGENTS.md discovery and merge
- https://developers.openai.com/codex/mcp — `env_vars`, `enabled_tools` / `disabled_tools`
- github.com/openai/codex @ de8fab6, under `codex-rs/`:
  - Memories: `memories/README.md`, `config/src/types.rs` (MemoriesToml), `features/src/lib.rs` (MemoryTool), `ext/memories/src/{prompts,extension,lib}.rs`, `ext/memories/templates/memories/read_path.md`, `memories/write/templates/memories/consolidation.md`, `memories/write/src/storage.rs`, `protocol/src/memory_version.rs`
  - Config: `config/src/config_layer_source.rs`, `config/src/loader/mod.rs`, `config/src/mcp_types.rs`, `codex-mcp/src/tools.rs`
  - Hooks: `hooks/src/lib.rs`, `hooks/src/events/{session_start,compact}.rs`, `hooks/src/schema.rs`, `hooks/schema/generated/session-start.command.output.schema.json`, `core/src/hook_runtime.rs`
  - Session and compaction: `core/src/session/mod.rs`, `core/src/compact.rs`
  - AGENTS.md: `core/src/agents_md.rs`, `codex-home/src/instructions/mod.rs`, `core/src/context/world_state/agents_md.rs`
  - Home directory: `utils/home-dir/src/lib.rs`
- Local, read-only: `codex --version` (0.162.0), `codex features list`, `codex mcp add --help`, `~/.codex/{config.toml,hooks.json,AGENTS.md,memories/,memories_1.sqlite schema}`
