# Recall and memory

[Back to Memgraph Context Graph](../README.md)

## Recall: memory for the harness's model

The hooks write sessions into Memgraph; recall reads them back. With
sessions-graph installed, `memgraph-context-graph mcp` serves a `recall(question)`
tool over stdio MCP. The Codex and Claude Code plugins bundle that server, so
installing a plugin is all it takes; for the other agents, `setup` registers it
in each one's own user-level MCP config (see
[every agent](#every-agent-registration-switch-and-session-start)). The tool returns context, not an answer:
the dated messages and facts from the user's own sessions that match the
question, behind a header with the rules for reading them. The harness's model
answers from those rows.

At session start the hook adds a line to the model's context saying the tool
exists, then where the user left off in this project: the last session's date,
agent, first and last ask and last reply, and the first ask of the two before
it, quoted from captured turns. Sessions from any agent count, so a session in
Grok picks up where one in Codex stopped. It is a pointer, not a recall result:
nothing is ranked against a question, and the model still calls `recall` when
one needs memory. A session's project is its checkout's key (the same one
memory folders use), recorded when it starts. Turn it off with
`memgraph-context-graph config set session_start.resume false`.

A connector may also add a line just before a tool runs
(`GraphConnector.context_before_tool`) — resources-graph's Nudge that what the
tool is about to fetch is already in memory. It is advice only: the call always
proceeds unchanged, and a connector that fails adds nothing. Only Claude Code
and Codex show it (`PreToolUse` `additionalContext` with no permission
decision); the other harnesses' pre-tool hooks can only allow, deny or rewrite
a call, so they get the session-start line alone.

The same search from a shell:

```bash
memgraph-context-graph recall "what did we decide about the deploy window?"
memgraph-context-graph recall --json "..."    # source rows and exchange groups as JSON
```

Whose memory is searched comes from `identity.user_id` in the config file, never
from the tool call, so a model can only read its own user's sessions.
`[recall]` overrides the search's lanes and widths, all optional:

| Key | Default | What it sets |
|---|---|---|
| `embedding_model` | `BAAI/bge-small-en-v1.5` | The model Memgraph embeds with; set with `config set recall.embedding_model` |
| `lanes` | all six | Comma-separated subset of `memories,turns,text,entities,facts,user_facts` |
| `memories_k` | `5` | Memory files shown, found by vector and full text (shown first) |
| `turns_k`, `text_k` | `8`, `8` | Vector/text source widths; exchanges widen enabled pools to at least 24 |
| `entities_k`, `edges_per_entity` | `15`, `6` | Entities matched, and facts read from each |
| `facts_k`, `fact_turns_k` | `15`, `8` | Facts matched and source width; exchanges widen enabled fact-source pools to at least 16 |
| `user_fact_types`, `user_facts_k` | `2`, `30` | Relation types gathered whole, for counting questions |
| `packing` | `exchanges` | Bounded source-only exchanges; `turns` restores separate turn/fact rows |
| `token_budget` | `6000` | Maximum row tokens in exchange mode; rendering instructions are outside the budget |
| `supplemental_search` | `true` | Discovery of complementary problem-resolution exchanges in vector-selected sessions |
| `turn_chars` | `1500` | Characters shown per message |
| `query_instruction` | per model | Prefix embedded with the question; bge's English models get their retrieval instruction, other models none |

Existing config files that omit these keys adopt the defaults above. Exchange
dependencies are installed with `sessions-graph`. Before offline use, populate
and preserve the tokenizer cache; see [exchange recall](../../eval/recall-exchanges.md)
for setup, measured tradeoffs and evaluation limits. The token budget applies
per space, rather than across all fused results.

Recall needs Memgraph with MAGE (`memgraph/memgraph-mage`) for its vector lanes;
without it, it answers from text search and says so. `doctor` checks both
(`embeddings` and `mcp`).

**What the model sees.** A tool result carries the rendered text only. Claude
Code and Codex both show the model a result's `structuredContent` JSON
*instead of* its text when both are present, and the text is the form recall
was benchmarked in; the JSON is what `--json` prints.

## Memory: Context Graph as the harness's memory

Every agent `setup` knows can use Context Graph as its memory. Claude Code and
Codex each keep their own memory, and Context Graph replaces it for a user who
opts in:

```bash
memgraph-context-graph setup claude-code --memory-backend context-graph
memgraph-context-graph setup claude-code --memory-backend native   # hand it back
```

(The interactive `setup` offers the same switch as a checkbox. `uninstall` turns the harness's own memory
back on, since without the plugin it would have none.)

Opting in does three things, in this order:
1. Imports the existing auto-memory files from every project once:
   - `user`/`feedback` memories go to `/memories/<type>/`;
   - `project`/`reference` memories go to that project's folder;
   - `MEMORY.md` is skipped, and anything already in the graph is never overwritten.
2. Sets `autoMemoryEnabled: false` in `~/.claude/settings.json`, the user-level
   settings file, because a plugin can't set it itself. The file is backed up
   once to `settings.json.context-graph.bak`.
3. Records `[memory] backend = "context-graph"`.

For Codex, `setup codex --memory-backend context-graph` does the same against
its global store, with three differences:
- **Import.** `memory_summary.md`, the profile Codex injected each session,
  becomes `/memories/user/codex-memory-summary.md`, and `MEMORY.md` becomes
  `/memories/codex/memory.md`.
- **Switch.** It turns off `features.memories`, `memories.use_memories` and
  `memories.generate_memories` in `~/.codex/config.toml` (or under
  `$CODEX_HOME`), keeping the user's comments and layout. Handing memory back
  restores whatever those keys were before. If the user's own
  `[mcp_servers.context-graph]` narrows tools with `enabled_tools`, `memory` is
  added there.
- **Compaction.** The Codex plugin's `SessionStart` hook also fires after
  compaction (`compact`), so the memory index comes back once the context is
  summarized.

Codex caps hook context at about 2,500 tokens, so pinned files come before the
index and a long index is cut from its tail. See
[the Codex research note](research/2026-10-codex-memory.md).

If the import can't run, for example because Memgraph is down, nothing is
switched.

Nothing writes to both memories: while the backend is `context-graph`, the
graph is the only one. Handing memory back (`--memory-backend native`, or
`uninstall`) first copies the graph's memories into Claude Code's auto memory,
the reverse of the import, so it picks up where the graph left off:
- a project's files go to every Claude Code project folder with that project
  key, `user`/`feedback` ones to every folder with auto memory, each with its
  `<type>_` prefix back;
- the graph's copy wins over the older one there, and each folder's
  `MEMORY.md` is rebuilt;
- files Claude Code has no folder for (a project it never opened, another
  agent's files) stay in the graph only, and the output lists them.

Then it removes the setting. The copy runs only while the backend is
`context-graph`, so handing back twice never overwrites newer files. If
Memgraph is down, `setup` switches nothing; `uninstall` turns the memory back
on anyway, with what it held before, and warns. Copilot CLI and Grok Build
get nothing copied back: their memory is hosted by the agent, or kept in
folders only it can name.

Codex rewrites its own memory files at each consolidation, so handing back
leaves each graph memory as an ad-hoc note in
`~/.codex/memories/extensions/ad_hoc/notes/` (`memories_v2/` under
`[memories] version = "v2"`), the one place Codex takes outside writes. Its
next consolidation folds them into `memory_summary.md` and `MEMORY.md`, in its
own words, so they aren't copied exactly. That runs only while
`features.memories` and `memories.generate_memories` are on; the output says
when they weren't before the switch. A note names its project when the memory
had one, since Codex keeps one store for all projects. Files the import took
from Codex and the graph never changed aren't noted, and a note already left
isn't left again.

`doctor` fails when the backend is `context-graph` but Claude Code's own memory
is still on, including when the project's `.claude/settings.json` or
`.claude/settings.local.json` turns `autoMemoryEnabled` back on, since those
outrank the user-level setting.

With that set, `memgraph-context-graph mcp` also serves a `memory` tool with the
Claude API memory tool's commands (`view`, `create`, `str_replace`, `insert`,
`delete`, `rename` on `/memories`). The model keeps its curated long-term memory there:
- `/memories/` for what applies everywhere;
- `/memories/projects/<project>/` for one codebase.

Every file is a `(:Memory)` node in the user's graph, shared across harnesses
and sessions. Whose memory it is comes from `identity.user_id`, never from
the call. Without the setting the tool isn't listed at all, so the harness's
own memory stays in charge. Harnesses list tools once per session, so the
change shows from the next session.

At session start the hook loads the memory, the way the harness's built-in
memory would have. It adds:
- an index of the root and the current project's files, one line each with
  the file's `description:` frontmatter or first line, up to 200 lines;
- the full text of any file with `pin: true` in its frontmatter;
- the current project's folder.

The index is generated from the graph, so the model never maintains an index
file and it can't go stale. The project folder is keyed by the checkout's git
`origin` remote (e.g. `github.com_memgraph_ai-toolkit`), so every clone of a
repository shares one folder. Without a remote, the repository's directory
name is used. See
[ADR 0006](adr/0006-memory-tool-writes-explicit-memory.md) for why this is
the one write the model makes over MCP.

### Every agent: registration, switch and session start

The other five have no plugin, so `setup <runtime>` registers the
`context-graph` MCP server in the agent's own user-level config: memory belongs
to the user, not to one project. It adds the server's tools to the agent's
persistent allow-list where it keeps one, because a headless run refuses a tool
that needs approval. `--memory-backend context-graph` then turns the agent's
own memory off, and `uninstall` undoes both. Every edited file keeps the rest of
its content and is backed up once to `<name>.context-graph.bak`.

| Agent | Built-in memory, and the switch | Server registered in | Session-start text reaches the model through |
|---|---|---|---|
| Claude Code | auto memory; `autoMemoryEnabled: false` | the plugin | `SessionStart` `additionalContext` |
| Codex | memories; three keys in `config.toml` | the plugin | `SessionStart` `additionalContext` (~2,500-token cap) |
| Copilot CLI | Copilot Memory (on GitHub, not importable); `"memory": false` in `~/.copilot/settings.json` | `~/.copilot/mcp-config.json` (no allow-list: approve once, or `--allow-tool=context-graph`) | `sessionStart` top-level `additionalContext` |
| Cursor | none (Memories was removed) | `~/.cursor/mcp.json`; allow `Mcp(context-graph:*)` in `~/.cursor/cli-config.json` | `sessionStart` `additional_context` (the IDE's agent window can drop it; untested here) |
| OpenCode | none | `mcp.servers` in `~/.config/opencode/opencode.json`, `codemode: false`; an allow rule in `permissions` | the plugin's `context` hook, added to the system prompt of every request |
| Antigravity CLI | none (Knowledge is off in the CLI) | `~/.gemini/config/mcp_config.json`; allow `mcp(context-graph/*)` in `~/.gemini/antigravity-cli/settings.json` | `SessionStart` `injectSteps` system message |
| Grok Build | topic notes under `~/.grok/memory-v2/`, imported to `/memories/grok/`; `[memory_v2] enabled = false` | `~/.grok/config.toml`; allow `MCPTool(context-graph__*)` | the MCP server's `instructions` (`mcp --session-context`): Grok's start hooks can't reach the model |

Antigravity's hooks carry no prompt or reply text, so its turns are read from
the transcript its `Stop` payload points at; that is what gives its sessions a
place in recall and in the next session's "where you left off".

`live-hooks-e2e.sh memory` checks this against each installed agent: a memory
written, one read back in a new session, and the next session's index and last
ask.
