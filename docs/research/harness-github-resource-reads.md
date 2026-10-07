# How GitHub resource reads appear in harness hook payloads

Research for [#455](https://github.com/memgraph/ai-toolkit/issues/455), part of map
[#454](https://github.com/memgraph/ai-toolkit/issues/454) (Resources component). Gathered 2026-10-07.

**Question.** When an agent reads a GitHub resource, which tool calls do it in each harness we support,
what do the hook payloads carry, what reaches our Event Protocol and Actions Graph today, and can a
resource the user handed the agent be told apart from one the agent fetched itself?

## Short answer

- A GitHub read arrives as one of four tool shapes: the harness's **web-fetch tool**, a **shell command**
  (`gh`, `curl`, a clone followed by file reads), a **GitHub MCP tool**, or a **file read of a local
  clone**. Every harness except Antigravity puts the tool name and its input in the pre-tool hook, so the
  resource's identity (URL, or `gh` argv, or MCP args) is always recoverable from the input side.
- The **content** side varies a lot. Claude Code's `WebFetch` gives the hook a small model's answer to
  the agent's question about the page, not the page itself. Shell and MCP output reaches the hook as the
  text the model saw, sometimes truncated. Antigravity hooks carry no output at all, and Codex's hosted web
  search never fires a hook.
- Today all of this lands as an `(:Action:ToolCall)` → `(:Action:ToolResult)` pair, with the input and
  output stored as JSON strings inside `properties` and copied again in `metadata`. Nothing parses a URL or
  `gh` argv into an identity, so no resource has a durable node.
- A **user-handed** resource can be told apart from an **agent fetch**. It arrives in the user-prompt
  hook as raw text (a pasted URL or pasted body), never as a tool call. Cursor also sends an `attachments`
  list. A pasted URL is only a pointer, though: if the agent then fetches it, the content arrives through
  a tool call like any other fetch.

## 1. Which tool calls fetch GitHub content

| Harness | Web fetch tool | Shell (`gh`/`curl`) | GitHub MCP | Notes |
|---|---|---|---|---|
| Claude Code | `WebFetch` `{url, prompt}` | `Bash` `{command}` | `mcp__<server>__<tool>`, e.g. `mcp__github__search_repositories`; remote server at `https://api.githubcopilot.com/mcp/` [mcp] | `WebSearch` `{query, allowed_domains, blocked_domains}` [tools] |
| Codex CLI | none: no fetch handler in `codex-rs/core/src/tools/handlers/` [codex-src] | shell / unified_exec | `mcp__server__tool` [codex-hooks] | Web search is a hosted tool (`web_search = cached\|live\|disabled`) [codex-websearch] |
| Copilot CLI | `web_fetch` | `bash` / `powershell` | built-in `github-mcp-server`, on by default with a CLI subset of tools [copilot-cmd] | PascalCase hook mode renames tools to Claude names (`web_fetch`→`WebFetch`) [copilot-hooks] |
| Cursor | no dedicated fetch tool; "Web" search and Browser [cursor-tools] | Shell | `MCP:<tool_name>` matcher [cursor-hooks] | |
| OpenCode | `webfetch`, returns the **raw** page as text, markdown or html, 5 MB max [opencode-src `tool/webfetch.ts`] | `bash` | `<server>_<tool>` (inferred from the docs' wildcard example, **unverified** in source) | `websearch` only with specific providers [opencode-tools] |
| Antigravity CLI | `read_url_content` | `run_command` | `call_mcp_tool` (payload shape **unverified**) [agy-hooks] | `search_web` |
| Grok Build | `web_fetch`, **off by default** (`GROK_WEB_FETCH`) [grok-settings] | Bash | MCP (`MCPTool` permission filter) [grok-perms] | `web_search` |

A **file read of a cloned repo** shows up everywhere as a plain read tool (`Read`, file-read hooks, and so
on) on a local path. It is a GitHub resource only if a prior clone or `gh repo clone` in the same session
links that path to a repo. The hook itself carries no provenance.

## 2. What the hook payloads carry

### Claude Code (primary target)

Hook events for tools are `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PostToolBatch`,
`PermissionRequest` and `PermissionDenied`, with `tool_name`, `tool_input` and `tool_use_id` throughout.
`PostToolUse` adds `tool_response` (the tool's structured `Output` object) and `duration_ms`. MCP tool
payloads also carry `mcp_server {name, source}` from v2.1.274 [hooks]. Per-tool output shapes come from the
Agent SDK reference [sdk]:

| Tool | `tool_response` shape | Is the content in it? |
|---|---|---|
| `WebFetch` | `{bytes, code, codeText, result, durationMs, url}` | **Usually no.** "For most fetches, it then runs the prompt against the content in a separate model call, and Claude receives the result of that call rather than the raw page." It is "lossy by design" [tools]. `bytes` is the raw size; `result` is the answer. |
| `Bash` | `{stdout, stderr, interrupted, …, persistedOutputPath?, ghRateLimitHint?, gitOperation?}` | Yes, up to the output limit below. |
| MCP | a string or content blocks; `{structuredContent, content}` when the server returns structured output | Yes, up to the MCP limits below. |
| `Read` | `{file: {content, …}}` | Yes (local file). |

Observed in our exploration Memgraph (`scripts/dev-memgraph.sh` instance, 6 sessions, 23 `WebFetch`
calls):

- A fetch of a 524,091-byte Python docs page returned `result` of about 500 characters, a direct
  answer to the agent's `prompt`. That is typical.
- A fetch of `code.claude.com/docs/en/tools-reference` returned `bytes: 77929` and a `result` of roughly the
  same size that was the page's own Markdown. So some fetches **skip** the summarising call. The docs say
  only "most fetches" are summarised and do not say which ones are skipped (**unverified rule**; plausibly
  pages served as Markdown or under a size threshold).
- `tool_input` sometimes carries an `offset` field next to `url`/`prompt`. It is not in the documented
  input schema (**unverified**).

Other WebFetch behaviour [tools, env-vars]:

- HTML is converted to Markdown, and HTTP is upgraded to HTTPS.
- Large pages are truncated to an undocumented fixed character limit before the model call.
- Results are cached for 15 minutes (`CLAUDE_CODE_WEBFETCH_CACHE_TTL_MS`), and downloads have a 5-minute
  deadline.
- A redirect to a different host is not followed: the tool returns text naming the target, and the
  agent must call `WebFetch` again.

Size limits [tools, mcp, env-vars]:

- **Bash**: up to about 30,000 characters inline (`BASH_MAX_OUTPUT_LENGTH`, maximum 150,000). Beyond
  that the model gets a file path plus a 2,000-character preview. On failure it gets a head-and-tail
  excerpt of about 10,000 characters.
- **MCP**: `MAX_MCP_OUTPUT_TOKENS` defaults to 25,000, with a warning above 10,000 tokens. Text over
  50,000 characters is saved to a file unless the tool declares `anthropic/maxResultSizeChars` (up to
  500,000).
- Whether `PostToolUse.tool_response` holds the pre-truncation value or the persisted-file stub is **not
  documented**. `PostToolBatch` explicitly gets "the same content the model receives". The only
  documented truncation inside a hook payload is the `PostToolUseFailure` `error` string.

User side: `UserPromptSubmit.prompt` is the raw prompt text. Collapsed `[Pasted text #N]` blocks arrive
expanded, possibly wrapped in `<pasted_content id=…>`. The hook also fires for scheduled tasks, background
subagents reporting back, and cross-session messages, so not every prompt is a human one. There is no
attachments field. `@path` mentions are presumably literal text in `prompt` (**unverified**).
`UserPromptExpansion` covers `/command` expansion [hooks].

Subagent tool calls fire the same hooks with `agent_id`. Our graph holds `WebFetch` calls made inside
forked subagents' worktrees, so fetches delegated to a subagent are captured too.

### Other harnesses

| Harness | Post-tool output in hook | Truncation | User-prompt hook |
|---|---|---|---|
| Codex | `tool_response`. Shell output is truncated by the model-output policy; there is no PostToolUse at all while a process keeps running in the background. MCP output is the full `CallToolResult`, untruncated [codex-src `core/src/tools/context.rs`]. Hosted `WebSearch` "do[es] not use the local function-tool hook path" [codex-hooks]. | Shell: `TruncationPolicy` budget (default **unverified**) | `UserPromptSubmit.prompt`, raw, no attachments [codex-src `hooks/src/schema.rs`] |
| Copilot CLI | `toolResult {resultType, textResultForLlm}`, which is the text the model saw [copilot-hooks] | `textResultForLlm` truncation undocumented | `userPromptSubmitted.prompt` (raw), plus `userPromptTransformed.transformedPrompt`. `@file` and `--attachment` exist but no hook field carries them. |
| Cursor | `postToolUse.tool_output` (JSON string), `afterShellExecution.output` (full terminal output), `afterMCPExecution.result_json` [cursor-hooks] | undocumented | `beforeSubmitPrompt {prompt, attachments: [{type: file\|rule, file_path}]}` |
| OpenCode | `tool.execute.after` → `{title, output, metadata}` [opencode-src `packages/plugin/src/index.ts`] | Built-in output truncated to 2,000 lines / 50 KB before the hook; full output goes to `outputPath` [opencode-src `tool/truncate.ts`] | `chat.message` → `{message, parts}`; file and attachment parts arrive in `parts` |
| Antigravity CLI | **None.** PostToolUse has only `toolCall {name, args}`, `stepIdx` and `error`; output exists only in `transcriptPath` [agy-hooks] | n/a | **None.** The prompt is in no payload; only the transcript has it. |
| Grok Build | No output field documented [grok-hooks]; our adapter nevertheless reads typed results (`output_for_prompt`, `FileContent.content`) seen in real payloads | Bash `output_byte_limit` default 20,000 bytes [grok-settings] | `UserPromptSubmit` field list undocumented; our adapter reads `prompt` |

## 3. What reaches the Event Protocol and Actions Graph today

Code path, on `main` at cc20d88:

1. Each adapter (`context-graph/agent-context-graph/src/agent_context_graph/adapters/*.py`) maps:
   - the pre-tool hook to a `ToolStartEvent(tool_name, tool_input, tool_use_id)`;
   - the post-tool hook to a `ToolEndEvent(result, is_error)`;
   - the user-prompt hook to a `MessageEvent(role="user", content=prompt)`.

   Every adapter also copies a fixed set of raw payload keys into `event.metadata` (`_spec.py`
   `_metadata_from_payload`). For Claude Code that set includes `tool_input` **and the whole
   `tool_response`**.
2. `result` unwrapping is per harness:
   - Claude Code unwraps Bash `stdout`/`stderr` and Read `file.content`. All other responses, **including
     WebFetch**, are kept whole as a dict (`claude_code.py` `_tool_result_text`).
   - Copilot, Codex and Cursor go through `extract_tool_result`, which takes `content`, `llmContent` or
     `textResultForLlm`.
   - OpenCode joins the text parts of `content`.
   - Grok takes `output_for_prompt` / `FileContent.content`.
   - Antigravity sets no result.
3. `ActionsGraphConnector` (`context-graph/actions-graph/src/actions_graph/connector.py`) writes:
   - `(:Action:ToolCall {tool_name, properties: json{tool_input, tool_use_id, mcp_server}, metadata: json})`
     with `-[:USED_TOOL]->(:Tool {name, is_mcp, mcp_server})`. `is_mcp` and `mcp_server` come from the
     `mcp__` prefix (`models.py` `ToolCall.__post_init__`), so only Claude/Codex-style names are
     recognised.
   - `(:Action:ToolResult {properties: json{content, …}, metadata: json})`, linked by `PARENT_OF`.
     A non-string, non-block-list result is stored as `str(value)`: a Python repr, not JSON. This is how a
     WebFetch result ends up, as `"{'bytes': …, 'result': …}"`.
   - Prompts as `(:Action:Message:UserMessage {text})`, where `text` is a plain property that the
     full-text and vector indexes cover.
4. Session Reconciliation (`context-graph/sessions-graph/src/sessions_graph/reconciliation.py`) turns
   each `ToolCall` (input JSON), `ToolResult` (content) and `Message` into text for entity extraction,
   **truncated to 8,000 characters** (`MAX_RECONCILABLE_CHARS`). A large `gh issue view` or MCP result is
   cut before chunking.

Observed consequences:

- **The URL survives** as `tool_input.url` (WebFetch), inside a shell `command` string (`gh issue view
  455`, `curl …`), or as MCP args. It is queryable only by `CONTAINS` on a JSON string.
- **The content is stored at most twice and sometimes not at all.** It is stored in `properties.content`
  and again in `metadata.tool_response` (Claude Code), and both are capped by what the harness truncated.
- In the exploration data, `Bash` results are stored as `"{'stdout': …}"` reprs. That data was written
  by the released plugin (0.1.12); `main`'s Bash unwrapping is newer and not yet in that build.

## 4. User-handed vs agent-fetched

| Signal | How it arrives | Distinguishable? |
|---|---|---|
| Pasted URL in prompt | user-prompt hook → `UserMessage.text` | Yes. It is never a tool call. It is only a pointer; any content comes from a later agent fetch of the same URL. |
| Pasted issue/PR body | user-prompt hook (Claude Code expands `[Pasted text #N]`, possibly in `<pasted_content>`) | Yes as "user-handed text". Nothing says which resource it came from unless the user included a URL. |
| `@file` / attachment | Cursor `attachments[]`; OpenCode `parts`; Claude Code / Copilot: literal text, no field (**unverified** for Claude Code) | Partly. Only Cursor and OpenCode give a structured field. |
| Agent fetch | pre/post tool hooks → `ToolCall`/`ToolResult` | Yes. |
| Agent fetch prompted by a pasted URL | `UserMessage` containing URL *U*, then `ToolCall` whose input contains *U* in the same session | Yes, by matching URLs within the session. Nothing records it today. |
| Prompt that is not from a human | Claude Code `UserPromptSubmit` also fires for scheduled tasks, subagent reports and cross-session messages | **Ambiguous.** A URL in such a "prompt" may be machine-handed. |
| Subagent fetch | tool hooks carry `agent_id`; the connector nests the call under `(:Agent)` | Yes. It counts as an agent fetch. |
| Antigravity | no prompt hook | **No.** The user side is invisible without reading the transcript. |

## 5. Implications for the map

**Capture model (#457).**

- Resolve identity from the **tool input**, never the output:
  - `WebFetch`/`web_fetch`/`read_url_content` → `url`;
  - shell → parse `gh issue|pr|repo|api …` and `curl <github url>`;
  - GitHub MCP → tool name plus `{owner, repo, number}` args;
  - user prompt → URLs in `prompt` (plus Cursor `attachments`).

  This works the same way in every harness except Antigravity's prompt side.
- **Do not treat a hook result as the resource's content.** Claude Code `WebFetch` (the most common
  path) is usually a query-specific summary. Shell and MCP output can be truncated (Codex policy,
  OpenCode 50 KB, Grok 20 KB, Claude Code ~30K chars / 25K tokens). Antigravity has no output at all. A
  Resource needs **content of its own, fetched by us** (e.g. from the GitHub API keyed on the identity),
  and the hook event becomes a pointer of the form "session S read resource R via tool T at time t".
- Keep provenance: `HANDED_BY_USER` (prompt) vs `FETCHED` (tool call), with the `ToolCall` as evidence.
  A fetch whose URL appeared earlier in a user prompt is both.
- WebFetch's `prompt` field is a free signal of *why* the agent read the resource. Worth keeping on the
  read edge.

**Cache-hit decision (#459).**

- A cache hit cannot replay a `WebFetch` result: that answer is specific to the prompt. It can only
  serve our own stored content, or a fresh answer computed over it.
- Hooks cannot substitute a tool's output before the model sees it in a way that works across
  harnesses. Claude Code `PreToolUse` can deny or modify input, and `additionalContext` is capped at
  10,000 characters, so serving 500 issues from memory has to go through **recall** (a read tool such as
  the MCP `recall` server), not through intercepting the fetch.
- The harness already caches: Claude Code holds `WebFetch` results for 15 minutes and Codex web search
  defaults to `cached`. Our cache only adds value across sessions, users, and longer windows.

## Open questions surfaced

1. Does Claude Code's `PostToolUse.tool_response` hold the full Bash/MCP output or the persisted-file
   stub once the inline limit is exceeded? This decides whether hook output is ever usable as content.
   It is testable with a large `gh api` call against the dev Memgraph.
2. Which `WebFetch` fetches skip summarisation (observed for a Markdown docs page)? Undocumented.
3. How does Grok Build's real `PostToolUse` payload look? Its docs list no output field, but our adapter
   handles typed results.
4. Should the Claude Code adapter unwrap `WebFetch.result` (and stop duplicating `tool_response` into
   `metadata`)? It is a small change that is independent of the Resources design.
5. Should Antigravity capture read `transcriptPath` to recover prompts and outputs?

## Sources

- [tools] Claude Code tools reference — https://code.claude.com/docs/en/tools-reference
- [hooks] Claude Code hooks — https://code.claude.com/docs/en/hooks
- [mcp] Claude Code MCP — https://code.claude.com/docs/en/mcp
- [env-vars] Claude Code environment variables — https://code.claude.com/docs/en/env-vars
- [sdk] Agent SDK TypeScript reference (tool output types) — https://code.claude.com/docs/en/agent-sdk/typescript
- [codex-hooks] Codex hooks — https://learn.chatgpt.com/docs/hooks (redirect target of developers.openai.com/codex/hooks)
- [codex-websearch] Codex web search — https://learn.chatgpt.com/docs/web-search.md
- [codex-src] Codex source — https://github.com/openai/codex (`codex-rs/core/src/tools/context.rs`, `codex-rs/hooks/src/schema.rs`, `codex-rs/core/src/tools/handlers/`)
- [copilot-hooks] Copilot CLI hooks reference — https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-hooks-reference
- [copilot-cmd] Copilot CLI command reference — https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
- [cursor-hooks] Cursor hooks — https://cursor.com/docs/agent/hooks
- [cursor-tools] Cursor tools — https://cursor.com/docs/agent/tools
- [opencode-tools] OpenCode tools — https://opencode.ai/docs/tools/ ; plugins — https://opencode.ai/docs/plugins/
- [opencode-src] OpenCode source — https://github.com/anomalyco/opencode (branch `dev`)
- [agy-hooks] Antigravity hooks — https://antigravity.google/docs/hooks
- [grok-hooks] Grok Build hooks — https://docs.x.ai/build/features/hooks.md
- [grok-settings] Grok Build settings — https://docs.x.ai/build/settings/reference.md
- [grok-perms] Grok Build permissions — https://docs.x.ai/build/features/permissions.md
- This repo, `main` @ cc20d88: `context-graph/agent-context-graph/src/agent_context_graph/{events.py,adapters/_spec.py,adapters/claude_code.py,adapters/*.py}`, `context-graph/actions-graph/src/actions_graph/{connector.py,core.py,models.py}`, `context-graph/sessions-graph/src/sessions_graph/reconciliation.py`
- Exploration Memgraph (`scripts/dev-memgraph.sh`, bolt://localhost:7687), queried 2026-10-07: 606 ToolCalls (493 Bash, 23 WebFetch, 0 MCP), 68 UserMessages
